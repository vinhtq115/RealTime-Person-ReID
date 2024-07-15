# Pipeline for multi-camera tracking (MCT).
from copy import deepcopy
from itertools import combinations
import multiprocessing as mp
from pathlib import Path
import pickle
import queue
from threading import Thread
import time
from typing import List, Tuple
from itertools import combinations

import cv2
from natsort import natsorted
import numpy as np
from sklearn.metrics import pairwise_distances

from modules.models.bot_sort import STrack
from modules.signaling import DoneSignal, discard_signal
from modules.models.yolo import KEYPOINTS_MAP, SKELETON_MAP
from utils.average_meter import AverageMeter


def remove_section_with_inf(x: np.ndarray, idxs: np.ndarray):
    valid_idxs = idxs[~np.isin(x[idxs[:, 0], idxs[:, 1]], np.inf)]

    return valid_idxs


def cosine_distance_with_mask(feat_data: np.ndarray, camera_idxs: np.ndarray) -> np.ndarray:
    """Calculate upper triangle cosine distance between pairwise targets.
    Values of all targets in same cameras are masked to infinity.

    Args:
        feat_data (np.ndarray): Features of the online targets. Shape: (num_targets, feat_dim)
        camera_idxs (np.ndarray): Camera indices of each online target. Shape: (num_targets,)

    Returns:
        np.ndarray: Upper triangle of cosine distance matrix.
    """
    # Calculate cosine distances between pairwise targets
    distances = np.full((len(feat_data), len(feat_data)), np.inf)
    cam_from_to = {}  # {cam_idx: (from_idx, to_idx)}
    for cam in np.unique(camera_idxs):
        cam_idxs = np.where(camera_idxs == cam)[0]
        cam_from_to[cam] = cam_idxs[0], cam_idxs[-1] + 1

    for i, j in combinations(cam_from_to.keys(), 2):
        start_idx1, end_idx1 = cam_from_to[i]
        start_idx2, end_idx2 = cam_from_to[j]
        distances[start_idx1:end_idx1, start_idx2:end_idx2] = pairwise_distances(feat_data[start_idx1:end_idx1], feat_data[start_idx2:end_idx2], metric="cosine")

    return distances


def get_sorted_idx(score_matrix: np.ndarray, threshold: float = 0.4) -> np.ndarray:
    """Get the sorted indices of the score matrix.
    Optionally excludes pairs with score higher than threshold.

    Args:
        score_matrix (np.ndarray): Score matrix
        threshold (float, optional): Threshold value. Defaults to 0.4.

    Returns:
        np.ndarray: Sorted indices of the score matrix
    """
    sorted_idx = np.vstack(np.unravel_index(np.argsort(score_matrix, axis=None), score_matrix.shape)).T
    sorted_idx = remove_section_with_inf(score_matrix, sorted_idx)

    if threshold is not None:
        sorted_idx_score = score_matrix[sorted_idx[:, 0], sorted_idx[:, 1]]
        sorted_idx = sorted_idx[sorted_idx_score < threshold]
    return sorted_idx


def assign_global_ids(
    sorted_pairs_idx: np.ndarray,
    score_matrix: np.ndarray,
    camera_idxs: np.ndarray,
    sct_true_idxs: np.ndarray,
    world_coords: np.ndarray,
    previous_global_id_to_cam_sct: dict = None,
    previous_cam_sct_to_global_id: dict = None,
    previous_global_id_counter: int = None,
    last_seen_global_id: dict = None,
    feat_distance_threshold: float = 0.4,
    world_distance_threshold: float = 1000,
) -> Tuple[dict, dict, int, dict]:
    """Assign global IDs to online tracklets.
    If previous global ID assignment is provided, update it.

    Args:
        sorted_pairs_idx (np.ndarray): List of sorted pairs of indices. Shape: (num_pairs, 2)
        score_matrix (np.ndarray): Cosine distance matrix. Shape: (num_targets, num_targets)
        camera_idxs (np.ndarray): Camera index of each target. Shape: (num_targets,)
        sct_true_idxs (np.ndarray): SCT ID of each target in its camera. Shape: (num_targets,)
        world_coords (np.ndarray): World coordinates of each target. Shape: (num_targets, 2)
        previous_global_id_to_cam_sct (dict, optional): Previous global ID to camera and SCT tracklet assigment pairs. Defaults to None.
        previous_cam_sct_to_global_id (dict, optional): Reverse mapping of previous_global_id_to_cam_sct. Defaults to None.
        previous_global_id_counter (int, optional): Previous global ID counter to start from. If not provided, new global ID will start from 0. Defaults to None.
        last_seen_global_id (dict, optional): Map global ID to the last seen timestamp. Defaults to None.
        feat_distance_threshold (float, optional): Feature distance threshold. Defaults to 0.4.
        world_distance_threshold (float, optional): World distance threshold. Defaults to 1500.

    Returns:
        Tuple[dict, dict, int, dict]: Latest global ID to cameras and SCT tracklets assigment, reverse mapping of it, latest global ID counter, and updated last seen timestamp.
    """
    current_timestamp = time.time()
    unmatched_set = set([i for i in range(len(world_coords))])
    mask_valid = np.ones_like(score_matrix, dtype=bool)

    # Use previous state if available
    global_id_to_cam_sct = deepcopy(previous_global_id_to_cam_sct)
    cam_sct_to_global_id = deepcopy(previous_cam_sct_to_global_id)
    global_id_counter = deepcopy(previous_global_id_counter)

    # For caching
    cam_from_to = {}  # {cam_idx: (from_idx, to_idx)}
    for cam in np.unique(camera_idxs):
        cam_idxs = np.where(camera_idxs == cam)[0]
        cam_from_to[cam] = cam_idxs[0], cam_idxs[-1] + 1
    cam_scts_to_index = {}  # {(cam_idx, sct_true_idx): index}
    for i in range(len(sct_true_idxs)):
        cam = camera_idxs[i]
        sct = sct_true_idxs[i]
        cam_scts_to_index[(cam, sct)] = i

    # results = []

    for p1, p2 in sorted_pairs_idx:
        sct1, cam1 = sct_true_idxs[p1], camera_idxs[p1]
        sct2, cam2 = sct_true_idxs[p2], camera_idxs[p2]
        score = score_matrix[p1, p2]

        if mask_valid[p1, p2] == False:
            # This pair has been masked as invalid because either
            # p1 or p2 has been matched prior to this pair.
            # print(f"Ignored {cam1}_{sct1} ({p1}) & {cam2}_{sct2} ({p2}) because one of these has been matched before.")
            # results.append((f"{p1} ({cam1}_{sct1})", f"{p2} ({cam2}_{sct2})", f"{score}", "Invalid"))
            continue
        elif score >= feat_distance_threshold:
            # This pair and remaining ones have score higher than threshold.
            # print(f"Remaining pairs have score higher than 0.4. Stopping at {i}.")
            # results.append((f"{p1} ({cam1}_{sct1})", f"{p2} ({cam2}_{sct2})", f"{score}", "Reached threshold"))
            break

        # Calculate world distance and check
        world_distance = np.linalg.norm(world_coords[p1] - world_coords[p2])
        if world_distance >= world_distance_threshold:
            # print(f"Ignored {cam1}_{sct1} ({p1}) and {cam2}_{sct2} ({p2}). World distance: ({world_distance:.2f}).")
            # results.append((f"{p1} ({cam1}_{sct1})", f"{p2} ({cam2}_{sct2})", f"{score}", "Too far"))
            continue

        # Check if any of the SCT IDs have been assigned to a global ID
        global_id_1 = cam_sct_to_global_id.get((cam1, sct1), None)
        if global_id_1 is not None and global_id_1 not in global_id_to_cam_sct:
            # This global ID has been removed/merged before
            global_id_1 = None
        global_id_2 = cam_sct_to_global_id.get((cam2, sct2), None)
        if global_id_2 is not None and global_id_2 not in global_id_to_cam_sct:
            # This global ID has been removed/merged before
            global_id_2 = None
        to_be_matched = False  # Set this to True to trigger matching

        if global_id_1 is None and global_id_2 is None:
            # Both SCT IDs are not assigned to any global ID
            # Create new global id
            global_id_1 = global_id_counter
            global_id_2 = global_id_counter
            global_id_counter += 1

            # Update global_id_to_cam_scts
            # global_id_to_cam_scts[global_id_1] = {cam1: {sct1}, cam2: {sct2}}
            global_id_to_cam_sct[global_id_1] = {cam1: sct1, cam2: sct2}
            cam_sct_to_global_id[(cam1, sct1)] = global_id_1
            cam_sct_to_global_id[(cam2, sct2)] = global_id_1
            last_seen_global_id[global_id_1] = current_timestamp
            to_be_matched = True
            # print(f"Assigned {cam1}_{sct1} ({p1}) and {cam2}_{sct2} ({p2}) to new global id {global_id_1}.")
            # results.append((f"{p1} ({cam1}_{sct1})", f"{p2} ({cam2}_{sct2})", f"{score}", "New GID"))
        elif global_id_1 is not None and global_id_2 is None:
            # P1 was matched before. Assign p2 to same global id as p1.
            # Check if global ID of 1 is in the camera of 2
            sct_of_gid_1_in_cam_2 = global_id_to_cam_sct[global_id_1].get(cam2, None)
            if sct_of_gid_1_in_cam_2 is not None and (cam2, sct_of_gid_1_in_cam_2) in cam_scts_to_index:
                # Compare and unlink if needed
                already_linked = cam_scts_to_index[(cam2, sct_of_gid_1_in_cam_2)]
                _p1, _p2 = min(p1, already_linked), max(p1, already_linked)
                already_linked_score = score_matrix[_p1, _p2]
                if already_linked_score < score:
                    # The existing pair has lower score. Ignore the new pair.
                    # print(f"Ignored {cam1}_{sct1} ({p1}) and {cam2}_{sct2} ({p2}). Existing pair has higher score ({already_linked_score:.4f}).")
                    # results.append((f"{p1} ({cam1}_{sct1})", f"{p2} ({cam2}_{sct2})", f"{score}", "Existing pair has higher score"))
                    continue

                # Unassign the existing pair
                _r = global_id_to_cam_sct[global_id_1].pop(cam2)
                cam_sct_to_global_id.pop((cam2, _r), None)
                unmatched_set.add(already_linked)
                # print(f"Unlinked {already_linked} ({cam2}_{_r}) from global id {global_id_1}.")

            if sct_of_gid_1_in_cam_2 is not None:  # Not visible
                # Remove the tracklet from the global id
                global_id_to_cam_sct[global_id_1].pop(cam2, None)
                cam_sct_to_global_id.pop((cam2, sct_of_gid_1_in_cam_2), None)

            # TODO: perhaps check if p2's distance to other tracklets with
            # the same global id as p1.
            global_id_2 = global_id_1
            global_id_to_cam_sct[global_id_1][cam2] = sct2
            cam_sct_to_global_id[(cam2, sct2)] = global_id_2
            last_seen_global_id[global_id_1] = current_timestamp
            to_be_matched = True
            # print(f"Assigned {cam2}_{sct2} ({p2}) to global id of p1 {global_id_1}.")
            # results.append((f"{p1} ({cam1}_{sct1})", f"{p2} ({cam2}_{sct2})", f"{score}", "P2->1"))
        elif global_id_1 is None and global_id_2 is not None:
            # P2 was matched before. Assign p1 to same global id as p2.
            # Check if global ID of 2 is in the camera of 1
            sct_of_gid_2_in_cam_1 = global_id_to_cam_sct[global_id_2].get(cam1, None)
            if sct_of_gid_2_in_cam_1 is not None and (cam1, sct_of_gid_2_in_cam_1) in cam_scts_to_index:
                # Compare and unlink if needed
                already_linked = cam_scts_to_index[(cam1, sct_of_gid_2_in_cam_1)]
                _p1, _p2 = min(p2, already_linked), max(p2, already_linked)
                already_linked_score = score_matrix[_p1, _p2]
                if already_linked_score < score:
                    # The existing pair has lower score. Ignore the new pair.
                    # print(f"Ignored {cam1}_{sct1} ({p1}) and {cam2}_{sct2} ({p2}). Existing pair has higher score ({already_linked_score:.4f}).")
                    # results.append((f"{p1} ({cam1}_{sct1})", f"{p2} ({cam2}_{sct2})", f"{score}", "Existing pair has higher score"))
                    continue

                # Unassign the existing pair
                _r = global_id_to_cam_sct[global_id_2].pop(cam1)
                cam_sct_to_global_id.pop((cam1, _r), None)
                unmatched_set.add(already_linked)
                # print(f"Unlinked {already_linked} ({cam1}_{_r}) from global id {global_id_2}.")

            if sct_of_gid_2_in_cam_1 is not None:  # Not visible
                # Remove the tracklet from the global id
                global_id_to_cam_sct[global_id_2].pop(cam1, None)
                cam_sct_to_global_id.pop((cam1, sct_of_gid_2_in_cam_1), None)

            # TODO: perhaps check if p1's distance to other tracklets with
            # the same global id as p2.
            global_id_1 = global_id_2
            global_id_to_cam_sct[global_id_1][cam1] = sct1
            cam_sct_to_global_id[(cam1, sct1)] = global_id_1
            last_seen_global_id[global_id_1] = current_timestamp
            to_be_matched = True
            # print(f"Assigned {cam1}_{sct1} ({p1}) to global id of p2 {global_id_2}.")
            # results.append((f"{p1} ({cam1}_{sct1})", f"{p2} ({cam2}_{sct2})", f"{score}", "P1->2"))
        elif global_id_1 == global_id_2:
            # Both tracklets have been assigned to the same global id previously
            last_seen_global_id[global_id_1] = current_timestamp
            to_be_matched = True
            # print(f"Both {cam1}_{sct1} ({p1}) and {cam2}_{sct2} ({p2}) have been assigned to global id {global_id_1}.")
            # results.append((f"{p1} ({cam1}_{sct1})", f"{p2} ({cam2}_{sct2})", f"{score}", "Same"))
        else:
            # Both tracklets have been assigned to different global ids previously
            # Decide whether to merge or not

            # Check if global ID of 1 is not in the camera of 2 and vice versa
            # If so, they are not mergeable
            gid_1_visible_cams = set()
            gid_1_invisible_cams = set()
            for cam in global_id_to_cam_sct[global_id_1].keys():
                if (cam, global_id_to_cam_sct[global_id_1][cam]) in cam_scts_to_index:
                    gid_1_visible_cams.add(cam)
                else:
                    gid_1_invisible_cams.add(cam)
            gid_2_visible_cams = set()
            gid_2_invisible_cams = set()
            for cam in global_id_to_cam_sct[global_id_2].keys():
                if (cam, global_id_to_cam_sct[global_id_2][cam]) in cam_scts_to_index:
                    gid_2_visible_cams.add(cam)
                else:
                    gid_2_invisible_cams.add(cam)
            if len(gid_1_visible_cams.intersection(gid_2_visible_cams)) > 0:
                # Global ID 1 and Global ID 2 have tracklets in common cameras
                # print(f"Global ID 1 ({global_id_1}) and Global ID 2 ({global_id_2}) have tracklets in common cameras.")
                # results.append((f"{p1} ({cam1}_{sct1})", f"{p2} ({cam2}_{sct2})", f"{score}", "GID1 and GID2 have common cameras"))
                continue

            for cam in gid_1_invisible_cams:
                sct = global_id_to_cam_sct[global_id_1].pop(cam)
                cam_sct_to_global_id.pop((cam, sct), None)
            for cam in gid_2_invisible_cams:
                sct = global_id_to_cam_sct[global_id_2].pop(cam)
                cam_sct_to_global_id.pop((cam, sct), None)

            # if global_id_to_cam_sct[global_id_1].get(cam2, None) is not None and \
            #     (cam2, global_id_to_cam_sct[global_id_1][cam2]) in cam_scts_to_index:
            #     # print(f"Global ID 1 ({global_id_1}) is in camera 2 ({cam2}).")
            #     # results.append((f"{p1} ({cam1}_{sct1})", f"{p2} ({cam2}_{sct2})", f"{score}", "GID1 in cam2"))
            #     continue
            # if global_id_to_cam_sct[global_id_2].get(cam1, None) is not None and \
            #     (cam1, global_id_to_cam_sct[global_id_2][cam1]) in cam_scts_to_index:
            #     # print(f"Global ID 2 ({global_id_2}) is in camera 1 ({cam1}).")
            #     # results.append((f"{p1} ({cam1}_{sct1})", f"{p2} ({cam2}_{sct2})", f"{score}", "GID2 in cam1"))
            #     continue

            # Gather tracklets of each global id
            gid_1_cam_scts, gid_2_cam_scts = [], []
            for cam, sct in global_id_to_cam_sct[global_id_1].items():
                # If the tracklet is not in the current frame, ignore it.
                if (cam, sct) in cam_scts_to_index:
                    gid_1_cam_scts.append(cam_scts_to_index[(cam, sct)])
            for cam, sct in global_id_to_cam_sct[global_id_2].items(): # BUG
                # If the tracklet is not in the current frame, ignore it.
                if (cam, sct) in cam_scts_to_index:
                    gid_2_cam_scts.append(cam_scts_to_index[(cam, sct)])

            # In case gid_1_cam_scts or gid_2_cam_scts is empty
            # Global ID has no tracklet in the current frame
            # Not enough clue to merge.
            if len(gid_1_cam_scts) == 0 or len(gid_2_cam_scts) == 0:
                # print(f"Global ID 1 ({global_id_1}) or Global ID 2 ({global_id_2}) has no tracklet in the current frame.")
                # results.append((f"{p1} ({cam1}_{sct1})", f"{p2} ({cam2}_{sct2})", f"{score}", "No tracklet visible for merge"))
                continue

            # Find the shortest feature and world distance
            # between tracklets of both global ids
            shortest_world_dist = np.inf
            shortest_feat_dist = np.inf
            for gid_1_pid in gid_1_cam_scts:
                _p1 = np.array([gid_1_pid] * len(gid_2_cam_scts))
                _gid_2_cam_scts = np.array(gid_2_cam_scts)
                avg_world_dist = np.sum(np.linalg.norm(world_coords[_p1] - world_coords[_gid_2_cam_scts], axis=1)) / len(_gid_2_cam_scts)
                avg_cosine_dist = np.sum(score_matrix[_p1, _gid_2_cam_scts], axis=0) / len(_gid_2_cam_scts)
                shortest_world_dist = min(shortest_world_dist, avg_world_dist)
                shortest_feat_dist = min(shortest_feat_dist, avg_cosine_dist)

            for gid_2_pid in gid_2_cam_scts:
                _p2 = np.array([gid_2_pid] * len(gid_1_cam_scts))
                _gid_1_cam_scts = np.array(gid_1_cam_scts)
                avg_world_dist = np.sum(np.linalg.norm(world_coords[_p2] - world_coords[_gid_1_cam_scts], axis=1)) / len(_gid_1_cam_scts)
                avg_cosine_dist = np.sum(score_matrix[_p2, _gid_1_cam_scts], axis=0) / len(_gid_1_cam_scts)
                shortest_world_dist = min(shortest_world_dist, avg_world_dist)
                shortest_feat_dist = min(shortest_feat_dist, avg_cosine_dist)

            # Decide whether to merge or not
            mergeable = shortest_world_dist < world_distance_threshold and shortest_feat_dist < feat_distance_threshold
            if not mergeable:
                # Do not merge
                last_seen_global_id[global_id_1] = current_timestamp
                last_seen_global_id[global_id_2] = current_timestamp
                # print(f"Both {cam1}_{sct1} ({p1}) and {cam2}_{sct2} ({p2}) have been assigned to different global ids {global_id_1} and {global_id_2} but they are not mergeable. Shortest feat. distance: {shortest_feat_dist:.4f}. Shortest world distance: {shortest_world_dist:.2f}.")
                # results.append((f"{p1} ({cam1}_{sct1})", f"{p2} ({cam2}_{sct2})", f"{score}", f"{global_id_1} and {global_id_2} not mergeable"))
                continue

            # Merge two global ids using the smaller global id
            min_global_id = min(global_id_1, global_id_2)
            max_global_id = max(global_id_1, global_id_2)
            for cam, sct in global_id_to_cam_sct[max_global_id].items():
                # Add tracklets from max_global_id to min_global_id
                global_id_to_cam_sct[min_global_id][cam] = sct
                # Replace old global id with new global id
                cam_sct_to_global_id[(cam, sct)] = min_global_id

            # Remove the max global id
            del global_id_to_cam_sct[max_global_id]
            del last_seen_global_id[max_global_id]

            # Get all tracklets that have been assigned to the merged global id
            merged_cam_scts = list(global_id_to_cam_sct[min_global_id].items())
            # Need to keep the order of camera ascending, else masking will behave incorrectly
            merged_cam_scts.sort(key=lambda x: x[0])
            # Ignore out-of-sight tracklets
            merged_cam_scts = [cam_sct for cam_sct in merged_cam_scts if (cam_sct[0], cam_sct[1]) in cam_scts_to_index]

            # Mask the merged tracklets
            for _p1, _p2 in combinations(merged_cam_scts, 2):
                _cam1_from_to = cam_from_to[_p1[0]]
                _cam2_from_to = cam_from_to[_p2[0]]
                _p1 = cam_scts_to_index[_p1]
                _p2 = cam_scts_to_index[_p2]
                to_be_masked = np.concatenate([
                    np.mgrid[_p1:_p1+1, _cam2_from_to[0]:_cam2_from_to[1]].squeeze(axis=1).T,
                    np.mgrid[_cam1_from_to[0]:_cam1_from_to[1], _p2:_p2+1].squeeze(axis=2).T
                ])
                mask_valid[to_be_masked[:, 0], to_be_masked[:, 1]] = False
            last_seen_global_id[min_global_id] = current_timestamp
            to_be_matched = True
            # print(f"Merged global id {max_global_id} to {min_global_id}.")
            # results.append((f"{p1} ({cam1}_{sct1})", f"{p2} ({cam2}_{sct2})", f"{score}", f"{max_global_id} -> {min_global_id}"))

        # Match the pair
        if to_be_matched:
            # Mask the row and column of the matched pairs
            cam1_from_to = cam_from_to[cam1]
            cam2_from_to = cam_from_to[cam2]
            to_be_masked = np.concatenate([
                np.mgrid[p1:p1+1, cam2_from_to[0]:cam2_from_to[1]].squeeze(axis=1).T,
                np.mgrid[cam1_from_to[0]:cam1_from_to[1], p2:p2+1].squeeze(axis=2).T
            ])
            mask_valid[to_be_masked[:, 0], to_be_masked[:, 1]] = False
            unmatched_set.discard(p1)
            unmatched_set.discard(p2)

    # For unmatched tracklets, assign each of them to a new global id
    for i in unmatched_set:
        cam = camera_idxs[i]
        sct = sct_true_idxs[i]
        if (cam, sct) in cam_sct_to_global_id:
            # This tracklet has been assigned to a global id before
            global_id = cam_sct_to_global_id[(cam, sct)]
            last_seen_global_id[global_id] = current_timestamp
            continue

        # Assign new global id
        global_id = global_id_counter
        global_id_counter += 1
        # print("Increased global id counter to", global_id_counter)

        global_id_to_cam_sct[global_id] = {cam: sct}
        cam_sct_to_global_id[(cam, sct)] = global_id
        last_seen_global_id[global_id] = current_timestamp
        # print(f"Assigned unmatched {cam}_{sct} ({i}) to new global id {global_id}.")

    # Refresh global_id_to_cam_sct
    global_id_to_cam_sct_refreshed = {}
    for (cam, sct), gid in cam_sct_to_global_id.items():
        if global_id_to_cam_sct_refreshed.get(gid, None) is None:
            global_id_to_cam_sct_refreshed[gid] = {}
        global_id_to_cam_sct_refreshed[gid][cam] = sct

    gid_cams = {}  # {global_id: {camera: True/False}}
    for (cam, sct), gid in cam_sct_to_global_id.items():
        if gid_cams.get(gid, None) is None:
            gid_cams[gid] = {}
        if gid_cams[gid].get(cam, None) is None:
            gid_cams[gid][cam] = True
        else:
            # print(cam_sct_to_global_id)
            raise ValueError(f"Global ID {gid} has been assigned to camera {cam} more than once.")

    return global_id_to_cam_sct_refreshed, cam_sct_to_global_id, global_id_counter, last_seen_global_id


class PipelineMCT(mp.Process):
    def __init__(self,
                 mct_config: dict,
                 camera_configs: dict,
                 data_queues: List[mp.Queue],
                 raw_data_dir: Path,
                 start_event: mp.Event,
                 global_kill: mp.Event,
                 screenshot_dir: Path):
        super().__init__()

        self.mct_config = mct_config
        self.camera_configs = camera_configs
        self.camera_names = natsorted(camera_configs.keys())
        self.data_queues = data_queues
        self.start_event = start_event
        self.raw_data_dir = raw_data_dir
        self.cam_latest_data = []
        self.appearance_threshold = mct_config["appearance_threshold"]
        self.distance_threshold = mct_config["distance_threshold"]
        self.screenshot_dir = screenshot_dir

        # Overlay
        self.room_height = mct_config["floor_height"]
        self.room_width = mct_config["floor_width"]
        self.top_down_file = Path(mct_config["top_down_file"])
        self.cam_color = []
        for cam_name in self.camera_names:
            self.cam_color.append(camera_configs[cam_name]["visualize_color"])

        for _ in range(len(camera_configs)):
            self.cam_latest_data.append({
                "frame": None,
                "frame_idx": -1,
                "online_targets": [],
                "removed_targets": set(),
                "terminated": False,
            })
        self.global_kill = global_kill

    def single_camera_queue_query(self, cam_idx: int):
        """Thread for getting data from SCT.

        Args:
            cam_idx (int): Camera index. For accessing data queue.
        """
        cam_name = self.camera_names[cam_idx]
        data_queue = self.data_queues[cam_idx]
        cam_data_dir: Path = self.raw_data_dir / cam_name
        if not cam_data_dir.exists():
            cam_data_dir.mkdir(parents=True)

        while True:
            try:
                # Get data from each process
                data = data_queue.get(block=True, timeout=0.02)
                if isinstance(data, DoneSignal):
                    # Camera has finished processing. No more data will be sent to it
                    self.cam_latest_data[cam_idx]["terminated"] = True
                    break

                frame, frame_idx, online_targets, removed_targets = data
                online_targets: List[STrack]
                removed_targets: List[STrack]

                # Read latest data from SCT and update for MCT
                self.cam_latest_data[cam_idx]["frame"] = frame
                self.cam_latest_data[cam_idx]["frame_idx"] = frame_idx
                self.cam_latest_data[cam_idx]["online_targets"] = online_targets
                self.cam_latest_data[cam_idx]["removed_targets"] = removed_targets

                # # For dumping data to disk
                # tracking_data = {}
                # for target in online_targets:
                #     x1, y1, x2, y2 = [int(i) for i in target.tlbr]
                #     x1 = max(0, x1)
                #     y1 = max(0, y1)
                #     x2 = min(frame.shape[1], x2)
                #     y2 = min(frame.shape[0], y2)
                #     tracking_data[target.track_id] = {
                #         "bbox": (x1, y1, x2, y2),
                #         "score": target.score,
                #         "pose": target.pose,
                #         "smooth_feat": target.smooth_feat,
                #         "curr_feat": target.curr_feat,
                #         "world_coord": target.world_coord
                #     }

                # # Save data
                # data_path = cam_data_dir / f"{frame_idx:06d}.pkl"
                # with open(data_path.as_posix(), "wb") as file:
                #     pickle.dump(tracking_data, file, protocol=5)

            except queue.Empty:
                pass

        print(f"[MCT] Detected done signal from camera {cam_name}. Terminating.")

    def get_latest_data(self):
        feat_data = []  # Features of the online targets. Shape: (num_targets, feat_dim)
        camera_idxs = []  # Camera indices of the online targets. Shape: (num_targets,)
        sct_true_idxs = []  # True SCT IDs of the online targets. Shape: (num_targets,)
        image_coords = []  # Image coordinates of the online targets. Shape: (num_targets, 4)
        world_coords = []  # World coordinates of the online targets. Shape: (num_targets, 2)
        frames = []  # Frames of cameras. List of num_cameras cameras
        poses = []
        removed_targets = []

        for cam_idx in range(len(self.cam_latest_data)):
            terminated = self.cam_latest_data[cam_idx]["terminated"]

            # Copy the (reference) data in case it gets changed by other running threads
            curr_online_targets: List[STrack] = self.cam_latest_data[cam_idx]["online_targets"]
            removed_targets.append(self.cam_latest_data[cam_idx]["removed_targets"])

            frame = self.cam_latest_data[cam_idx]["frame"]
            if frame is None:
                frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            frames.append(frame)

            if not terminated:
                for target in curr_online_targets:
                    feat_data.append(target.smooth_feat)  # or target.curr_feat
                    camera_idxs.append(cam_idx)
                    image_coords.append(target.tlbr)
                    world_coords.append(target.world_coord)
                    sct_true_idxs.append(target.track_id)
                    poses.append(target.pose)

        return feat_data, camera_idxs, sct_true_idxs, image_coords, world_coords, poses, frames, removed_targets

    def run(self):
        # Discard SIGTERM and SIGKILL signals
        discard_signal()

        # Initialize variables
        global_id_to_cam_sct = {}  # {global_id: {camera: sct_id}}
        cam_sct_to_global_id = {}  # {(camera, sct_id): global_id}
        global_id_counter = 0
        global_id_last_seen = {}  # Global ID -> last seen time. For garbage collection purpose.

        # Load 2D map image
        map_2d = cv2.imread(self.top_down_file.as_posix(), cv2.IMREAD_COLOR)
        cv2.namedWindow("Top down", cv2.WINDOW_GUI_NORMAL)

        # Start threads to refresh data from camera
        camera_data_getter_threads = [Thread(target=self.single_camera_queue_query, args=(idx,)) for idx in range(len(self.camera_names))]
        for thread in camera_data_getter_threads:
            thread.start()

        # Initialize windows for visualization
        window_pos = {
            "c200_1": (0, 28),
            "c200_2": (0, 560),
            "zed2i_1": (960, 28),
            "zed2i_2": (960, 560),
        }
        for cam_name in self.camera_names:
            cv2.namedWindow(cam_name, cv2.WINDOW_GUI_NORMAL)
            cv2.moveWindow(cam_name, *window_pos[cam_name])
            cv2.resizeWindow(cam_name, 960, 460)

        # Wait for the start event (all cameras are ready)
        while not self.start_event.is_set():
            self.start_event.wait()

        tick_runtime_meter = AverageMeter()
        while not all([self.cam_latest_data[cam_idx]["terminated"] for cam_idx in range(len(self.cam_latest_data))]):
            current_time = time.time()

            feat_data, camera_idxs, sct_true_idxs, image_coords, world_coords, poses, frames, removed_targets = self.get_latest_data()
            camera_idxs = np.array(camera_idxs)
            world_coords = np.array(world_coords)
            poses = np.array(poses)

            ### Remove offline targets ###
            for cam_idx, removed_targets in enumerate(removed_targets):
                for target_id in removed_targets:
                    global_id = cam_sct_to_global_id.pop((cam_idx, target_id), None)

                    # If None, the tracklet has been previously removed
                    if global_id is None:
                        continue

                    if global_id in global_id_to_cam_sct and global_id_to_cam_sct[global_id].get(cam_idx, None) == target_id:
                        del global_id_to_cam_sct[global_id][cam_idx]
                        if len(global_id_to_cam_sct[global_id]) == 0:
                            # Remove the global id if it has no tracklets
                            global_id_to_cam_sct.pop(global_id)
                            global_id_last_seen.pop(global_id)

            # with open(f"/mnt/8tb_ext4/recording_2024_07_01/{tick_runtime_meter.count:06d}.pkl", "wb") as f:
            #     pickle.dump({
            #         "feat_data": feat_data,
            #         "camera_idxs": camera_idxs,
            #         "sct_true_idxs": sct_true_idxs,
            #         "image_coords": image_coords,
            #         "world_coords": world_coords,
            #         "poses": poses,
            #         "frames": frames,
            #         "global_id_to_cam_scts": global_id_to_cam_scts,
            #         "cam_scts_to_global_id": cam_scts_to_global_id,
            #         "global_id_counter": global_id_counter,
            #         "global_id_last_seen": global_id_last_seen,
            #     }, f, protocol=5)

            if len(feat_data) == 0:  # No online targets
                time.sleep(0.01)
                continue

            # Calculate cosine distances between pairwise targets
            distances = cosine_distance_with_mask(feat_data, camera_idxs)

            # Sort the distances by score ascending to get candidate pairs.
            sorted_idx = get_sorted_idx(distances)  # Using default threshold

            in_case_of_crash = {
                "feat_data": feat_data,
                "camera_idxs": camera_idxs,
                "sct_true_idxs": sct_true_idxs,
                "image_coords": image_coords,
                "world_coords": world_coords,
                "poses": poses,
                "frames": frames,
                "sorted_idx": sorted_idx,
                "distances": distances,
                "global_id_to_cam_sct": deepcopy(global_id_to_cam_sct),
                "cam_sct_to_global_id": deepcopy(cam_sct_to_global_id),
                "global_id_counter": deepcopy(global_id_counter),
                "global_id_last_seen": deepcopy(global_id_last_seen),
            }
            # Assign global IDs to the online tracklets
            try:
                global_id_to_cam_sct, cam_sct_to_global_id, global_id_counter, global_id_last_seen = assign_global_ids(
                    sorted_idx, distances, camera_idxs, sct_true_idxs, world_coords,
                    global_id_to_cam_sct, cam_sct_to_global_id, global_id_counter, global_id_last_seen,
                    self.appearance_threshold, self.distance_threshold
                )
            except Exception as e:
                print(e)

                # Save data
                with open(f"error.pkl", "wb") as f:
                    pickle.dump(in_case_of_crash, f, protocol=5)
                raise e
            ### End of Multi-camera Tracklet Assignment ###

            # Visualization
            ready_for_visualization = [True for _ in range(len(frames))]
            for idx, frame in enumerate(frames):
                if frame is None:
                    ready_for_visualization[idx] = False

            if all(ready_for_visualization):
                out_frames = [frame.copy() for frame in frames]
                overlay_map = map_2d.copy()
                for p in range(len(image_coords)):
                    x1, y1, x2, y2 = [int(i) for i in image_coords[p]]
                    cam = camera_idxs[p]
                    sct_id = sct_true_idxs[p]
                    g_id = cam_sct_to_global_id[(cam, sct_id)]
                    cv2.rectangle(out_frames[cam], (x1, y1), (x1+28, y1+28), (0, 0, 0), -1)
                    cv2.rectangle(out_frames[cam], (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(out_frames[cam], f"{sct_id}/{g_id}", (x1 + 4, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Draw pose
                    pose = poses[p]
                    for j in range(17):
                        x, y, score = pose[j]
                        if score < 0.5:
                            continue

                        x = int(x)
                        y = int(y)
                        color = KEYPOINTS_MAP[j]
                        cv2.circle(out_frames[cam], (x, y), 3, color, -1)

                        if j in SKELETON_MAP:
                            for link in SKELETON_MAP[j]:
                                link_id, link_color = link
                                x_link, y_link, score_link = pose[link_id]
                                if score_link < 0.5:
                                    continue
                                x_link = int(x_link)
                                y_link = int(y_link)
                                cv2.line(out_frames[cam], (x, y), (x_link, y_link), link_color, 2)

                    # Draw on the overlay map
                    x_w, y_w = world_coords[p]
                    if np.isnan(x_w) or np.isnan(y_w):
                        continue
                    x_w_scale = int(x_w / self.room_width * overlay_map.shape[1])
                    y_w_scale = int(y_w / self.room_height * overlay_map.shape[0])
                    cv2.circle(overlay_map, (x_w_scale, y_w_scale), 5, self.cam_color[cam], -1)
                    cv2.putText(overlay_map, f"[{cam}]{sct_id}/{g_id}", (x_w_scale, y_w_scale - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.cam_color[cam], 2)

                for cam_idx, cam_name in enumerate(self.camera_names):
                    cv2.imshow(cam_name, out_frames[cam_idx])
                cv2.imshow("Top down", overlay_map)

                k = cv2.waitKey(1)
                if k == ord('s'):
                    ss_time = time.strftime("%Y%m%d-%H%M%S")
                    # Save screenshots
                    for cam_idx, cam_name in enumerate(self.camera_names):
                        file_path = self.screenshot_dir / f"{ss_time} {cam_name}.png"
                        cv2.imwrite(file_path.as_posix(), out_frames[cam_idx])
                    file_path = self.screenshot_dir / f"{ss_time} Top down.png"
                    cv2.imwrite(file_path.as_posix(), overlay_map)

            tick_runtime_meter.update(time.time() - current_time)

        print("[MCT] Received global kill.")
        print(f"[MCT] Average time per tick: {tick_runtime_meter.avg:.4f} seconds")
        print(f"[MCT] Average ticks/second: {1 / tick_runtime_meter.avg:.2f} ticks/second")

        for idx, thread in enumerate(camera_data_getter_threads):
            print(f"[MCT] Joining thread of camera {self.camera_names[idx]}")
            thread.join()
