# Pipeline for single camera tracking (SCT).
# Includes human detection, pose estimation and single camera tracking.
import logging
import multiprocessing as mp
from pathlib import Path
import time
from typing import List, Tuple

import cv2
import numpy as np

from modules.cameras import get_camera_instance
from modules.models.bot_sort import BoTSORT, STrack
from modules.models.yolo import YOLOWrapper
from modules.pose_utils import *
from modules.signaling import DoneSignal, discard_signal


def transform(H: np.ndarray, cam_coord: np.ndarray, rounding=True):
    """Used for transforming image coordinate to world coordinate.

    Args:
        H (np.ndarray): Homography matrix
        cam_coord (np.ndarray): Image coordinates. Format: [x, y]
        rounding (bool, optional): Round to nearest integer. Defaults to True.

    Returns:
        np.ndarray: World coordinates. Format: [x, y]
    """
    # Return world coordinates
    cam_coord = np.array([cam_coord[0], cam_coord[1], 1]).reshape(3, 1)
    estimated_position = np.matmul(H, cam_coord)
    estimated_position = estimated_position / estimated_position[2, :]
    estimated_position = estimated_position[:2, :]
    if rounding:
        estimated_position = np.round(estimated_position, 0).astype(int)

    return estimated_position.T.squeeze()


def filter_detections(det_results: np.ndarray, keypoints_results: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Filter detection and keypoints results.

    Args:
        det_results (np.ndarray): Detection results. Shape: (N, 5) or (N, 6)
        keypoints_results (np.ndarray): Keypoints results. Shape: (N, 17, 3)

    Returns:
        List[np.ndarray, np.ndarray]: Filtered detection and keypoints results.
    """
    heads = keypoints_results[:, 0:5, 2] >= 0.5
    n_heads_detected = np.sum(heads, axis=1)
    n_heads_detected_mask = n_heads_detected >= 1

    rests = keypoints_results[:, 5:, 2] >= 0.5
    rests_detected = np.sum(rests, axis=1)
    rests_detected_mask = rests_detected >= 4

    mask = np.logical_and(n_heads_detected_mask, rests_detected_mask)

    return det_results[mask], keypoints_results[mask]


def estimate_position(kpts_int: np.ndarray, kpts_score: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray | None:
    """Estimate position in image coordinate from keypoints.

    Args:
        kpts_int (np.ndarray): Keypoint coordinates. Shape: (17, 2)
        kpts_score (np.ndarray): Keypoint scores. Shape: (17,)
        bbox (Tuple[int, int, int, int]): Bounding box coordinates. Format: (x1, y1, x2, y2)

    Returns:
        np.ndarray | None: Return coordinate if possible, else None.
    """
    x1, y1, x2, y2 = bbox

    pose_status = get_pose_status(kpts_int, kpts_score)
    if pose_status == 1:  # Standing
        if np.all(kpts_score[15:17] >= 0.5):
            # If both ankles are visible, use the average of them
            return np.mean(kpts_int[15:17], axis=0)
        else:
            # If both ankles are not visible, infer the leg direction from hip
            # and use thigh length to estimate the ankle position.
            return get_ankle_from_body_and_knee_standing(kpts_int, kpts_score)
    elif pose_status == 0:  # Sitting
        return get_sitting_position(kpts_int, kpts_score)
    else:  # Unknown
         # Check if any leg is visible and pose is in standing position.
        # This is likely to happen only when a person is moving out-of-frame on the top.
        estimated_leg_position = get_position_if_leg_visible_and_standing(kpts_int, kpts_score)
        if estimated_leg_position is not None:
            # At least one leg is visible and pose is in standing position.
            return estimated_leg_position
        else:
            shoulder_visible = np.any(kpts_score[5:7] >= 0.5)
            hip_visible = np.any(kpts_score[11:13] >= 0.5)
            elbow_visible = np.any(kpts_score[7:9] >= 0.5)

            if shoulder_visible:
                # Shoulder is visible
                if hip_visible:
                    # Shoulder and hips are available.
                    return estimate_position_from_shoulder_hip(kpts_int, kpts_score)
                elif elbow_visible:
                    # Shoulder and elbow are available.
                    return estimate_position_from_shoulder_elbow(kpts_int, kpts_score)
                elif head_visible(kpts_score):
                    # Shoulder and head are available. This case is likely to be wrong.
                    # Center of bbounging box is used as x-coordinate.
                    return estimate_position_from_head_shoulder(kpts_int, kpts_score, x1, x2)
                else:
                    # This case should be ignored.
                    return None
            elif head_visible(kpts_score):
                # Wild guess. Likely to be wrong.
                # Assume that this is caused because the shoulder is not visible.
                # In this case, the bottom of the bounding box will be used in place of the shoulder.
                return estimate_position_from_head_bbox(kpts_int, kpts_score, x1, x2, y2)
            else:
                # This case should be ignored.
                return None


class PipelineSCT(mp.Process):
    def __init__(self,
                 model_config: dict,
                 camera_config: dict,
                 visualize_dir: Path,
                 data_queue: mp.Queue,
                 ready: mp.Event,
                 start_event: mp.Event,
                 global_kill: mp.Event):
        super().__init__()

        self.global_kill = global_kill
        self.ready = ready
        self.start_event = start_event
        self.data_queue = data_queue

        self.logger = logging.getLogger("SCT")

        self.yolo_config = model_config["yolo"]
        self.bot_sort_config = model_config["bot_sort"]
        self.camera_config = camera_config

        if self.camera_config.pop("dump", False):
            camera_config["dump_path"] = visualize_dir / (camera_config["name"] + "_clean.mkv")
        else:
            camera_config["dump_path"] = None

    def run(self):
        # Discard SIGTERM and SIGKILL signals
        discard_signal()

        # Initialize camera
        camera = get_camera_instance(self.camera_config)
        name = camera.name
        self.name = name
        camera_H = camera.homography_matrix

        # Start camera capture
        camera.start_capture()
        self.logger.info(f"[{name}] Camera started capturing.")

        # Initialize YOLO detector
        det_track = YOLOWrapper(**self.yolo_config)

        # Initialize tracker
        self.tracker = BoTSORT(
            self.bot_sort_config,
            int(camera.video.get(cv2.CAP_PROP_FPS))
        )

        # Warmup
        frame, _ = camera.capture(warmup=True)
        det_track.detect(frame)

        self.ready.set()

        # Wait for start event
        while not self.start_event.is_set():
            self.start_event.wait()

        self.logger.info(f"[{name}] Started.")

        # Main loop
        run_times = []
        while not self.global_kill.is_set():
            start_time = time.time()
            frame, idx = camera.capture()
            if frame is None:
                if camera.error:
                    self.logger.error(f"[{name}] Something is wrong with the camera. Terminating...")
                    self.global_kill.set()
                break

            # Detection
            det_results, keypoints_results = det_track.detect(frame)

            # # Filter detections
            # if len(det_results) > 0:
            #     det_results, keypoints_results = filter_detections(det_results, keypoints_results)

            # Update BoT-SORT with human detection results.
            # Get tracked objects in return.
            online_targets: List[STrack] = self.tracker.update(det_results, frame, keypoints_results)
            removed_targets = set([t.track_id for t in self.tracker.removed_stracks])

            # Estimate real-world coordinates of online targets from keypoints
            # if homography matrix is provided.
            if camera_H is not None:
                for target in online_targets:
                    if target.pose is None:
                        continue

                    kpts_int = target.pose[:, :2].astype(int)  # (17, 2)
                    kpts_score = target.pose[:, 2]  # (17,)
                    x1, y1, x2, y2 = target.tlbr

                    image_coord = estimate_position(kpts_int, kpts_score, (x1, y1, x2, y2))
                    if image_coord is None:
                        target.world_coord = np.array([np.nan, np.nan])
                    else:
                        target.world_coord = transform(camera_H, image_coord)

            # Send data to main process
            self.data_queue.put([frame, idx, online_targets, removed_targets])

            run_times.append(time.time() - start_time)

            if idx % 900 == 0:
                self.logger.info(f"[{name}] Frame {idx}. Tracked/Lost/Removed: {len(self.tracker.tracked_stracks)}/{len(self.tracker.lost_stracks)}/{len(self.tracker.removed_stracks)}")

        self.logger.info(f"[{name}] Releasing...")
        camera.release()
        self.logger.info(f"[{name}] Camera shut down.")
        self.logger.info(f"[{name}] Average FPS: {1. / np.average(run_times[1:]):.2f}")

        # Signal that the camera has finished processing
        self.data_queue.put(DoneSignal())

        self.ready.clear()
