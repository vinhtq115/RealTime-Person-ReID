from collections import deque, defaultdict
import logging
import multiprocessing as mp
import os
import pickle
import queue
from threading import Thread
import time
from typing import Dict, Set

import annoy
import cv2
import numpy as np

from modules.models.reid.semi_wrapper import SEMIWrapper
from modules.signaling import DoneSignal, discard_signal
from utils.average_meter import AverageMeter


class PipelineReID(mp.Process):
    def __init__(self,
                 model_config: dict,
                 save_dir: str,
                 mct_to_reid_data_queue: mp.Queue,
                 reid_to_mct_data_queue: mp.Queue,
                 global_kill: mp.Event):
        super().__init__()

        self.global_kill = global_kill
        self.data_queue = mct_to_reid_data_queue
        self.reply_queue = reid_to_mct_data_queue
        self.model_config = model_config["semi"]
        self.save_dir = save_dir

        self.terminated = False

        # Data from Re-ID
        self.global_id_to_cam_sct = {}
        self.cam_sct_to_global_id = {}
        self.camera_idxs = []
        self.sct_true_idxs = []

        self.cam_sct_to_frames = None
        self.cam_sct_to_poses = None
        self.cam_sct_to_last_seen = None

        self.global_id_to_name = {}

        self.logger = logging.getLogger("Re-ID")

    def data_queue_query(self):
        while not self.terminated:
            try:
                data = self.data_queue.get(block=True, timeout=0.01)
                if isinstance(data, DoneSignal) or self.terminated:
                    # Terminate
                    break

                self.global_id_to_cam_sct = data["global_id_to_cam_sct"]
                self.cam_sct_to_global_id = data["cam_sct_to_global_id"]
                self.camera_idxs = data["camera_idxs"]
                self.sct_true_idxs = data["sct_true_idxs"]
                frame_data = data["frame_data"]
                pose_data = data["pose_data"]
                frame_idxs = data["frame_idxs"]

                # Release frames of removed targets
                removed_targets_all_cams = data["removed_targets"]
                for cam_idx, removed_targets in enumerate(removed_targets_all_cams):
                    for target_id in removed_targets:
                        self.cam_sct_to_frames.pop((cam_idx, target_id), None)
                        self.cam_sct_to_poses.pop((cam_idx, target_id), None)
                        self.cam_sct_to_last_seen.pop((cam_idx, target_id), None)

                # Update frames and poses
                for i in range(len(frame_data)):
                    cam = self.camera_idxs[i]
                    sct = self.sct_true_idxs[i]
                    frame_idx = frame_idxs[i]
                    if self.cam_sct_to_last_seen.get((cam, sct), None) is not None and frame_idx - self.cam_sct_to_last_seen[(cam, sct)] < 4:
                        continue  # Skip if frame is too close to the last seen frame

                    frame = frame_data[i]
                    pose = pose_data[i]
                    self.cam_sct_to_frames[(cam, sct)].append(frame)
                    self.cam_sct_to_poses[(cam, sct)].append(pose)
                    self.cam_sct_to_last_seen[(cam, sct)] = frame_idx

            except queue.Empty:
                pass

        self.logger.info(f"[Re-ID] Data refresh thread received DoneSignal.")

    def get_unidentified(self) -> Set[int]:
        """Get global IDs that have not been identified yet."""
        current_global_ids = set(self.cam_sct_to_global_id.values())
        already_ided = set(self.global_id_to_name.keys())
        return current_global_ids - already_ided

    def reid(self, global_id: int, semi_model: SEMIWrapper, annoy_index: annoy.AnnoyIndex, identity_info: Dict[int, str]):
        if self.global_id_to_cam_sct.get(global_id, None) is None:
            # Global ID not found
            return

        cam_scts = self.global_id_to_cam_sct[global_id]

        # Decide which camera to use for Re-ID
        best_cam_sct = None
        best_cam_sct_kpts_score = 0
        for cam, sct in cam_scts.items():
            if len(self.cam_sct_to_frames[(cam, sct)]) < 8:
                # Not enough frames
                continue

            if best_cam_sct is None:
                best_cam_sct = (cam, sct)
                best_cam_sct_kpts_score = 0
                for frame_pose in self.cam_sct_to_poses[(cam, sct)]:
                    pose_score = frame_pose[:, 2]
                    # Count number of keypoints with score > 0.5
                    best_cam_sct_kpts_score += np.sum(pose_score > 0.5)
            else:
                # Compare with best cam_sct
                cam_sct = (cam, sct)
                cam_sct_kpts_score = 0
                for frame_pose in self.cam_sct_to_poses[cam_sct]:
                    pose_score = frame_pose[:, 2]
                    cam_sct_kpts_score += np.sum(pose_score > 0.5)

                if cam_sct_kpts_score > best_cam_sct_kpts_score:
                    best_cam_sct = cam_sct
                    best_cam_sct_kpts_score = cam_sct_kpts_score

        if best_cam_sct is None:
            # No camera has enough frames
            return

        # Re-ID
        frames = list(self.cam_sct_to_frames[best_cam_sct])
        embeddings = semi_model(frames).squeeze()  # 2048-d vector
        save_dir = os.path.join(self.save_dir, str(global_id))
        os.makedirs(save_dir, exist_ok=True)
        for frame_idx, frame in enumerate(frames):
            cv2.imwrite(f"{save_dir}/{frame_idx}.jpg", frame)

        # Compare
        nns = annoy_index.get_nns_by_vector(embeddings, 1)
        name = identity_info.get(nns[0], "Unknown")
        self.global_id_to_name[global_id] = name

    def exhaust_queue(self, q: mp.Queue):
        try_count = 0
        while True:
            try:
                # Get data from each process
                q.get(block=True, timeout=0.05)

            except queue.Empty:
                try_count += 1
                if try_count == 3 and self.global_kill.is_set():
                    break
                pass  # Retry

    def run(self):
        # Discard SIGTERM and SIGKILL signals
        discard_signal()

        self.cam_sct_to_frames = defaultdict(lambda: deque(maxlen=8))
        self.cam_sct_to_poses = defaultdict(lambda: deque(maxlen=8))
        self.cam_sct_to_last_seen = defaultdict(int)

        # Initialize the Re-ID model
        semi_model = SEMIWrapper(**self.model_config)

        # Initialize annoy and mapping file
        with open(self.model_config["identity_info"], "rb") as f:
            identity_info = pickle.load(f)

        annoy_index = annoy.AnnoyIndex(2048, 'angular')
        annoy_index.load(self.model_config["annoy_file"])

        # Get data from MCT
        data_thread = Thread(target=self.data_queue_query)
        data_thread.start()

        tick_runtime_meter = AverageMeter()
        tick = 0
        while not self.global_kill.is_set():
            # Get global ID that needs to be re-id
            remaining_global_ids = self.get_unidentified()

            # Nothing to Re-ID yet
            if len(remaining_global_ids) == 0:
                time.sleep(0.01)
                continue

            current_time = time.time()

            for global_id in remaining_global_ids:
                self.reid(global_id, semi_model, annoy_index, identity_info)

            # Send the result back to MCT
            self.reply_queue.put(self.global_id_to_name)

            tick_runtime_meter.update(time.time() - current_time)
            tick += 1

            time.sleep(0.01)

        self.logger.info(f"[Re-ID] Received global kill.")
        self.reply_queue.put(DoneSignal())
        self.terminated = True
        # Wait for data refresh thread to finish
        data_thread.join()
        self.logger.info(f"[Re-ID] Data refresh thread finished.")

        if tick_runtime_meter.avg > 0:
            self.logger.info(f"[Re-ID] Average time per tick: {tick_runtime_meter.avg:.4f} seconds")
            self.logger.info(f"[Re-ID] Average ticks/second: {1 / tick_runtime_meter.avg:.4f} ticks")

        self.exhaust_queue(self.data_queue)  # Exhaust queue to prevent deadlock
