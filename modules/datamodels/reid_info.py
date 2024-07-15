from collections import deque
import time
from typing import List, Deque, Tuple

import numpy as np

from modules.datamodels.detection_result import DetectionResult


class ReIDInfo:
    identity: str | None = None  # Identity of the person
    last_frame_time: float = 0.  # Time of the last frame
    last_identified_time: float = 0.
    reid_count: int = 0  # Number of times reid has been called

    def __init__(self, sequence_length: int):
        self.images: Deque[Tuple[int, np.ndarray, DetectionResult]] = deque(maxlen=sequence_length)  # A deque of images with max length of sequence_length

    def add_image(self, frame_idx: int, image: np.ndarray, det_result: DetectionResult):
        self.images.append((frame_idx, image, det_result))
        self.last_frame_time = time.time()

    def clear_images(self):
        self.images.clear()
        self.last_frame_time = 0.

    def get_idxs_images(self) -> List[Tuple[int, np.ndarray, DetectionResult]]:
        return list(self.images)

    def set_identity(self, identity: str | None):
        self.identity = identity
        self.last_identified_time = time.time()
