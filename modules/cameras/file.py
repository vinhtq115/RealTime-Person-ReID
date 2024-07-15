import os
import time
from typing import Tuple

import cv2
import numpy as np

from modules.cameras.base import BaseCamera


class FileCamera(BaseCamera):
    def __init__(self, path_to_file: str, name: str | None = None, fps: int = 15, **kwargs):
        """Initialize file video stream.

        Args:
            path_to_file (str): Path to file
            name (str, optional): Name for logging. If set to None, path to file will be used. Defaults to None.
            fps (int, optional): Frames per second to simulate. Defaults to 15.

        Raises:
            FileNotFoundError: If file is not found.
        """
        if not os.path.exists(path_to_file):
            raise FileNotFoundError(f"File {path_to_file} not found")

        super().__init__(
            cv2.VideoCapture(path_to_file),
            name if name else path_to_file,
            **kwargs
        )

        self.frame_time = 1 / fps  # Time to wait between frames (in second)
        self.last_frame_time = 0

    def capture(self) -> Tuple[np.ndarray, int] | Tuple[None, None]:
        """Read frame from video capture.

        Returns:
            tuple[np.ndarray, int] | None: Pair of frame and its index. Returns None if failed to capture frame.
        """
        while True:
            current_time = time.time()
            if current_time - self.last_frame_time >= self.frame_time:
                break

            time.sleep(self.frame_time - (current_time - self.last_frame_time))

        frame, frame_idx = super().capture()
        self.last_frame_time = time.time()
        return frame, frame_idx
