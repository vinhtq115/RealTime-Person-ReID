import logging
from pathlib import Path
from threading import Thread
import time
from typing import Tuple

import cv2
import numpy as np

from utils.ffmpeg import FFMPEG

RETRY_INTERVAL = 0.005


class BaseCamera:
    def __init__(self,
                 video: cv2.VideoCapture,
                 name: str,
                 dump_path: Path | None = None,
                 **kwargs):
        """Base camera class. Distortion is corrected if matrix and distortion coefficients are provided.

        Args:
            video (cv2.VideoCapture): Video source
            name (str): Name of camera
            dump_path (Path, optional): Path to save clean feed. Defaults to None.
        """
        self.video: cv2.VideoCapture = video
        self.name = name

        self.latest_frame = None
        self.hw_last_frame_idx = -1
        self.hw_frame_idx = -1
        self.frame_idx = -1

        self.error = False
        self.terminate = False

        # Get camera resolution
        if kwargs.get("resolution", None):
            width = int(kwargs["resolution"]["width"])
            height = int(kwargs["resolution"]["height"])
            self.resolution = (width, height)
        else:
            self.resolution = (
                int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )

        # Load camera matrix and distortion coefficients for distortion correction
        if kwargs.get("matrix_path", None) and kwargs.get("distortion_coeff_path", None):
            self.matrix = np.load(kwargs["matrix_path"])
            self.distortion_coeff = np.load(kwargs["distortion_coeff_path"])
            self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(
                self.matrix, self.distortion_coeff, self.resolution, 0, self.resolution
            )
            self.distorted = True
        else:
            self.distorted = False  # Disable distortion correction

        # Load homography matrix (if provided)
        if kwargs.get("homography_matrix_path", None):
            self.homography_matrix = np.load(kwargs["homography_matrix_path"])
        else:
            self.homography_matrix = None

        self.capture_thread = Thread(target=self.fast_capture, daemon=True)
        if dump_path:
            self.ffmpeg = FFMPEG(dump_path)
        else:
            self.ffmpeg = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def start_capture(self):
        self.capture_thread.start()
        self.logger.info(f"[{self.name}] Capture thread started.")

    def fast_capture(self):
        """Internal function to continuously grab latest frame."""
        while not self.terminate and not self.error:
            ret, frame = self.video.read()
            if not ret:
                if not self.terminate:
                    self.logger.error(f"[{self.name}] Failed to capture frame.")
                    self.error = True
                return  # Exit thread

            # Distortion correction (if enabled)
            if self.distorted:
                frame = cv2.undistort(frame, self.matrix, self.distortion_coeff, None, self.newcameramtx)

            self.latest_frame = frame
            self.hw_frame_idx += 1
        self.logger.info(f"[{self.name}] Capture thread terminated.")

    def capture(self, warmup: bool = False) -> Tuple[np.ndarray, int] | Tuple[None, None]:
        """Read frame from video capture. Also save it if visualization is enabled.

        Args:
            warmup (bool): Warmup mode. Will not increment index and write to output file when set to True. Defaults to False.

        Returns:
            tuple[np.ndarray, int] | None: Pair of frame and its index. Returns None if failed to capture frame.
        """
        while self.hw_frame_idx <= self.hw_last_frame_idx:
            time.sleep(RETRY_INTERVAL)

        if self.error:
            return None, None

        # New frame available
        frame = self.latest_frame
        frame_idx = self.hw_frame_idx
        if not warmup:
            self.hw_last_frame_idx = frame_idx
            self.frame_idx += 1

            if self.ffmpeg:
                self.ffmpeg.write(frame.tobytes())

        return frame, self.frame_idx

    def release(self):
        self.terminate = True
        if self.ffmpeg:
            self.ffmpeg.close()
            self.logger.info(f"[{self.name}] Dump file saved.")

        self.capture_thread.join()
        self.logger.info(f"[{self.name}] Capture thread joined.")
        self.video.release()
        self.logger.info(f"[{self.name}] Video capture released.")
