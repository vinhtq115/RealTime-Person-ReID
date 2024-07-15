### Simple YOLO wrapper ###
import os
from typing import Tuple

import numpy as np
import torch
from ultralytics import YOLO


# BGR color codes for each keypoint
KEYPOINTS_MAP = {
    0: (255, 153, 51),  # Nose
    1: (255, 153, 51),  # Left eye
    2: (255, 153, 51),  # Right eye
    3: (255, 153, 51),  # Left ear
    4: (255, 153, 51),  # Right ear
    5: (0, 255, 0),     # Left shoulder
    6: (0, 128, 255),   # Right shoulder
    7: (0, 255, 0),     # Left elbow
    8: (0, 128, 255),   # Right elbow
    9: (0, 255, 0),     # Left wrist
    10: (0, 128, 255),  # Right wrist
    11: (0, 255, 0),    # Left hip
    12: (0, 128, 255),  # Right hip
    13: (0, 255, 0),    # Left knee
    14: (0, 128, 255),  # Right knee
    15: (0, 255, 0),    # Left ankle
    16: (0, 128, 255)   # Right ankle
}

# BGR color codes for each skeleton link
SKELETON_MAP = {
    15: [(13, (0, 255, 0))],  # Left ankle -> Left knee
    13: [(11, (0, 255, 0))],  # Left knee -> Left hip
    16: [(14, (0, 128, 255))],  # Right ankle -> Right knee
    14: [(12, (0, 128, 255))],  # Right knee -> Right hip
    11: [(12, (255, 153, 51))],  # Left hip -> Right hip
    5: [(11, (255, 153, 51)), (6, (255, 153, 51)), (7, (0, 255, 0))],  # Left shoulder -> Left hip, # Left shoulder -> Right shoulder
    6: [(12, (255, 153, 51)), (8, (0, 255, 0))],  # Right shoulder -> Right hip, Right sholder -> Right elbow
    7: [(9, (0, 255, 0))],  # Left elbow -> Left wrist
    8: [(10, (0, 255, 0))],  # Right elbow -> Right wrist
    1: [(2, (255, 153, 51)), (3, (255, 153, 51))],  # Left eye -> Right eye. Left eye -> Left ear
    0: [(1, (255, 153, 51)), (2, (255, 153, 51))],  # Nose -> Left eye, Nose -> Right eye
    2: [(4, (255, 153, 51))],  # Right eye -> Right ear
    3: [(5, (255, 153, 51))],  # Left ear -> Left shoulder
    4: [(6, (255, 153, 51))]  # Right ear -> Right shoulder
}


class YOLOWrapper:
    def __init__(self,
                 checkpoint: str,
                 device: str = "cuda",
                 imgsz: int = 640,
                 conf: float = 0.3,
                 target_class: int = 0,
                 half: bool = True):
        """Initializes the human detector.

        Args:
            checkpoint (str): Path to the model file.
            device (str, optional): Device to run model on. Defaults to "cpu".
            imgsz (int, optional): Inference size of YOLO. Defaults to 640.
            conf (float, optional): Confidence threshold. Defaults to 0.3.
            target_class (int, optional): Target class to detect. Defaults to 0.
            half (bool, optional): Use half precision. Defaults to True.

        Raises:
            FileNotFoundError: If model file is not found.
            RuntimeError: If CUDA is not available on the machine and `device` is set to `cuda`.
        """
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Model file not found: {checkpoint}")

        device = device.lower()
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this machine")

        self.det_model = YOLO(checkpoint)
        self.det_model = self.det_model.to(device)
        if self.det_model.task != "pose":
            raise ValueError("Must use model with pose estimation module.")

        self.imgsz = imgsz
        self.conf = conf
        self.target_class = target_class
        self.half = half

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return detected bounding boxes and keypoints.

        Args:
            frame (np.ndarray): Input frame (BGR). Format: (H, W, C)

        Returns:
            Tuple[np.ndarray, np.ndarray]: Detected bounding boxes and keypoints.
        """
        with torch.no_grad():
            det_results = self.det_model.predict(
                frame,
                classes=[self.target_class],
                verbose=False,
                conf=self.conf,
                half=self.half,
                imgsz=self.imgsz
            )[0]

        return det_results.boxes.data.cpu().numpy(), det_results.keypoints.data.cpu().numpy()
