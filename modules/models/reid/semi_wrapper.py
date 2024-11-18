import os

import cv2
import numpy as np
import torch

from modules.models.reid.semi import SEMI


class SEMIWrapper:
    def __init__(self, app_model: str, checkpoint: str, device: str = "cuda", **kwargs):
        self.device = device if device is not None else \
            ("cuda" if torch.cuda.is_available() else "cpu")

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
        self.mean = self.mean.to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
        self.std = self.std.to(self.device)

        assert os.path.exists(checkpoint), f"Checkpoint file {checkpoint} not found."

        self.model = SEMI(app_model)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=False)
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, images: list[np.ndarray]) -> np.ndarray:
        """Forward function to get average embeddings of a list of images.

        Args:
            images (list[np.ndarray]): List of BGR images.

        Returns:
            np.ndarray: Embedding vector of shape (1, 2048).
        """
        # Data preprocessing
        x = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
        x = [cv2.resize(image, (128, 256)) for image in x]
        x = torch.tensor(np.array(x), dtype=torch.float32, device=self.device)  # (T, H, W, C)
        x = x.permute(3, 0, 1, 2).unsqueeze(0)  # (B, C, T, H, W)
        x = (x / 255.0 - self.mean) / self.std  # Normalize

        with torch.no_grad():
            f, x = self.model(x)

        return f.cpu().numpy()
