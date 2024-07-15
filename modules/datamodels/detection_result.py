from dataclasses import dataclass


@dataclass
class DetectionResult:
    """Class for storing human detection result"""
    conf: float  # Confidence score
    tracking_id: int | None  # Tracking ID
    x1: int  # Top left x-coordinate
    y1: int  # Top left y-coordinate
    x2: int  # Bottom right x-coordinate
    y2: int  # Bottom right y-coordinate

    def to_dict(self) -> dict:
        return {
            "conf": str(self.conf),
            "tracking_id": self.tracking_id,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2
        }