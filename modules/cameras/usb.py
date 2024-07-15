import cv2

from modules.cameras.base import BaseCamera


class USBCamera(BaseCamera):
    def __init__(self,
                 device_id: int,
                 fourcc: str | None = None,
                 width: int | None = None,
                 height: int | None = None,
                 fps: int | None = None,
                 name: str | None = None,
                 **kwargs):
        """Initialize USB camera.

        Args:
            device_id (int): Camera ID in host machine.
            fourcc (str | None, optional): Video codec. Defaults to None.
            width (int | None, optional): Camera width (in pixels). Defaults to None.
            height (int | None, optional): Camera height (in pixels). Defaults to None.
            fps (int | None, optional): Framerate of camera. Defaults to None.
            name (str | None, optional): Name for logging. If set to None, it will be set to "USB@<camera_id>". Defaults to None.
        """
        super().__init__(
            cv2.VideoCapture(device_id),
            name if name else f"USB@{device_id}",
            **kwargs
        )

        # Set FOURCC
        assert len(fourcc) == 4 if fourcc else True, "FOURCC must be 4 characters long"
        if fourcc:
            self.video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

        # Set resolution
        if width:
            self.video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Set FPS
        if fps:
            self.video.set(cv2.CAP_PROP_FPS, fps)
