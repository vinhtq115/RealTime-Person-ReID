import cv2

from modules.cameras.base import BaseCamera


class RTSPCamera(BaseCamera):
    def __init__(self,
                 address: str,
                 username: str | None = None,
                 password: str | None = None,
                 name: str | None = None,
                 **kwargs):
        """Initialize RTSP video stream.

        Args:
            address (str): RTSP address (without rtsp:// prefix)
            username (str | None, optional): Username (if needed). Defaults to None.
            password (str | None, optional): Password (if needed). Defaults to None.
            name (str | None, optional): Name for logging. If set to None, RTSP address will be used. Defaults to None.
        """
        if username and password:
            _rtsp_address = f"rtsp://{username}:{password}@{address}"
        else:
            _rtsp_address = f"rtsp://{address}"

        super().__init__(
            cv2.VideoCapture(_rtsp_address),
            name if name else f"rtsp://{address}",
            **kwargs
        )
