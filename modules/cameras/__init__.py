from modules.cameras.base import BaseCamera
from modules.cameras.file import FileCamera
from modules.cameras.rtsp import RTSPCamera
from modules.cameras.usb import USBCamera


def get_camera_instance(camera_config: dict) -> BaseCamera:
    """Factory function to create camera instance based on configuration.

    Args:
        camera_config (dict): Configuration of the camera.

    Returns:
        BaseCamera: Camera instance.
    """
    camera_type = camera_config.pop("type")
    if camera_type == "rtsp":
        return RTSPCamera(**camera_config)
    elif camera_type == "usb":
        return USBCamera(**camera_config)
    elif camera_type == "file":
        return FileCamera(**camera_config)
    else:
        raise ValueError(f"Unsupported camera type: {camera_type}")
