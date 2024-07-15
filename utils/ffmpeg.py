from pathlib import Path
import subprocess as sp


FFMPEG_CMD = [
    "ffmpeg", "-y", "-hide_banner",
    "-loglevel", "error", "-nostats",
    "-f", "rawvideo",
    "-use_wallclock_as_timestamps", "1",
    "-pix_fmt", "bgr24",
    "-s", "1920x1080",
    "-vsync", "1",
    "-r", "15",
    "-i", "-",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-preset", "fast",
    "-crf", "26",
    "-g", "30",
    "-f", "matroska",
]


class FFMPEG:
    def __init__(self, save_path: Path):
        self.save_path = save_path
        self.ffmpeg_pipe = sp.Popen(
            FFMPEG_CMD + [save_path.as_posix()],
            stdin=sp.PIPE
        )

    def write(self, frame: bytes):
        self.ffmpeg_pipe.stdin.write(frame)

    def close(self):
        self.ffmpeg_pipe.stdin.close()
        self.ffmpeg_pipe.wait()
