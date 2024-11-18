# Module for various signaling
import logging
import multiprocessing as mp
import signal


class GracefulKiller:
    # Used for terminating all processes using SIGINT/SIGTERM
    # Reference: https://stackoverflow.com/a/31464349
    def __init__(self, global_kill_event: mp.Event):
        self.global_kill_event = global_kill_event
        self.logger = logging.getLogger("GracefulKiller")
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.global_kill_event.set()
        self.logger.info("SIGINT/SIGTERM received. Terminating all processes.")


class DoneSignal:
    # Used for signaling from SCT to MCT that a camera has finished
    # processing and no more data will be sent to it
    def __init__(self):
        pass


def discard_signal():
    # Ignore SIGINT and SIGTERM signals (for child processes)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)