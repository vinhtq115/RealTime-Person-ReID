import logging
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import List

from dotenv import load_dotenv
from natsort import natsorted
import yaml

from modules.pipeline_mct import PipelineMCT
from modules.pipeline_sct import PipelineSCT
from modules.pipeline_reid import PipelineReID
from modules.signaling import GracefulKiller


logging.basicConfig(level=logging.DEBUG, format="{asctime} - {levelname} - {message}", datefmt="%Y-%m-%d %H:%M:%S", style="{")


if __name__ == "__main__":
    load_dotenv()
    mp.set_start_method("spawn")

    # Set path to save videos
    root_dir = Path(os.environ["ROOT_DIR"]) / time.strftime("%Y%m%d_%H%M%S")
    save_dir = root_dir / "tracklets"
    os.makedirs(save_dir.as_posix(), exist_ok=True)
    visualize_dir = root_dir / "visualize"
    os.makedirs(visualize_dir.as_posix(), exist_ok=True)
    raw_data_dir = root_dir / "raw_data"
    os.makedirs(raw_data_dir.as_posix(), exist_ok=True)
    screenshot_dir = root_dir / "screenshots"
    os.makedirs(screenshot_dir.as_posix(), exist_ok=True)
    reid_dump_dir = root_dir / "reid_dump"
    os.makedirs(reid_dump_dir.as_posix(), exist_ok=True)

    # Read config file
    with open(os.environ["CONFIG_FILE"], "r") as file:
        config = yaml.safe_load(os.path.expandvars(file.read()))
    camera_configs: dict = config["cameras"]
    mct_config: dict = config["mct"]
    for name, camera_config in camera_configs.items():
        camera_config["name"] = name

    model_configs = config["models"]

    ### Single Camera Tracking ###
    # Each camera runs in a separate process
    processes_sct: List[PipelineSCT] = []
    global_kill = mp.Event()  # For terminating all processes
    process_killer = GracefulKiller(global_kill_event=global_kill)
    start_event = mp.Event()  # For starting all processes at the same time
    ready: List[mp.Event] = []
    data_queues: List[mp.Queue] = []
    for name in natsorted(camera_configs.keys()):
        _ready_event = mp.Event()
        _data_queue = mp.Queue()
        ready.append(_ready_event)
        data_queues.append(_data_queue)
        pipeline = PipelineSCT(
            model_config=model_configs,
            camera_config=camera_configs[name],
            visualize_dir=visualize_dir,
            data_queue=_data_queue,
            ready=_ready_event,
            start_event=start_event,
            global_kill=global_kill,
        )
        processes_sct.append(pipeline)
        pipeline.start()

    ### Multi Camera Tracking ###
    reid_queue = mp.Queue()  # For MCT to Re-ID module coomunication
    reid_result_queue = mp.Queue()  # For Re-ID to MCT module communication
    process_mct = PipelineMCT(
        mct_config=mct_config,
        camera_configs=camera_configs,
        data_queues=data_queues,
        reid_queue=reid_queue,
        reid_result_queue=reid_result_queue,
        raw_data_dir=raw_data_dir,
        start_event=start_event,
        global_kill=global_kill,
        screenshot_dir=screenshot_dir,
    )
    process_mct.start()

    ### Re-ID ###
    process_reid = PipelineReID(
        model_config=model_configs,
        save_dir=reid_dump_dir,
        mct_to_reid_data_queue=reid_queue,
        reid_to_mct_data_queue=reid_result_queue,
        global_kill=global_kill,
    )
    process_reid.start()

    # Wait for all cameras to get ready
    while not all([r.is_set() for r in ready]):
        time.sleep(0.05)

    # Start all cameras at the same time
    start_event.set()

    while not global_kill.is_set():
        time.sleep(0.1)

    for idx, process in enumerate(processes_sct):
        logging.info(f"Terminating camera {idx}")
        process.join()
    logging.info("All cameras terminated.")
    process_mct.join()
    logging.info("MCT finished.")
    process_reid.join()
    logging.info("Re-ID finished.")

    logging.info("All cameras finished.")
