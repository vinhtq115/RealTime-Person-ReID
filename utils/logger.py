import logging
import multiprocessing


def create_logger():
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(\
        '[%(asctime)s| %(levelname)s| %(processName)s] %(message)s')
    handler = logging.FileHandler('logs/your_file_name.log')
    handler.setFormatter(formatter)

    if not len(logger.handlers):
        logger.addHandler(handler)
    return logger
