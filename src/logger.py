"""logging module"""
import logging
from typing import Union


def init_logger(stream: bool = True, file: bool = False, level: str = "INFO", file_path: Union[str, None] = None):
    """Function that init a logger which is then usable via logging module everywhere in the code.

    Args:
        stream (bool, optional): Logger will write logs in the console if set to True. Defaults to True.
        file (bool, optional): Logger will write logs in a file if set to True. Defaults to False.
        level (str, optional): Log level (see logging documentation for details). Defaults to "INFO".
        file_path (str, optional): Path of the log file to where to write logs. Must be specified if file is True. Defaults to None.

    Raises:
        ValueError: Raise value if no handler is provided that is you must at least set stream or file to True
    """
    handlers = []
    if file:
        file_formatter = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s %(funcName)s: %(message)s")
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    if stream:
        stream_formatter = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s %(funcName)s: %(message)s")
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(stream_formatter)
        handlers.append(stream_handler)
    if len(handlers) == 0:
        raise ValueError("You provide no handler, you must at least set stream or file to True")

    logging.basicConfig(level=level, datefmt="%Y-%m-%dT%H:%M:%S", handlers=handlers)
