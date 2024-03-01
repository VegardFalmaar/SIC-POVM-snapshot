import logging
from pathlib import Path

from environment_variables import get_debug_flag


def get_logger(path: Path, name: str) -> logging.Logger:
    """Create/get a logger for the current numerical simulation.

    args:
        path (Path): the path in which the logs should be saved
        name (str): the name of the logger

    returns:
        (logging.Logger): the logger
    """
    result = logging.getLogger(name)

    result.setLevel(get_debug_flag())

    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

    file_handler = logging.FileHandler(path / 'log.txt')
    file_handler.setFormatter(formatter)

    result.addHandler(file_handler)

    return result


def _main():
    logger = logging.getLogger('test')
    logger.setLevel(get_debug_flag())

    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

    file_handler = logging.FileHandler('log.txt')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.info('Here is some information')
    logger.warning('Here is a warning')


if __name__ == '__main__':
    _main()
