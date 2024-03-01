import os
import logging
from pathlib import Path


def get_experimental_flag() -> bool:
    """Get experimental flag from environment variable 'POVM_EXPERIMENTAL'.

    Environment variable should be a boolean, one of:
        - True
        - False

    Default value is False.

    returns:
        (bool): True or False corresponding to the value of the environment
            variable
    """
    flag = os.getenv('POVM_EXPERIMENTAL', default='False')
    if flag == 'True':
        return True
    if flag == 'False':
        return False
    raise ValueError(
        f'Unknown value \'{flag}\' for environment variable '
        + '\'POVM_EXPERIMENTAL\''
    )


def get_debug_flag() -> int:
    """Get debug flag from environment variable 'POVM_DEBUG'.

    Environment variable should be an integer, one of
        - 1 (Critical),
        - 2 (Error),
        - 3 (Warning),
        - 4 (Info),
        - 5 (Debug)

    Default value is 4.

    returns:
        (int): the one of
            - logging.CRITICAL
            - logging.ERROR
            - logging.WARNING
            - logging.INFO
            - logging.DEBUG

            corresponding to the value of the debug environment variable
    """
    flag = os.getenv('POVM_DEBUG', default='4')
    if flag == '1':
        return logging.CRITICAL
    if flag == '2':
        return logging.ERROR
    if flag == '3':
        return logging.WARNING
    if flag == '4':
        return logging.INFO
    if flag == '5':
        return logging.DEBUG
    raise ValueError(
        f'Unknown value \'{flag}\' for environment variable \'POVM_DEBUG\''
    )


def result_directory() -> Path:
    """Get the name of the result directory.

    Depends on the 'POVM_EXPERIMENTAL' environment variable.
    """
    if get_experimental_flag():
        return Path('experimental_results')
    home = os.getenv('HOME')
    if home is None:
        raise ValueError('Environment variable \'HOME\' not set')
    return Path(home) / 'SIC-POVM-results'


if __name__ == '__main__':
    print(result_directory())
