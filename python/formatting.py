import numpy as np


def set_numpy_print_options():
    """Set printing format for numbers in numpy arrays.
    """
    np.set_printoptions(
        edgeitems=30,
        linewidth=100000,
        formatter={'float': lambda x: f'{x: .6e}'}
    )
