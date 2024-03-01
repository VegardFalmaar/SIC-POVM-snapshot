import numpy as np


def shift(a: np.ndarray) -> np.ndarray:
    r"""Implementation of the linear shift operator X.

    It is defined as
    X \ket{j} = \ket{j + 1 (mod d)}
    """
    return np.roll(a, 1)


def phase(v: np.ndarray) -> np.ndarray:
    r"""Implementation of the linear phase operator Z.

    It is defined as
    Z \ket{j} = \omega^j \ket{j},
    where
    \omega = \exp(2 \pi i / d).
    """
    a = v.copy()
    dim = len(a)
    omega = np.exp(2j*np.pi/dim)
    omega_i = omega
    for i in range(1, dim):
        a[i] *= omega_i
        omega_i *= omega
    return a


def weyl_heisenberg(a: np.ndarray, k: int, l: int) -> np.ndarray:
    d = len(a)
    result = a.copy()
    for _ in range(l):
        result = phase(result)
    for _ in range(k):
        result = shift(result)
    return np.exp(1j*k*l*np.pi/d)*result
