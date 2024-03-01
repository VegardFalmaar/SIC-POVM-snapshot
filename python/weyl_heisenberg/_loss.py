import time
import logging

import numpy as np
from numba import jit   # type: ignore


def target_function(x: np.ndarray) -> float:
    """Wrapper for G-matrix loss to define target function.

    Does not enforce norm <= 1, but penalizes this with a large value of the
    loss. The reason for this is that ``scipy.optimize.minimize``, even with
    constraints, sometimes evaluates the function just outside the specified
    boundaries.

    args:
        x (np.ndarray of float): the candidate array, should be of length
            ``2*d - 2`` where ``d`` is the dimensionality of the Hilbert space

    returns:
        (float): the loss
    """
    n = np.linalg.norm(x)
    if n > 1.0:
        return 0.1 * n**2

    result = loss(real_to_complex(x))

    if result < 1e-15:
        logging.getLogger('weyl-heisenberg.loss').warning(
            'Loss clipped from %f to 1e-15', result
        )
        return 1e-15
    return result


def real_to_complex(a: np.ndarray) -> np.ndarray:
    """
    Cast array of 2N real parameters to array of N + 1 complex parameters.

    First element of result array is set to have zero complex phase and modulus
    such that the result array is normalized.
    """
    assert a.size % 2 == 0
    n = np.linalg.norm(a)
    assert n <= 1.0
    z = np.empty(a.size//2 + 1, dtype=np.complex128)
    z.real[0] = np.sqrt(1.0 - n**2)
    z.imag[0] = 0.0
    z.real[1:] = a[::2]
    z.imag[1:] = a[1::2]
    return z


@jit(nopython=True)
def _loss_2(a: np.ndarray) -> float:
    """Calculate the G-matrix loss of the input vector.

    args:
        a (np.ndarray of complex numbers):
            the candidate vector for which the loss should be calculated

    returns:
        (float): the loss
    """
    d = a.size
    A = np.empty(shape=(d, d), dtype=np.complex128)
    for i in range(d):
        A[i] = np.roll(a, -i)
    A_conj = A.conj()
    result = (np.abs(a)**4).sum()**2
    for k in range(1, d):
        for l in range(k):
            result += 2*abs((a * A_conj[k] * A_conj[l]).dot(A[(k+l) % d]))**2
        result += abs((a * A_conj[k]**2).dot(A[(2*k) % d]))**2
    bound = 2.0/(d + 1)
    result -= bound
    return result


@jit(nopython=True)
def loss(a: np.ndarray):
    """Calculate the G-matrix loss of the input vector.

    args:
        a (np.ndarray of complex numbers):
            the candidate vector for which the loss should be calculated

    returns:
        (float): the loss, lower limit set to 1e-15
    """
    d = a.size
    A = np.empty(2*d, dtype=np.complex128)
    A[:d] = a
    A[d:] = a
    A_conj = A.conj()
    result = (np.abs(a)**4).sum()**2
    for k in range(1, d):
        for l in range(k):
            result += 2*abs((a * A_conj[k:k+d] * A_conj[l:l+d]).dot(A[(k+l) % d:(k+l) % d + d]))**2
        result += abs((a * A_conj[k:k+d]**2).dot(A[(2*k) % d : (2*k) % d + d]))**2
    bound = 2.0/(d + 1)
    result -= bound
    return result


def time_loss():
    np.random.seed(12345)
    x = np.random.random(1000)
    x /= np.linalg.norm(x)
    x *= 0.9
    a = real_to_complex(x)
    N_trials = 20
    start = time.perf_counter()
    for _ in range(N_trials):
        _loss_2(a)
        # loss(a)
    duration = time.perf_counter() - start
    print(f'Total duration: {duration:.3f} s')
    print(f'Time per evalution: {duration/N_trials:.5f} s')


if __name__ == '__main__':
    time_loss()
