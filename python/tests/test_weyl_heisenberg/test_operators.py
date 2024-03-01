import pytest
import numpy as np

from weyl_heisenberg.operators import shift, phase


@pytest.mark.parametrize(
    'vec, exp',
    [(np.array([1+0j, 0+0j]), np.array([0+0j, 1+0j])),
        (np.array([0+0j, 1+0j]), np.array([1+0j, 0+0j])),
        (np.array([1+2j, 0+3j]), np.array([0+3j, 1+2j])),
        (np.array([1+0j, 0+2j, 3+4j]), np.array([3+4j, 1+0j, 0+2j]))
    ]
)
def test_shift(vec, exp):
    assert np.allclose(shift(vec), exp)


def test_shift_does_not_modify_input():
    a = np.array([1+0j, 0+2j, 3+4j])
    a_copy = a.copy()
    shift(a)
    assert np.allclose(a, a_copy)


@pytest.mark.parametrize(
    'vec, exp',
    [(np.array([1+0j, 0+0j]), np.array([1+0j, 0+0j])),
        (np.array([0+0j, 1+0j]), np.array([0+0j, -1+0j])),
        (np.array([1+2j, 0+3j]), np.array([1+2j, 0-3j])),
        (np.array([1+0j, 0+2j, 3+4j]), np.array([1+0j, np.exp(2j*np.pi/3)*2j,
            np.exp(4j*np.pi/3)*(3+4j)]))
    ]
)
def test_phase(vec, exp):
    assert np.allclose(phase(vec), exp)


def test_phase_does_not_modify_input():
    a = np.array([1+0j, 0+2j, 3+4j])
    a_copy = a.copy()
    phase(a)
    assert np.allclose(a, a_copy)
