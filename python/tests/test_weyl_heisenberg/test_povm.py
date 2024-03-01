import pytest
import numpy as np

from weyl_heisenberg import povm


class TestVerifyPovmWithAnalytic2D:
    s_13 = 1/np.sqrt(3)
    s_23 = np.sqrt(2/3)
    a = np.array([
        [1 + 0j, 0 + 0j],
        [s_13 + 0j, s_23 + 0j],
        [s_13 + 0j, s_23*np.exp(2j*np.pi/3)],
        [s_13 + 0j, s_23*np.exp(4j*np.pi/3)]
    ], dtype=np.cdouble)

    def test_verifies_correct_povm(self):
        assert povm.verify_povm(self.a)

    @pytest.mark.parametrize('idx', [2, 3])
    @pytest.mark.parametrize(
        'coeff',
        [np.sqrt(2/3) + 0j, np.sqrt(2/3)*np.exp(1.999j*np.pi/3), np.sqrt(2/3)*np.exp(3.999j*np.pi/3)]
    )
    def test_fails_when_input_vectors_not_equiangular(self, idx: int, coeff: complex):
        self.a[idx, 1] = coeff
        assert not povm.verify_povm(self.a)

    @pytest.mark.parametrize('idx', [0, 1, 2, 3])
    @pytest.mark.parametrize(
        'coeff',
        [1.01, np.sqrt(1.01/3)]
    )
    def test_fails_when_input_vectors_not_normalized_first_coeff(self, idx: int, coeff: float):
        self.a[idx, 0] = coeff
        assert not povm.verify_povm(self.a)

    @pytest.mark.parametrize(
        'idx, coeff',
        [(0, 0.01), (1, np.sqrt(2.01/3)), (2, np.sqrt(2.01/3)*np.exp(2j*np.pi/3)), (3, np.sqrt(2.01/3)*np.exp(4j*np.pi/3))]
    )
    def test_fails_when_input_vectors_not_normalized_second_coeff(self, idx: int, coeff: float):
        self.a[idx, 0] = coeff
        assert not povm.verify_povm(self.a)


@pytest.mark.parametrize(
    'a, b, exp',
    [(np.array([1 + 0j]), np.array([1 + 0j]), 1),
        (np.array([1 + 0j, 1 + 0j]), np.array([1 + 0j, -1 + 0]), 0),
        (np.array([1 + 0j, 1j]), np.array([1j/2, 3/4]), 1/16)
    ]
)
def test_inner_product_sq(a, b, exp):
    eps = 1E-15
    assert abs(povm.inner_product_sq(a, b) - exp) < eps
