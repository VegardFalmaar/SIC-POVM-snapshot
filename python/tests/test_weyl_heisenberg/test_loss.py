import numpy as np

from weyl_heisenberg.loss import loss


tol = 1e-15


def test_loss_2d():
    v = np.array([1.0 + 0.0j, 0.0 + 0.0j])
    expected = 1.0/3
    assert abs(loss(v) - expected) < tol

    v = np.array([1.0 + 0.0j, 1.0 + 0.0j])
    expected = 46.0/3
    assert abs(loss(v) - expected) < tol

    v = np.array([2.0 + 0.0j, 0.0 + 1.0j])
    expected = 1441.0/3
    assert abs(loss(v) - expected) < tol


def test_loss_3d():
    v = np.array([0.1 + 0.0j, 0.0 + 0.2j, 0.3 + 0.0j])
    expected = -0.49977912
    assert abs(loss(v) - expected) < tol
