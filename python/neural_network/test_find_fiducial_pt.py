import torch

from find_fiducial_pt import GMatrixLoss
from utils import load_povm_from_file


class TestGMatrixLoss:
    def test_with_grassl_fiducial(self):
        fname = '../data/grassl-scott/sicfiducial_2a.txt'
        dim = 2
        a = load_povm_from_file(fname, dim)
        t = torch.from_numpy(a).unsqueeze(0)
        eps = 1E-14
        loss = GMatrixLoss(dim)
        assert loss(t) < eps

    def test_generate_indices(self):
        d = 4
        sol = torch.tensor([
            [2, 0, 1, 1],
            [2, 0, 2, 2],
            [2, 1, 2, 3],
            [2, 0, 3, 3],
            [2, 1, 3, 0],
            [2, 2, 3, 1],
            [1, 1, 1, 2],
            [1, 2, 2, 0],
            [1, 3, 3, 2],
        ], dtype=torch.int)
        loss = GMatrixLoss(d)
        assert torch.equal(loss.generate_indices(), sol)

    def test_sort_indices(self):
        unsorted = torch.tensor([
            [1, 2, 1, 4],
            [1, 0, 8, 6],
            [2, 5, 45, 3],
            [3, 8, 0, 4],
            [8, 2, 2, 2],
            [3, 7, 7, 6],
            [9, 7, 7, 6],
            [2, 7, 7, 5],
            [3, 7, 3, 6],
        ], dtype=torch.int)
        sortd = torch.tensor([
            [1, 0, 8, 6],
            [1, 2, 1, 4],
            [8, 2, 2, 2],
            [2, 5, 45, 3],
            [3, 7, 3, 6],
            [2, 7, 7, 5],
            [3, 7, 7, 6],
            [9, 7, 7, 6],
            [3, 8, 0, 4],
        ], dtype=torch.int)
        d = 4   # arbitrary
        loss = GMatrixLoss(d)
        assert torch.equal(loss.sort_indices(unsorted), sortd)
