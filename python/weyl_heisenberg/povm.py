import os
import numpy as np

from weyl_heisenberg.operators import weyl_heisenberg


fiducial_path = '../data/fiducial'


def verify_povm(a: np.ndarray, rel_tol=1E-15) -> bool:
    dim = a.shape[1]
    assert a.shape[0] == dim**2
    for i in range(dim**2):
        for j in range(dim**2):
            computed = inner_product_sq(a[i], a[j])
            expected = ((i == j)*dim + 1)/(dim + 1)
            if abs(computed - expected)/expected > rel_tol:
                return False
    return True


def inner_product_sq(a: np.ndarray, b: np.ndarray) -> float:
    return abs(a.conj().dot(b))**2


def load_povm_from_file(fname: str, dim: int) -> np.ndarray:
    a = np.zeros(dim, dtype=np.cdouble)
    with open(fname, 'r', encoding='UTF-8') as f:
        for i, line in enumerate(f):
            a[i] = complex(line.replace('i', 'j'))
    return a


def save_povm_to_file(a: np.ndarray, fid_path: str = fiducial_path) -> None:
    dim = len(a)
    fname = os.path.join(fid_path, f'fiducial_{dim}.txt')
    with open(fname, 'w', encoding='UTF-8') as f:
        for coeff in a:
            line = str(coeff).strip('(').strip(')').replace('j', 'i')
            f.write(line + '\n')


def generate_povm_from_fiducial(a: np.ndarray) -> np.ndarray:
    dim = len(a)
    povm = np.zeros((dim*dim, dim), dtype=a.dtype)
    for i in range(dim):
        for j in range(dim):
            povm[i*dim + j] = weyl_heisenberg(a, i, j)
    return povm


def main():
    for dimension in range(2, 13):
        filename = os.path.join(fiducial_path, f'fiducial_{dimension}.txt')
        fid = load_povm_from_file(filename, dimension)
        povm = generate_povm_from_fiducial(fid)
        print(f'Dimension {dimension:3d}:', verify_povm(povm, rel_tol=1E-6))


if __name__ == '__main__':
    main()
