import numpy as np


def main():
    sigma_x = np.array([[0, 1], [1, 0]], dtype='complex128')
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype='complex128')
    sigma_z = np.array([[1, 0], [0, -1]], dtype='complex128')

    s2 = 1/np.sqrt(2)

    basis = [s2 * m for m in [np.eye(2), sigma_x, sigma_y, sigma_z]]
    eigenvalues = [[s2, s2], [s2, -s2], [s2, -s2], [s2, -s2]]
    eigenvectors = [
        [np.array([1 + 0j, 0 + 0j]), np.array([0 + 0j, 1 + 0j])],
        [s2 * np.array([1 + 0j, 1 + 0j]), s2 * np.array([1 + 0j, -1 + 0j])],
        [s2 * np.array([1 + 0j, 0 + 1j]), s2 * np.array([1 + 0j, 0 - 1j])],
        [np.array([1 + 0j, 0 + 0j]), np.array([0 + 0j, 1 + 0j])],
    ]

    # verify that the eigenvalues and eigenvectors are correct
    for i, M in enumerate(basis):
        for val, vec in zip(eigenvalues[i], eigenvectors[i]):
            assert np.allclose(M@vec.reshape(-1, 1), val*vec.reshape(-1, 1))


    trial_state = np.array([3/5, 4/5*np.exp(np.pi*1j/4)])

    for i in range(len(basis)):
        print('\n' + '-'*20 + '\n')
        print('Matrix', i)
        for j, (val, vec) in enumerate(zip(eigenvalues[i], eigenvectors[i])):
            print('-'*5)
            print('Vector', j)
            prob = abs(vec.reshape(1, -1) @ trial_state.reshape(-1, 1))**2
            print(f'Probability of obtaining eigenvalue {val} is {prob}')


if __name__ == '__main__':
    main()
