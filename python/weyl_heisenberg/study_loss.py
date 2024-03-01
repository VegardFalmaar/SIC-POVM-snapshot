import numpy as np
import matplotlib.pyplot as plt
from ft.plots import use_tex, set_ax_info   # type: ignore

from weyl_heisenberg import real_to_complex, loss


use_tex()

def f(a, b):
    n = a**2 + b**2
    if n < 1:
        return loss(real_to_complex(np.array([a, b])))
    return np.nan


def main():
    N = 201
    rnge = np.linspace(-1.1, 1.1, N)
    x, y = np.meshgrid(rnge, rnge)

    z = np.zeros_like(x)
    for i in range(N):
        for j in range(N):
            z[i, j] = f(x[i, j], y[i, j])

    fig, ax = plt.subplots()
    contourf = ax.contourf(x, y, z, levels=15, cmap='magma')
    plt.colorbar(contourf)
    ax.axis('equal')
    set_ax_info(ax, '$a$', '$b$', 'Loss in $d = 2$', legend=False)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig('../tex/Figures/GMatrixLoss-d2-v2.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
