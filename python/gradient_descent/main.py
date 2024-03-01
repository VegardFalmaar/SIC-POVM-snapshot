import numpy as np

from gradient_descent.random_BFGS import run
from gradient_descent import Parameters
from weyl_heisenberg import target_function
from catalogue import catalogue_parameters
from environment_variables import result_directory


def main():
    """Run the gradient descent algorithm on the SIC-POVM problem.
    """
    # for complex_dimension in range(10, 31):
    for complex_dimension in [6]:
        print('\n' + '='*20)
        print(f'Gradient descent in d = {complex_dimension}')
        parameters = Parameters(
            target_name='SICPOVM', n_dims=2*complex_dimension - 2
        )
        path = catalogue_parameters(result_directory(), parameters)
        run(target_function, (-1.0, 1.0), parameters, path)


if __name__ == "__main__":
    main()
