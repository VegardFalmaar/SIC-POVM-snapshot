import logging

import shgo
from weyl_heisenberg import target_function
from catalogue import catalogue_parameters
from environment_variables import result_directory


def main() -> None:
    # run_one()
    run_all()


def run_all() -> None:
    shgo_logger = logging.getLogger('scipy.optimize.shgo')
    shgo_logger.setLevel(logging.DEBUG)

    min_logger = logging.getLogger('minimization_history')
    min_logger.setLevel(logging.DEBUG)

    # for sampler in ['simplicial', 'halton', 'sobol']:
    for sampler in ['sobol']:
        # for use_constraints in [False]:
        for use_constraints in [True, False]:
            # for complex_dimension in range(10, 11):
            for complex_dimension in [10]:
                print('\n' + '='*20)
                print(f'SHGO in d = {complex_dimension}')
                parameters = shgo.Parameters(
                    target_name='SICPOVM',
                    n_dims=2*complex_dimension - 2,
                    sampling_method=sampler,
                    use_constraints=use_constraints,

                )
                path = catalogue_parameters(result_directory(), parameters)
                shgo_logger.handlers.clear()
                min_logger.handlers.clear()

                handler = logging.FileHandler(path / 'shgo_log.txt')
                handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
                shgo_logger.addHandler(handler)

                handler = logging.FileHandler(path / 'minimization_history_log.txt')
                handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
                min_logger.addHandler(handler)

                shgo.run(target_function, (-1.0, 1.0), parameters, path)


def run_one() -> None:
    """Run the SHGO algorithm on the SIC-POVM problem.
    """
    complex_dimension: int = 10

    print('\n' + '='*20)
    print(f'SHGO in d = {complex_dimension}')
    parameters = shgo.Parameters(
        target_name='SICPOVM',
        n_dims=2*complex_dimension - 2,
        sampling_method='simplicial',
        use_constraints=True,
    )
    path = catalogue_parameters(result_directory(), parameters)

    logger = logging.getLogger('scipy.optimize.shgo')
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(path / 'shgo_log.txt')
    handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    logger.addHandler(handler)

    logger = logging.getLogger('minimization_history')
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(path / 'minimization_history_log.txt')
    handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    logger.addHandler(handler)

    shgo.run(target_function, (-1.0, 1.0), parameters, path)


if __name__ == "__main__":
    main()
    # print(target_function(np.array([0.5, -0.5, -0.5, 0.5])))
    # print(loss(np.array([0 + 0j, 0.5 - 0.5j, -0.5 + 0.5j])))
