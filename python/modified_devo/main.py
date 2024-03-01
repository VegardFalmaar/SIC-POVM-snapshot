from modified_devo import Parameters
from modified_devo.devo_BFGS import run
from weyl_heisenberg import target_function
from catalogue import catalogue_parameters
from environment_variables import result_directory


def run_one_dimension(complex_dimension: int, pop_thinning_factor: float):
    msg = f'Modified devo in d = {complex_dimension} ' \
        + f'with pop-thinning {pop_thinning_factor}'
    print('\n' + '='*len(msg))
    print(msg)

    # always use minimizer, as it will not focus to a solution otherwise
    p = Parameters(
        target_name='SICPOVM',
        n_dims=2*complex_dimension - 2,
        use_minimizer=True,
        pop_thinning_factor=pop_thinning_factor
    )
    path = catalogue_parameters(result_directory(), p)
    run(target_function, (-1.0, 1.0), p, path)


def main():
    # for pt in [1.0, 0.9, 0.8, 0.7, 0.6]:
    for pt in [0.6]:
        # for d in range(10, 21):
            # run_one_dimension(complex_dimension=d, pop_thinning_factor=pt)
        for d in range(21, 31):
            run_one_dimension(complex_dimension=d, pop_thinning_factor=pt)


if __name__ == "__main__":
    main()
