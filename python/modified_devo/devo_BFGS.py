from pathlib import Path
import multiprocessing as mp
from typing import Callable, Tuple, List

import numpy as np
from scipy.spatial.distance import pdist    # type: ignore
from scipy.optimize import minimize         # type: ignore
import matplotlib.pyplot as plt

import target_functions as tf
from modified_devo.my_scipy_diffevo import DifferentialEvolutionSolver
from modified_devo import Parameters
import formatting
import plot
from minimization_history import TargetWrapper, MinimizationHistory
from catalogue import catalogue_parameters
from log import get_logger
from environment_variables import result_directory

formatting.set_numpy_print_options()


def max_relative_distance(pop, x_best, x_min, x_max):
    x_best_unit_cube = (x_best - x_min) / (x_max - x_min)
    np.append(pop, x_best_unit_cube)
    return np.max(pdist(pop, 'minkowski', p=np.inf))


def sorted_results(de_solver, x_min, x_max):
    """Sort the population according to f vals, best point first."""
    population = x_min + (x_max - x_min) * de_solver.population
    function_values = de_solver.population_energies
    ordering = np.argsort(function_values)
    return population[ordering], function_values[ordering]


def run(
    function: Callable,
    bounds: Tuple[float, float],
    parameters: Parameters,
    path: Path
) -> MinimizationHistory:
    logger = get_logger(path, name=f'devo-{path.name}')

    if parameters.seed:
        logger.debug('Setting random seed %d', parameters.seed)
        np.random.seed(parameters.seed)

    target = TargetWrapper(function, parameters.n_dims)
    target.history.start_timing()
    x_min, x_max = bounds

    # Create the initial population
    if parameters.target_name == 'SICPOVM':
        # create a population of (N + 1)-dimensional normalized vectors, and
        # remove one dimension to create a population of N-dimensional
        # subnormalized vector
        current_pop = x_min + (x_max - x_min) \
            * np.random.random((parameters.de_n_pop, parameters.n_dims + 1))
        current_pop /= np.linalg.norm(current_pop, axis=1).reshape(-1, 1)
        current_pop = current_pop[:, :-1]
    else:
        current_pop = x_min + (x_max - x_min) \
            * np.random.random(size=(parameters.de_n_pop, parameters.n_dims))
    current_f_vals = None

    for trial_i in range(parameters.n_trials):
        solver = DifferentialEvolutionSolver(
            target,
            bounds=[(x_min, x_max)]*parameters.n_dims,
            strategy=parameters.de_strategy,
            init=current_pop,
            init_f_vals=current_f_vals,
            x0=target.x_best if parameters.use_x0_insertion else None,
            maxiter=parameters.de_maxiter,
            popsize=int(len(current_pop)/parameters.n_dims),
            tol=0.01,
            mutation=(0.1, 1.0),  # in [0,2)
            recombination=0.1,   # in [0,1]
            seed=parameters.seed + trial_i if parameters.seed else None,
            callback=None,
            disp=False,
            polish=False,
            atol=0,
            updating='immediate',  # 'deferred'
            workers=1,
            constraints=parameters.get_constraints(),
            integrality=None,
            vectorized=False
        )
        solver.solve()
        current_pop, current_f_vals = sorted_results(solver, x_min, x_max)

        # Run minimizer on current best point?
        # if parameters.use_minimizer and (trial_i % parameters.n_dims == 0):
        if parameters.use_minimizer and (trial_i % parameters.n_dims == 0):
            res = minimize(
                target,
                target.x_best,
                bounds=[(x_min, x_max)]*parameters.n_dims,
                method='trust-constr' if parameters.use_constraints else 'L-BFGS-B',
                constraints=parameters.get_constraints(),
                options=parameters.get_options(),
            )
            current_pop[0] = res.x
            current_f_vals[0] = res.fun

        # Check if current smallest function value is very close to the
        # theoretical f_min = 0
        if np.isclose(target.current_f_min, 0.0, atol=parameters.minimization_threshold):
            target.history.solution_found = True
            logger.info("Converged due to current_f_min close to 0.0.")
            logger.info("Final population from DE:")
            logger.info(current_pop)
            # Break out of trials loop
            break

        # Check if entire DE population has collapsed into the current x_best
        max_rel_dist = max_relative_distance(
            solver.population, target.x_best, x_min, x_max
        )
        if max_rel_dist < parameters.max_rel_dist_threshold:
            logger.info(info_line(trial_i, target, parameters, max_rel_dist))
            logger.info(
                'Converged due to max_rel_dist < %e',
                parameters.max_rel_dist_threshold
            )
            logger.info('Final population from DE:')
            logger.info(current_pop)
            # Break out of trials loop
            break

        if target.number_of_evaluations >= parameters.f_evals_max:
            logger.info('Maximum number of function evaluations reached')
            break

        # Thin the population?
        min_pop_size = max(parameters.n_dims, 10)
        samples_to_keep = int(parameters.pop_thinning_factor * len(current_pop))
        if samples_to_keep > min_pop_size:
            current_pop = current_pop[:samples_to_keep]
            current_f_vals = current_f_vals[:samples_to_keep]

        logger.info(info_line(trial_i, target, parameters, max_rel_dist))

    target.history.stop_timing()
    target.append_best_evaluation()
    target.history.save_results(path)

    logger.info('Elapsed time: %.3f seconds', target.history.elapsed_time)
    logger.info(target)
    return target.history


def info_line(trial_i: int, target: TargetWrapper, p: Parameters, d: float) -> str:
    fields = [
        f'trial: {trial_i:>6_d} / {p.n_trials:_}',
        f'f_evals: {target.number_of_evaluations:>11_d} / {p.f_evals_max:_}',
        f'f_min: {target.current_f_min:.6e}',
        f'max_rel_dist: {d:.2e}',
    ]
    return '  |  '.join(fields)


def label(use_minimizer, use_pop_thinning):
    result = 'min' if use_minimizer else '---'
    result += ', '
    result += 'thi' if use_pop_thinning else '---'
    return result


def run_in_parallel(
    f: Callable,
    b: Tuple[float, float],
    params: List[Parameters],
    paths: List[Path]
) -> List[MinimizationHistory]:
    """Run a set of simulations in parallel using the multiprocessing module.

    args:
        f (Callable): the function to minimize
        b (Tuple[float, float]): the bounds of the function
        params (List[Parameters]): a list of the parameters for each
            simulation to be run in parallel, the number of simulations is
            `len(params)`
        paths (List[Path]): a list of paths to the directories in which the
            logs for each run will be saved, should be of the same length as
            `params`

    returns:
        (List[MinimizationHistory]): a list of the results of each simulation
    """
    mp_parameters = [[f, b, p, path] for p, path in zip(params, paths)]

    num_processes = 4
    with mp.Pool(num_processes) as pool:
        result = pool.starmap(run, mp_parameters)
    return result


def plot_results(
    parameters: List[Parameters],
    path: Path,
    run_results: List[MinimizationHistory]
):
    # TODO: use set_ax_info and seaborn
    xmax = max(max(history.evaluations) for history in run_results)
    ymin = min(min(history.f_mins) for history in run_results)
    ymax = max(max(history.f_mins) for history in run_results)
    fig, ax = plt.subplots(figsize=(15, 5))
    plot.plot_grid_lines(ax, xmax, ymin, ymax)
    for p, history in zip(parameters, run_results):
        lbl = label(p.use_minimizer, p.pop_thinning_factor)
        ax.plot(history.evaluations, history.f_mins, linewidth=1.5, label=lbl)
    ax.set_xlabel("Number of function evaluations")
    ax.set_ylabel("Best function value")
    ax.set_yscale('log')
    ax.legend()
    ax.set_title(plot_title(parameters[0]), fontsize=9)
    fig.savefig(path / 'evaluations.pdf')
    plt.close(fig)


def plot_title(p: Parameters) -> str:
    return f"target: {p.target_name}, " \
        + f"n_dims: {p.n_dims}, " \
        + f"de_n_pop: {p.de_n_pop}, " \
        + f"de_maxiter: {p.de_maxiter}, " \
        + f"de_strategy: {p.de_strategy}, " \
        + f"pop_thinning_factor: {p.pop_thinning_factor}, " \
        + f"minimization_gtol: {p.minimization_gtol}, " \
        + f"seed: {p.seed}"


def run_one_dimension(d: int) -> None:
    parameters = []
    paths = []
    for use_min in (True, False):
        for pop_thinning_factor in (0.8, 1.0):
            p = Parameters(
                target_name='dark_machines_2',
                n_dims=d,
                use_minimizer=use_min,
                pop_thinning_factor=pop_thinning_factor,
            )

            # reminder: catalogue_parameters has side effects
            paths.append(catalogue_parameters(result_directory(), p))

            parameters.append(p)
    run_results = run_in_parallel(
        tf.functions[parameters[0].target_name],
        tf.bounds[parameters[0].target_name],
        parameters,
        paths
    )

    plot_results(parameters, paths[-1], run_results)


def main():
    for d in [7, 10]:#, 50, 100]:
        run_one_dimension(d)


if __name__ == "__main__":
    main()
