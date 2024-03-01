"""
Functionality for running ``scipy.optimize.minimize`` with randomly initialized
points in parameter space.

Current problem is that running on test function dark machines 2 and sometimes
also on the SIC-POVM loss function gives warnings
stating that Hessian matrix for updating the trial vector during optimization
should be set to zero. Don't know how to elegantly do this without effecting
the results of SIC-POVM minimization, so it has been left as-is so far.
"""

from typing import Callable, Tuple
import os
from pathlib import Path

import numpy as np
import scipy    # type: ignore
import matplotlib.pyplot as plt

import target_functions as tf
import formatting
import plot
from minimization_history import TargetWrapper, MinimizationHistory
from gradient_descent import Parameters
from catalogue import catalogue_parameters
from log import get_logger

formatting.set_numpy_print_options()


def run(
    target_function: Callable,
    bounds: Tuple[float, float],
    parameters: Parameters,
    path: Path
) -> MinimizationHistory:
    """Run the gradient descent algorithm.

    Uses ``scipy.optimize.minimize`` to minimize the target function for
    randomly initialized points in parameter space, stopping if the function
    value drops below a certain threshold. The results obtained during the
    minimization process are stored and returned.

    args:
        target_function (Callable): the function to be minimized
        bounds (Tuple[float, float]): the bounds (min, max) of the elements in
            the array passed as input to the target function, it is assumed
            that all elements have the same bounds
        parameters (Parameters): the hyperparameters used for the
            optimization
        path (Path): directory in which logs and results will be saved

    returns:
        (MinimizationHistory): the results obtained during the minimization
            process
    """
    logger = get_logger(path, name=f'random-gd-{path.name}')

    if parameters.seed:
        logger.debug('Setting random seed %d', parameters.seed)
        np.random.seed(parameters.seed)

    target = TargetWrapper(target_function, parameters.n_dims)
    target.history.start_timing()
    x_min, x_max = bounds

    def initialize_vector():
        if parameters.target_name == 'SICPOVM':
            # create (N + 1)-dimensional normalized vector, and remove one
            # dimension to create an N-dimensional subnormalized vector
            x0 = x_min + (x_max - x_min) \
                * np.random.random(parameters.n_dims + 1)
            x0 /= np.linalg.norm(x0)
            x0 = x0[:-1]
        else:
            x0 = x_min + (x_max - x_min) * np.random.random(parameters.n_dims)
        return x0

    for trial_i in range(parameters.n_trials):
        result_i = scipy.optimize.minimize(
            target,
            x0=initialize_vector(),
            method='trust-constr' if parameters.use_constraints else 'L-BFGS-B',
            bounds=[bounds]*parameters.n_dims,
            constraints=parameters.get_constraints(),
            options=parameters.get_options()
        )

        logger.info(info_line(trial_i, target, parameters))
        logger.debug('Result of minimization %d:\n%s', trial_i, result_i)

        if target.current_f_min < parameters.minimization_threshold:
            logger.info("Converged due to current_f_min close to 0.0.")
            target.history.solution_found = True
            break

        stop = target.number_of_evaluations >= parameters.f_evals_max
        if stop:
            logger.info('Maximum number of function evaluations reached')
            break

    target.history.stop_timing()
    target.append_best_evaluation()
    target.history.save_results(path)

    logger.info('Elapsed time: %.3f seconds', target.history.elapsed_time)
    logger.info(target)
    return target.history


def info_line(trial_i: int, target: TargetWrapper, p: Parameters):
    fields = [
        f'trial {trial_i:>6_d} / {p.n_trials:_}',
        f'f_evals {target.number_of_evaluations:>11_d} / {p.f_evals_max:_}',
        f'f_min {target.current_f_min:.6e}'
    ]
    return '  |  '.join(fields)


def plot_results(
    history: MinimizationHistory,
    parameters: Parameters,
    path: str
):
    plt.style.use('seaborn-v0_8')
    plot.use_tex()

    xmax = max(history.evaluations)

    fig, ax = plt.subplots(figsize=(15, 5))
    plot.plot_grid_lines(ax, xmax, min(history.f_mins), max(history.f_mins))

    ax.plot(
        history.evaluations,
        history.f_mins,
        linewidth=1.5,
        label='random + minimize'
    )

    ax.set_yscale('log')

    plot.set_ax_info(
        ax,
        xlabel='Number of function evaluations',
        ylabel='Best function value',
        title=plot_title(parameters),
    )

    fig.tight_layout()
    fig.savefig(os.path.join(path, 'evaluations.pdf'))
    plt.close(fig)


def plot_title(p: Parameters) -> str:
    return f"target: {p.target_name}, " \
        + f"n_dims: {p.n_dims}, " \
        + f"minimization_gtol: {p.minimization_gtol}, " \
        + f"seed: {p.seed}"


def main():
    """Run the gradient descent algorithm on a test function.

    Demonstrates the usage of the gradient descent functions on a test
    function.
    """
    target_name = 'dark_machines_2'
    n_dims = 10
    parameters = Parameters(target_name, n_dims)
    path = catalogue_parameters('experimental_results', parameters)

    history = run(
        tf.functions[target_name],
        tf.bounds[target_name],
        parameters,
        path
    )
    history.save_results(path)
    plot_results(history, parameters, path)


if __name__ == "__main__":
    main()
