"""
Functionality for running ``scipy.optimize.shgo``.
"""

from typing import Callable, Tuple
from pathlib import Path

import scipy    # type: ignore
import numpy as np

import formatting
from minimization_history import TargetWrapper, MinimizationHistory
from log import get_logger
from gradient_descent import BaseParameters

formatting.set_numpy_print_options()


class Parameters(BaseParameters):
    def __init__(self,
            target_name: str,
            n_dims: int,
            sampling_method: str,
            use_constraints: bool
        ):
        super().__init__(target_name, n_dims)
        assert sampling_method in ['simplicial', 'sobol', 'halton']
        self._sampling_method = sampling_method
        self._use_constraints = use_constraints

    def __str__(self):
        return 'shgo'

    @property
    def use_constraints(self):
        return self._use_constraints

    @property
    def log_minimization_history(self):
        return True

    @property
    def minimize_every_iter(self):
        return True

    @property
    def sampling_method(self):
        return self._sampling_method


def run(
    target_function: Callable,
    bounds: Tuple[float, float],
    p: Parameters,
    path: Path
) -> MinimizationHistory:
    """Run the simplicial homology global optimization algorithm.

    Uses ``scipy.optimize.shgo`` to minimize the target function. The results
    obtained during the minimization process are stored and returned.

    args:
        target_function (Callable): the function to be minimized
        bounds (Tuple[float, float]): the bounds (min, max) of the elements in
            the array passed as input to the target function, it is assumed
            that all elements have the same bounds
        p (GDParameters): the hyperparameters used for the
            optimization
        path (Path): directory in which logs and results will be saved

    returns:
        (MinimizationHistory): the results obtained during the minimization
            process
    """
    logger = get_logger(path, name=f'shgo-{path.name}')

    if p.seed:
        logger.debug('Setting random seed %d', p.seed)
        np.random.seed(p.seed)

    target = TargetWrapper(target_function, p.n_dims)
    target.history.start_timing()

    constraints = p.get_constraints() if p.use_constraints else None

    result = scipy.optimize.shgo(
        func=target,
        bounds=[bounds]*p.n_dims,
        workers=1,
        sampling_method=p.sampling_method,
        # iters=6,
        constraints=constraints,
        options={
            'f_min': 0.0,
            'f_tol': p.minimization_threshold,
            'minimize_every_iter': p.minimize_every_iter,
            'maxfev': p.f_evals_max,   # passed on to local minimizer
            'disp': True,
        },
        minimizer_kwargs={
            'method': 'trust-constr' if p.use_constraints else 'L-BFGS-B',
            'options': p.get_options()
        }
    )

    if target.current_f_min < p.minimization_threshold:
        logger.info('Converged due to current_f_min close to 0.0.')
        target.history.solution_found = True

    target.history.stop_timing()
    target.append_best_evaluation()
    target.history.save_results(path)

    logger.info('Elapsed time: %.3f seconds', target.history.elapsed_time)
    logger.info('Return value of SHGO:\n%s', result)
    logger.info(target)
    return target.history
