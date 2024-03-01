import numpy as np
from scipy.optimize import (    # type: ignore
    LinearConstraint,
    NonlinearConstraint
)

import catalogue


class BaseParameters(catalogue.Parameters):
    """Base class for storing parameters that are common for GD and DEVO.
    """
    def __init__(self, target_name: str, n_dims: int):
        assert target_name in ['SICPOVM', 'dark_machines_2', 'dark_machines_4']
        self._target_name = target_name
        self._n_dims = n_dims

    @property
    def target_name(self) -> str:
        return self._target_name

    @property
    def n_dims(self) -> int:
        return self._n_dims

    @property
    def use_constraints(self):
        return False

    @property
    def minimization_gtol(self) -> float:
        """The minimization will stop when the norm of the gradient drops below
        this value.
        """
        return 1e-11

    @property
    def minimization_threshold(self) -> float:
        """Threshold for considering the minimization a success.

        The minimization will stop when the current smallest function value
        obtained drops below this threshold.
        """
        return 1e-13

    @property
    def seed(self) -> int:
        """Random seed used to initilized the numpy random seed generator.
        """
        # seed = np.random.randint(int(1e5), int(1e6))
        return 585997

    @property
    def f_evals_max(self) -> int:
        """The maximum number of evaluations of the target function allowed
        before the minimization will be aborted.
        """
        return int(5e7)

    def get_constraints(self):
        """Get constraints on the input variables to the target function.

        Callable method instead of property to not be saved as a parameter.
        """
        if self.use_constraints and self.target_name == 'SICPOVM':
            return NonlinearConstraint(np.linalg.norm, 0.0, 1.0)

        if self.use_constraints:
            raise NotImplementedError(
                'Constraints are not implement for other problems than SICPOVM'
            )

        return None

    def get_options(self):
        """"Options dictionary passed to ``scipy.optimize.minimize``.

        See the description of the different parameters in the last return
        statement.
        """
        if self.use_constraints:
            return {
                'gtol': self.minimization_gtol,
                'disp': False,
            }
        return {
            # gtol sets a convergence limit based on the gradient, stops if
            # the biggest component of (a scaled version of) the gradient
            # (with respect to the bounds) is lower than this threshold
            'gtol': self.minimization_gtol,

            # ftol sets a convergence limit based on the function values
            # obtained, it is scaled by 0.1 compared to the convergence
            # criterion specified in the parameters due to some treatment
            # internal to scipy which modifies the threshold based on the
            # machine precision of float on the current device
            'ftol': self.minimization_threshold * 0.1,

            'disp': False,  # do not display extra information
            'maxls': 20,    # default
        }


class Parameters(BaseParameters):
    def __str__(self):
        return 'random-gd'

    @property
    def n_trials(self) -> int:
        """The max number of different initial vectors will be tried before
        aborting.
        """
        return int(1e5)


if __name__ == '__main__':
    # c = BaseParameters('test', 10)
    c = Parameters('test', 10)
