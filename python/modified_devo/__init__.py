from gradient_descent import BaseParameters


class Parameters(BaseParameters):
    def __init__(
        self,
        target_name: str,
        n_dims: int,
        use_minimizer: bool,
        pop_thinning_factor: float = 0.8
    ):
        super().__init__(target_name, n_dims)
        self._use_minimizer = use_minimizer
        assert pop_thinning_factor <= 1.0
        self._pop_thinning_factor = pop_thinning_factor

    def __str__(self):
        return 'devo'

    @property
    def de_n_pop(self) -> int:
        return max(15 * self.n_dims, 50)

    @property
    def n_trials(self) -> int:
        return int(1e8 / self.de_n_pop)

    @property
    def use_x0_insertion(self) -> bool:
        return True

    @property
    def de_maxiter(self) -> int:
        return 2  # int(1 * n_dims)

    @property
    def de_strategy(self) -> str:
        return 'best1exp'  # 'best1exp', 'best1bin', 'rand1bin'

    @property
    def pop_thinning_factor(self) -> float:
        return self._pop_thinning_factor

    @property
    def max_rel_dist_threshold(self) -> float:
        return 1e-7 * self.n_dims

    @property
    def use_constraints(self) -> bool:
        return False

    @property
    def use_minimizer(self) -> bool:
        return self._use_minimizer
