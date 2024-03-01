from pathlib import Path

from catalogue import catalogue_parameters, Parameters


class ModelSpecificParameters(Parameters):
    def __init__(self):
        self.learning_rate = 0.5

    def __str__(self):
        return 'SpecificModel'

    @property
    def use_acceleration(self) -> bool:
        return True


if __name__ == '__main__':
    p = ModelSpecificParameters()
    catalogue_parameters(Path('dev'), p)
