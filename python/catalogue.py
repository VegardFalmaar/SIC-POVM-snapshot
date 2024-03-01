"""
Module with functionality for keeping track of the parameters used for
different runs of numerical calculations. The outside interface consists of the
``Parameters`` class and the ``catalogue_parameters`` function. The rest are
intended only for use internal within this module.
"""

from typing import Dict, Union
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod

from pandas import read_csv     # type: ignore


class Parameters(ABC):
    """Abstract base class for storing parameters for numerical calculations.

    When combined with the ``catalogue_parameters`` function, all attributes
    (both instance variables and @property's)
    not starting with an underscore (_) will be treated as model parameters and
    saved to the registry. Callable methods will be ignored.

    Magic method ``__str__`` must be overridden. This defines the name given to
    the directory in which parameters and results will be catalogued and saved.

    In order to use the built-in functionality for loading the results from a
    registry file using the ``load`` function, all the parameters must be
    setable, and ``__init__`` can take no arguments. Thus, if the parameters
    are defined as @property's, there must be corresponding setters allowing
    the values to be set.
    """
    @abstractmethod
    def __str__(self) -> str:
        """Defines the name of the type of simulation.

        returns:
            (str): the desired name of the type of simulation you are running.
        """
        raise NotImplementedError

    @classmethod
    def load(cls, parent_dir: Path, sample_id: int) -> 'Parameters':
        """Load parameters from the registry file.

        args:
            parent_dir (Path): the directory in which the desired registry file
                exists
            sample_id (int): the id (e.g. 10013) of the sample whose parameters
                you would like to load from file

        returns:
            (subclass of Parameters): the loaded parameters
        """
        # TODO: implement
        pass


def catalogue_parameters(parent_dir: Path, parameters: Parameters) -> Path:
    """Save the parameters passed as input to the registry file.

    args:
        parent_dir (Path): the directory in which the parameters and results
            will be saved.
        parameters (subclass of Parameters):
            instance of a subclass of the Parameter class which has the
            parameters for the numerical calculations as instance variables.

    returns:
        (Path): full path to the directory in which results for this particular
            calculation should be saved.
    """
    assert parent_dir.is_dir()
    path = parent_dir / str(parameters)
    if not path.is_dir():
        path.mkdir()
        create_registry_files(path, parameters)
    verify_correct_directory(path, parameters)
    sample_dir = append_sample(path, parameters)
    full_path_to_sample_dir = path / sample_dir
    full_path_to_sample_dir.mkdir()
    return full_path_to_sample_dir


def expand_registry(registry_file: Path, kw: str, val: str) -> None:
    """Expand an existing regitry to include a new parameter.

    args:
        registry_file (Path): the registry file to update
        kw (str): the keyword of the new parameter you wish to include
        val (str): the value to set for the new parameter in the already
            existing samples in the registry

    returns:
        None
    """
    assert registry_file.is_file()

    with registry_file.open('r', encoding='UTF-8') as f:
        fields = f.readline().strip().split(',')

        samples = []
        for line in f.readlines():
            values = line.strip().split(',')
            d = dict(zip(fields, values))
            if kw in d:
                raise ValueError(f'Keyword \'{kw}\' already exists in registry')
            d[kw] = val
            samples.append(d)

    new_fields = fields[:2] + sorted(fields[2:] + [kw])

    lines = [','.join(new_fields)]
    for s in samples:
        lines.append(','.join(s[f] for f in new_fields))

    with registry_file.open('w', encoding='UTF-8') as f:
        f.write('\n'.join(lines) + '\n')


def extract_parameters(
    p: Parameters
) -> Dict[str, Union[str, int, float, bool]]:
    """Extract the parameters from an instance of the ``Parameter`` class.

    args:
        p (Parameters): the hyperparameters of a numerical calculation.

    returns:
        (Dict[str, Union[str, int, float]]): the parameters as a dictionary on
            the form {parameter name: parameter value}
    """
    attributes = [
        a for a in dir(p)
        if not a.startswith('_') and not callable(getattr(p, a))
    ]
    return {a: getattr(p, a) for a in sorted(attributes)}


def create_registry_files(path: Path, parameters: Parameters):
    fields = ['Sample', 'Time'] + list(extract_parameters(parameters).keys())
    file = path / 'registry.csv'
    with file.open('w', encoding='UTF-8') as f:
        f.write(','.join(fields) + '\n')
    csv_to_html(file, path / 'registry.html')


def verify_correct_directory(path: Path, parameters: Parameters):
    file = path / 'registry.csv'
    with file.open('r', encoding='UTF-8') as f:
        expected_fields = f.readline().strip().split(',')

    observed = ['Sample', 'Time'] + list(extract_parameters(parameters).keys())
    msg = f'Expected fields that exist in the registry ({expected_fields}) ' \
        + f'do not match observed fields ({observed})'
    assert expected_fields == observed, msg


def append_sample(path: Path, parameters: Parameters) -> str:
    """Create an id for the new sample and append it to the registry.

    The new id will be one number larger than the previous sample, even if
    there are intermittent samples missing. The first sample in a fresh
    registry is 10 000.

    args:
        path (Path): the path to the directory in which the registry and
            samples are stored.
        parameters (Parameters): the parameters to the simulation which should
            be stored.

    returns:
        (str): the id of the new sample
    """
    registry_file = path / 'registry.csv'

    with registry_file.open('r', encoding='UTF-8') as f:
        last_line = [line.strip() for line in f.readlines()][-1]

    previous_sample = last_line.split(',')[0]
    if previous_sample == 'Sample':
        sample = str(10_000)
    else:
        sample = str(int(previous_sample) + 1)

    time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

    line = [sample, time] \
        + [str(a) for a in extract_parameters(parameters).values()]

    with registry_file.open('a', encoding='UTF-8') as f:
        f.write(','.join(line) + '\n')

    csv_to_html(registry_file, path / 'registry.html')

    return sample


def csv_to_html(csv_file: Path, html_file: Path):
    """Create an HTML copy of a CSV file using Pandas.
    """
    assert csv_file.is_file(), f'File {csv_file.name} does not exist'
    df = read_csv(csv_file)
    df.to_html(html_file, index=False)


if __name__ == '__main__':
    reg = Path('/home/vegard/Documents/SIC-POVM/code/experimental_results/shgo/registry.csv')
    expand_registry(reg, 'sampling_method', 'simplicial')
