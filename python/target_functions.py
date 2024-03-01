from typing import Dict, Tuple, Callable

import numpy as np


def rosenbrock(x):
    """The Rosenbrock function, 2-dim"""
    res = sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
    return res


def rastrigin(x):
    """The Rastrigin function, n-dim"""
    d = x.shape[0]
    res = 10. * d + np.sum(x * x - 10. * np.cos(2 * np.pi * x))
    return res


def rastrigin_shifted(x):
    """The Rastrigin function with shifted minimum, n-dim"""
    d = x.shape[0]
    x_shift = x + 0.23
    res = 10. * d + np.sum(x_shift * x_shift - 10. * np.cos(2 * np.pi * x_shift))
    return res


def ackley(x, a=20, b=0.2, c=2*np.pi):
    """The Ackley function, n-dim"""
    d = x.shape[0]
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    res = term1 + term2 + a + np.exp(1.0)
    return res


def griewank(x):
    """The Griewank function, n-dim"""
    d = x.shape[0]
    indices = np.arange(1,d+1)
    s = np.sum(x**2 / 400.)
    p = np.prod(np.cos(x / np.sqrt(indices)))
    res = s - p + 1.
    return res


def schwefel(x):
    """The Schwefel function, n-dim"""
    d = x.shape[0]
    s = np.sum(x * np.sin(np.sqrt(np.abs(x))))
    res = 418.9829 * d - s
    return res


def dark_machines_1(x):
    """Analytical function 1 in arxiv:2101.04525, n-dim"""
    s1 = np.sum( ((x - 2.) / 15.)**6 )
    s2 = np.sum( (x - 2.)**2 )
    p1 = np.prod( np.cos(x - 2.)**2 )
    res = np.exp(-s1) - 2 * np.exp(-s2) * p1
    # Adding 1.0 to get f_min = 0.0
    res += 1.0
    return res


def dark_machines_2(x):
    """Analytical function 2 in arxiv:2101.04525, n-dim"""
    # Function 2 is just a shifted Rastrigin function
    return rastrigin_shifted(x)


def dark_machines_3(x):
    """Analytical function 3 in arxiv:2101.04525, n-dim"""
    d = x.shape[0]
    res = -1. / d * np.sum( np.sin(5 * np.pi * (x**0.75 - 0.05))**6 )
    # Adding 1.0 to get f_min = 0.0
    res += 1.0
    return res


def dark_machines_4(x):
    """Analytical function 4 in arxiv:2101.04525, n-dim"""
    # Function 4 is just the Schwefel function
    return schwefel(x)


#
# Global dictionaries with function names and parameter bounds
#

functions: Dict[str, Callable] = {
    'rosenbrock': rosenbrock,
    'rastrigin': rastrigin,
    'ackley': ackley,
    'griewank': griewank,
    'schwefel': schwefel,
    'dark_machines_1': dark_machines_1,
    'dark_machines_2': dark_machines_2,
    'dark_machines_3': dark_machines_3,
    'dark_machines_4': dark_machines_4,
}

bounds: Dict[str, Tuple[float, float]] = {
    'rosenbrock': (-5.0, 10.0),
    'rastrigin': (-5.12, 5.12),
    'ackley':    (-32.768, 32.768),
    'griewank':  (-600., 600.),
    'schwefel':  (-500., 500.),
    'dark_machines_1':  (-30., 30.),
    'dark_machines_2':  (-7, 7),
    'dark_machines_3':  (0.0, 1.0),
    'dark_machines_4':  (-500., 500.),
}
