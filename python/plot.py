from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt


def colors(c: str) -> List[str]:
    """Get hex colors pleasantly viewed with seaborn style of plotting.

    args:
        c (str): string code for the desired color category.
            Should be one of ['blue', 'red', 'green', 'orange']

    returns:
        (List[str]): list of hexadecimal color codes in the desired category
    """
    if c == 'blue':
        return ['#7382D4', '#2E2465', '#5D6383', '#495282', '#484561']

    if c == 'red':
        return ['#B46C9F','#B54993', '#91467A', '#A52E5B', '#D64D81']

    if c == 'green':
        return ['#8FB46C', '#8BC971', '#86B342', '#6D955C', '#586F4E']

    if c == 'orange':
        return ['#F8C23D', '#D8B93E', '#E1C900', '#D38521', '#D96917']

    raise ValueError(f'Unknown color key \'{c}\'')


def use_tex():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "DejaVu Sans",
        "font.serif": ["Computer Modern"]}
    )
    # for e.g. \text command
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


def set_ax_info(ax, xlabel, ylabel, title=None, zlabel=None, legend=True):
    """Write title and labels on an axis with the correct fontsizes.

    Args:
        ax (matplotlib.axis): the axis on which to display information
        xlabel (str): the desired label on the x-axis
        ylabel (str): the desired label on the y-axis
        title (str, optional): the desired title on the axis
            default: None
        zlabel (str, optional): the desired label on the z-axis for 3D-plots
            default: None
        legend (bool, optional): whether or not to add labels/legend
            default: True
    """
    if zlabel is None:
        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=15)
        # ax.ticklabel_format(style='plain')
    else:
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_zlabel(zlabel, fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.ticklabel_format(style='scientific', scilimits = (-2, 2))
    if title is not None:
        ax.set_title(title, fontsize=20)
    if legend:
        ax.legend(fontsize=15)


def plot_grid_lines(ax, xmax, ymin, ymax, xmin=0):
    y = 1.0
    while y > ymax:
        y *= 0.1
    y *= 10
    while y > ymin:
        ax.plot([xmin, xmax], [y, y], linestyle='--', linewidth=0.5, color='0.5')
        y *= 0.1
    ax.plot([xmin, xmax], [0.0, 0.0], linestyle='--', linewidth=0.5, color='0.0')
