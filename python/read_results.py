"""
Read the results from numerical computations from the results directory, and
plot the scaling of different schemes with dimensionality.
"""

from typing import List
from pathlib import Path

import matplotlib.pyplot as plt

from minimization_history import MinimizationHistory
import plot
from environment_variables import result_directory
from catalogue import csv_to_html


plt.style.use('seaborn-v0_8')
plot.use_tex()


def plot_f_evals_as_function_of_dimensionality():
    def add_series(scheme: str, runs: List[int], label: str, color: str):
        points = []
        crosses = []
        for run in runs:
            path = result_directory() / scheme / str(run)
            history = MinimizationHistory.load_results(path)
            dim = history.dim//2 + 1
            num_evals = history.evaluations[-1]
            if history.solution_found:
                points.append((dim, num_evals))
            else:
                crosses.append((dim, num_evals))
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            marker='o',
            color=color,
            label=label
        )
        ax.scatter(
            [p[0] for p in crosses],
            [p[1] for p in crosses],
            marker='X',
            color=color,
        )

    fig, ax = plt.subplots(figsize=(15, 7))
    add_series(
        'random-gd',
        list(range(10000, 10021)),
        'random gd constr',
        plot.colors('red')[0]
    )

    add_series(
        'random-gd',
        list(range(10042, 10063)),
        'random gd unconstr',
        plot.colors('red')[2]
    )

    add_series(
        'devo',
        list(range(10000, 10021)),
        'devo gd, pt 1.0',
        plot.colors('blue')[0]
    )

    add_series(
        'devo',
        list(range(10021, 10042)),
        'devo gd, pt 0.9',
        plot.colors('blue')[1]
    )

    add_series(
        'devo',
        list(range(10042, 10062)) + [10063],
        'devo gd, pt 0.8',
        plot.colors('green')[0]
    )

    add_series(
        'devo',
        list(range(10064, 10085)),
        'devo gd, pt 0.7',
        plot.colors('orange')[0]
    )

    add_series(
        'devo',
        list(range(10085, 10106)),
        'devo gd, pt 0.6',
        plot.colors('orange')[4]
    )


    ax.set_xticks(list(range(10, 31)))
    title = 'Number of function calls to minimize loss,\n' \
        + 'crosses mark no solution obtained'
    plot.set_ax_info(
        ax,
        xlabel='Dimensionality $d$',
        ylabel='Number of function calls',
        title=title,
    )
    fig.tight_layout()
    fig.savefig('plots/dimensionality_scaling_d10-30.pdf')
    plt.close(fig)


def plot_minimization(
    path: Path,
    title_appendix: str | None = None
):
    history = MinimizationHistory.load_results(path)

    f_mins = history.f_mins
    evals = history.evaluations

    fig, ax = plt.subplots(figsize=(15, 5))
    plot.plot_grid_lines(
        ax,
        xmax=evals[-1],
        ymin=f_mins[-1],
        ymax=f_mins[0],
        xmin=evals[0]
    )

    ax.plot(evals, f_mins)
    ax.set_yscale('log')
    title = 'Minimization'
    if title_appendix is not None:
        title += f' ({title_appendix})'
    plot.set_ax_info(
        ax,
        xlabel='Number of function evaluations',
        ylabel='Best function value',
        title=title,
        legend=False
    )
    fig.tight_layout()
    fig.savefig(path / 'minimization.pdf')
    plt.close(fig)


def plot_interesting_phase_of_minimization(
    path: Path,
    title_appendix: str | None = None
):
    history = MinimizationHistory.load_results(path)

    # find index where last vector started improving the results by starting
    # from the back and locating the index where the slope suddenly becomes 10
    # times as large (in abs value) as the slope between the previous two
    # points
    f_mins = history.f_mins
    evals = history.evaluations
    idx = len(f_mins) - 1
    keep_going = True
    right_slope = 0.0
    while keep_going:
        left_slope = (f_mins[idx - 10] - f_mins[idx]) / (evals[idx - 10] - evals[idx])
        # stop if left side is less than a tenth as steep as right side
        keep_going = left_slope < 0.1*right_slope
        # move to the left
        idx -= 1
        right_slope = left_slope

    # print(f'Keeping results from {idx = }, where {evals[idx] = }')

    # keep only information from the last phase of minimization
    f_mins = f_mins[idx:]
    evals = history.evaluations[idx:]

    fig, ax = plt.subplots(figsize=(15, 5))
    plot.plot_grid_lines(
        ax,
        xmax=evals[-1],
        ymin=f_mins[-1],
        ymax=f_mins[0],
        xmin=evals[0]
    )

    ax.plot(evals, f_mins)
    ax.set_yscale('log')
    title = 'Final phase of minimization'
    if title_appendix is not None:
        title += f' ({title_appendix})'
    plot.set_ax_info(
        ax,
        xlabel='Number of function evaluations (including previous vectors)',
        ylabel='Best function value',
        title=title,
        legend=False
    )
    fig.tight_layout()
    fig.savefig(path / 'minimization_of_last_vector.pdf')
    plt.close(fig)


def plot_history_of_individual_samples():
    # method = 'random-gd'
    # method = 'devo'
    method = 'shgo'

    # for i in range(10039, 10043):
    samples = [
        # 10000,
        # 10001,
        # 10002,
        # 10003,
        # 10004,
        # 10005,
        # 10009,
        # 10010,
        # 10011,
        # 10012,
        # 10014,
        # 10018,
        # 10019,
        # 10020,
        # 10021,
        # 10027,
        # 10029,
        # 10036,
        # 10037,
        10038,
        10039,
    ]
    for i in samples:
        path = result_directory() / f'{method}/{i}'
        print('Sample', i)

        plot_minimization(path, title_appendix=f'{method} {i}')

        history = MinimizationHistory.load_results(path)
        if history.solution_found:
            print('Plotting final phase')
            plot_interesting_phase_of_minimization(
                path,
                title_appendix=f'{method} {i}'
            )
        else:
            print('Skipping final phase')


def generate_registry_of_histories(directory: Path):
    sub_dirs = sorted([e for e in directory.iterdir() if e.is_dir()])
    lines = ['Sample,Success,Time']
    for p in sub_dirs:
        try:
            h = MinimizationHistory.load_results(p)
            lines.append(f'{p.name},{h.solution_found},{h.elapsed_time}')
        except FileNotFoundError:
            lines.append(f'{p.name},No record (stopped/killed),No record (stopped/killed)')

    # write to csv file
    csv_file = directory / 'results.csv'
    with csv_file.open('w', encoding='UTF-8') as f:
        f.write('\n'.join(lines))

    # write to html file
    html_file = directory / 'results.html'
    csv_to_html(csv_file, html_file)


if __name__ == '__main__':
    # plot_f_evals_as_function_of_dimensionality()

    # plot_interesting_phase_of_minimization(
        # result_directory() / 'random-gd/1029'
    # )

    plot_history_of_individual_samples()

    # generate_registry_of_histories(result_directory() / 'shgo')
