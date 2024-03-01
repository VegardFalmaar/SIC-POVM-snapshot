from pathlib import Path
from minimization_history import MinimizationHistory


def main():
    history = MinimizationHistory.load_results(
        Path('experimental_results/shgo/10025')
    )
    print(history.evaluations)
    print(history.f_mins)
    print(history.x_bests)


if __name__ == '__main__':
    main()
