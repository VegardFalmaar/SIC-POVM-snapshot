import multiprocessing as mp
import math
import time


global_start = time.perf_counter()


def f(x, t):
    a = 0.0
    for i in range(t*10000000):
        a = math.sin(a*i/1/0.4/0.2/8)
    return x*3


def main1():
    num_procs = 6
    with mp.Pool(num_procs) as pool:
        res = pool.starmap(f, [[f'Hello number {i}', 3*(i+1)] for i in range(num_procs)])
    print(res)


def my_sleep(t: int):
    start = time.perf_counter() - global_start
    time.sleep(t)
    stop = time.perf_counter() - global_start
    print(f'{mp.current_process().name}, {t = } seconds, {start = }, {stop = }')


def main2():
    num_procs = 3
    times = [[i] for i in [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]]
    with mp.Pool(num_procs) as pool:
        _ = pool.starmap(my_sleep, times)


if __name__ == '__main__':
    main2()
