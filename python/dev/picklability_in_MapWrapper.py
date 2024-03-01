from scipy._lib._util import MapWrapper     # type: ignore


class Target:
    def __new__(cls, *args, **kwargs):
        print('Target.__new__ called')
        return super(Target, cls).__new__(cls)

    def __init__(self):
        print('Target.__init__ called')
        self.f_calls = 0

    def __call__(self, x):
        print(f'Function called with {x = } with object {id(self)}')
        self.f_calls += 1
        return 2*x


def main():
    print('Creating mapper')
    mapper = MapWrapper(pool=1)
    print()

    target = Target()
    print(f'Created target with id {id(target)}')
    print()

    print('Calling the stuff')
    results = mapper(target, [1, 2, 3])
    print()

    print('Results:')
    print(results)
    print(f'{target.f_calls = }')
    print()

    print('Looping through result of mapper call:')
    for e in results:
        print(type(e))
        print(repr(e))
        print(e)

    print(f'{target.f_calls = }')


if __name__ == '__main__':
    main()
