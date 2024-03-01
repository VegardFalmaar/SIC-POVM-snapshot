"""
File to be run on Betzy.

Import structure does not work the same there for some reason. Running the
``main.py`` files in the subdirectories on Betzy (from this directory, not from
within the subdirectories) results in ``ModuleNotFoundError``.
"""

from modified_devo.main import main


if __name__ == '__main__':
    main()
