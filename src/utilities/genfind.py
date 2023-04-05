# SPDX-FileCopyrightText: 2021 Centrum Wiskunde en Informatica
# SPDX-License-Identifier: MPL-2.0

"""
A function that generates files that match a given filename pattern
"""

from pathlib import Path


def gen_find(filepat, top):
    """
    gen_find
    """

    yield from Path(top).rglob(filepat)


# Example use

if __name__ == "__main__":
    lognames = gen_find("access-log*", "www")
    for name in lognames:
        print(name)

# finis

# Local Variables:
# compile-command: "pyflakes genfind.py; pylint-3 -d E0401 -f parseable genfind.py" # NOQA, pylint: disable=C0301
# End:
