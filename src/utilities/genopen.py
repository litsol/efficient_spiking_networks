# SPDX-FileCopyrightText: 2021 Centrum Wiskunde en Informatica
#
# SPDX-License-Identifier: MPL-2.0

"""Takes a sequence of filenames as input and yields a sequence of
file objects that have been suitably open"""

import bz2
import gzip


def gen_open(paths):
    """genopen"""
    for path in paths:
        if path.suffix == ".gz":
            yield gzip.open(path, "rt")
        elif path.suffix == ".bz2":
            yield bz2.open(path, "rt")
        else:
            yield open(path, "rt")  # pylint: disable=R1732, W1514


# Example use

if __name__ == "__main__":
    from pathlib import Path

    lognames = Path("www").rglob("access-log*")
    logfiles = gen_open(lognames)
    for f in logfiles:
        print(f)

# finis

# Local Variables:
# compile-command: "pyflakes genopen.py pylint-3 -d E0401 -f parseable genopen.py" # NOQA, pylint: disable=C0301
# End:
