# SPDX-FileCopyrightText: 2021 Centrum Wiskunde en Informatica
# SPDX-License-Identifier: MPL-2.0

"""
Grep a sequence of lines that match a re pattern
"""

import re


def gen_grep(pat, lines):
    """
    gen_grep
    """

    patc = re.compile(pat)
    return (line for line in lines if patc.search(line))


# Example use

if __name__ == "__main__":
    from pathlib import Path

    from .gencat import gen_cat
    from .genopen import gen_open

    lognames = Path("www").rglob("access-log*")
    logfiles = gen_open(lognames)
    loglines = gen_cat(logfiles)

    # Look for ply downloads (PLY is my own Python package)
    plylines = gen_grep(r"ply-.*\.gz", loglines)
    for line in plylines:
        print(line, end="")

# finis

# Local Variables:
# compile-command: "pyflakes gengrep.py; pylint-3 -d E0401 -f parseable gengrep.py" # NOQA, pylint: disable=C0301
# End:
