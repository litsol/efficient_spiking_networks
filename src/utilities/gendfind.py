# SPDX-FileCopyrightText: 2021 Centrum Wiskunde en Informatica
#
# SPDX-License-Identifier: MPL-2.0
# gendfind.py
#

""" A function that generates files that match a given regex pattern """

import os
import re


def gen_dfind(dirpat, top):
    """Traverse the directorys and yield the results."""
    regexp = re.compile(dirpat)
    for path, dirlist, _ in os.walk(top):
        for name in [dir for dir in dirlist if regexp.search(dir) is not None]:
            yield os.path.join(path, name)


# Example use

if __name__ == "__main__":
    print(list(gen_dfind(r"^(?!_).*", "www")))

# finis

# Local Variables:
# compile-command: "pyflakes gendfind.py; pylint-3 -d E0401 -f parseable gendfind.py" # NOQA, pylint: disable=C0301
# End:
