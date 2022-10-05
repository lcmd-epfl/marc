#!/usr/bin/env python

from __future__ import absolute_import

import sys

import numpy as np

from .exceptions import InputError
from .helpers import processargs, setflags
from .molecule import molecule

if __name__ == "__main__" or __name__ == "marc.marc":
    (
        molecules,
        c,
        m,
        plotmode,
        verb,
    ) = processargs(sys.argv[1:])
else:
    exit(1)

# Fill in molecule data
if verb > 0:
    print(f"marc has detected {len(molecules)} molecules in input.")
