#!/usr/bin/env python

import os

from navicat_marc.helpers import test_molecules_from_file
from navicat_marc.molecule import (
    test_compare_origin,
    test_molecule_from_file,
    test_molecule_from_lines,
    test_molecule_to_file,
)

test_files_dir = f"{os.path.dirname(os.path.abspath(__file__))}/test_files/"

if __name__ == "__main__":
    test_compare_origin(path=test_files_dir)
    test_molecule_from_lines()
    test_molecule_from_file(path=test_files_dir)
    test_molecule_to_file(path=test_files_dir)
    test_molecules_from_file(path=test_files_dir)
