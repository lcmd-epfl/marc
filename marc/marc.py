#!/usr/bin/env python

from __future__ import absolute_import

import sys

import numpy as np
import networkx as nx

from .da import da_matrix
from .erel import erel_matrix
from .exceptions import InputError
from .helpers import processargs
from .molecule import Molecule
from .rmsd import rmsd_matrix

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

# Check for atom ordering
for molecule_a, molecule_b in zip(molecules, molecules[1:]):
    atoms_a = molecule_a.atoms
    atoms_b = molecule_b.atoms
    if all(atoms_a == atoms_b):
        continue
    else:
        if verb > 0:
            print("Molecule geometries are not sorted.")
        sort = True
        break
    sort = False


# Check for isomorphism
for molecule_a, molecule_b in zip(molecules, molecules[1:]):
    g_a = molecule_a.graph
    g_b = molecule_b.graph
    if nx.is_isomorphic(g_a, g_b):
        continue
    else:
        if verb > 0:
            print("Molecule topologies are not isomorphic.")
        isomorph = False
        break
    isomorph = True

# Generate metric matrices
if m in ["rmsd", "ewrmsd", "mix"]:
    rmsd_matrix = rmsd_matrix(molecules)

if m in ["erel", "ewrmsd", "ewda", "mix"]:
    energies = [molecule.energy for molecule in molecules]
    if None in energies:
        raise InputError(
            "One or more molecules do not have an associated energy. Cannot use energy metrics. Exiting."
        )
    erel_matrix = erel_matrix(molecules)

if m in ["da", "ewda", "mix"]:
    da_matrix = da_matrix(molecules)
