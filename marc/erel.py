#!/usr/bin/env python

import numpy as np


def erel_matrix(mols):
    """
    Compute pairwise relative energy matrix between all molecules in mols.

    Parameters
    ----------
    mols : list of N molecule objects

    Returns
    -------
    M : array
        (N,N) matrix
    """
    energies = np.array([mol.energy for mol in mols])
    n = len(mols)
    M = np.zeros((n, n))
    for i, _ in range(n):
        for j in range(i, n - 1):
            M[i, j] = M[j, i] = np.abs(energies[i] - energies[j])
    return M
