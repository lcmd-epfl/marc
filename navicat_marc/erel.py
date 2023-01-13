#!/usr/bin/env python

import numpy as np


def erel_matrix(mols, normalize=True):
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
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = np.abs(energies[i] - energies[j])
    if normalize:
        max = np.max(M)
        M = np.abs(M) / max
    return M, max
