#!/usr/bin/env python

import numpy as np


def erel_matrix(mols, normalize=True):
    """
    Compute pairwise relative energy matrix between all molecules in mols.

    Parameters
    ----------
    mols : list of N molecule objects
    normalize : whether to scale to the [0,1] range. Default: True

    Returns
    -------
    M : array
        (N,N) matrix
    max : maximum relative energy difference
    """
    energies = np.array([mol.energy for mol in mols])
    n = len(mols)
    M = np.zeros((n, n))
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = np.abs(energies[i] - energies[j])
    if normalize:
        maxval = np.max(np.abs(M))
        M = np.abs(M) / maxval
    return M, maxval
