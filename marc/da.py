#!/usr/bin/env python

import networkx as nx
import numpy as np


def da_matrix(mols):
    """
    Compute pairwise main dihedral matrix between all molecules in mols.


    Parameters
    ----------
    mols : list of N molecule objects

    Returns
    -------
    M : array
        (N,N) matrix
    """
    graphs = np.array([mol.graph for mol in mols])
    coords = np.array([mol.coordinates for mol in mols])
    n = len(mols)
    M = np.zeros((n, n))
    # All molecules share the same connectivity (at least in principle)
    bc = nx.betweenness_centrality(graphs[0], normalized=True, endpoints=True)
    indices = sorted(range(len(bc)), key=lambda i: bc[i])[-4:]
    for i, _ in range(0, n - 1):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = delta_dihedral(indices, coords[i], coords[j])
    return M


def delta_dihedral(indices, P, Q):
    """
    Compute dihedral angle difference between two sets based on indices.

    Parameters
    ----------
    indices : four integers from which to compute dihedrals.
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    delta : float
        dihedral deviation.
    """
    a0, a1, a2, a3 = P[indices]
    b0, b1, b2, b3 = Q[indices]
    da = dihedral(a0, a1, a2, a3)
    db = dihedral(b0, b1, b2, b3)
    return da - db


def dihedral(p0, p1, p2, p3):
    """
    Compute dihedral angle given by four points.

    Parameters
    ----------
    p0 : float.
        coordinates of point.
    p1 : float.
        coordinates of point.
    p2 : float.
        coordinates of point.
    p3 : float.
        coordinates of point.

    Returns
    -------
    d : float
        dihedral angle in radians.
    """

    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.arctan2(y, x)
