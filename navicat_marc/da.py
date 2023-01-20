#!/usr/bin/env python

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels


def da_matrix(mols, normalize=True, kernel="rbf", mode="dfs"):
    """
    Compute pairwise main dihedral matrix between all molecules in mols.


    Parameters
    ----------
    mols : list of N molecule objects
    normalize : whether to normalize the matrix or not
    kernel : kernel flavor to process the dihedral angle lists into a pairwise metric
    mode : how to choose the indexing used to traverse the molecular graph

    Returns
    -------
    M : array
        (N,N) matrix
    """
    n = len(mols)
    graphs = [mol.graph for mol in mols]
    refgraph = graphs[0]
    natoms = len(mols[0].atoms)
    M = np.ones((n, n))
    if natoms > 4:
        n_d = max(natoms // 4, 1)
    else:
        return M
    DA = np.zeros((n, n_d))
    coords = np.array([mol.coordinates for mol in mols])

    # All molecules share the same connectivity (at least in principle)
    # Lets traverse the graph and collect dihedrals
    if mode == "auto":
        bc = nx.betweenness_centrality(refgraph, endpoints=True, weight="coulomb_term")
        all_indices = sorted(range(len(bc)), key=lambda i: bc[i])[::-1]
        for i in range(n):
            for d in range(n_d - 1):
                k = 4 * d
                l = 4 * (d + 1)
                a0, a1, a2, a3 = coords[i][all_indices[k:l]]
                DA[i, d] = dihedral(a0, a1, a2, a3)
    elif mode == "dfs":
        dfs_nodes = list(nx.dfs_preorder_nodes(refgraph, source=0))
        n_d = max(len(dfs_nodes) // 4, 1)
        for i in range(n):
            for d in range(n_d - 1):
                k = 4 * d
                l = 4 * (d + 1)
                a0, a1, a2, a3 = coords[i][dfs_nodes[k:l]]
                DA[i, d] = dihedral(a0, a1, a2, a3)
    else:
        return M
    # Now generate a kernel based on the dihedrals
    if kernel == "rbf":
        euclid_0 = np.linalg.norm(DA[:, :] - DA[0, :], axis=0)
        gamma_heuristic = 1 / (euclid_0.std())
        M -= pairwise_kernels(DA, DA, gamma=gamma_heuristic, metric="rbf")
    else:
        M -= pairwise_kernels(DA, DA, metric=kernel)
    M = np.tril(M) + np.triu(M.T, 1)
    if normalize:
        maxval = np.max(M)
        M = np.abs(M) / maxval
    return M, maxval


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
