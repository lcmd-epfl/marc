#!/usr/bin/env python

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from navicat_marc.helpers import at_eq, b_eq


def rmsd_matrix(mols, sort=False, truesort=False, normalize=True):
    """
    Compute pairwise RMSD matrix between all molecules in mols.

    Parameters
    ----------
    mols : list of N molecule objects

    Returns
    -------
    M : array
        (N,N) matrix
    maxval : maximum pairwise RMSD
    """
    n = len(mols)
    M = np.zeros((n, n))
    pos_ibj = []
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            if sort:
                GM = nx.algorithms.isomorphism.GraphMatcher(
                    mols[i].graph, mols[j].graph, at_eq, b_eq
                )
                iviews = [
                    (
                        np.array(list(im.keys()), dtype=int),
                        np.array(list(im.values()), dtype=int),
                    )
                    for im in GM.isomorphisms_iter()
                ]
                min_res = np.inf
                for k, (ibi, ibj) in enumerate(iviews):
                    ibi, ibj = ibi[ibi.argsort()], ibj[ibi.argsort()]
                    res, rotated_coordinates = kabsch_rmsd(
                        mols[i].coordinates, mols[j].coordinates[ibj]
                    )
                    if res < min_res:
                        save_ibj = ibj
                        min_res = res
                        save_rotated_coordinates = rotated_coordinates
                    if np.isclose(0, min_res):
                        break
                if len(iviews) == 0:
                    min_res, save_rotated_coordinates = kabsch_rmsd(
                        mols[i].coordinates, mols[j].coordinates
                    )
                    assert all(mols[i].atoms == mols[j].atoms)
                    mols[j].update(mols[j].atoms, save_rotated_coordinates)
                else:
                    pos_ibj.append(save_ibj)
                    assert all(mols[i].atoms == mols[j].atoms[save_ibj])
                    mols[j].update(mols[j].atoms[save_ibj], save_rotated_coordinates)
                M[i, j] = M[j, i] = min_res
            if not sort and len(pos_ibj) > 0:
                for k, ibj in enumerate(pos_ibj):
                    res, rotated_coordinates = kabsch_rmsd(
                        mols[i].coordinates, mols[j].coordinates[ibj]
                    )
                    if res < min_res:
                        save_ibj = ibj
                        min_res = res
                        save_rotated_coordinates = rotated_coordinates
                    if np.isclose(0, min_res):
                        break
                assert all(mols[i].atoms == mols[j].atoms[save_ibj])
                mols[j].update(mols[j].atoms[save_ibj], save_rotated_coordinates)
                M[i, j] = M[j, i] = min_res
            if not sort and len(pos_ibj) == 0:
                res, rotated_coordinates = kabsch_rmsd(
                    mols[i].coordinates, mols[j].coordinates
                )
                assert all(mols[i].atoms == mols[j].atoms)
                mols[j].update(mols[j].atoms, rotated_coordinates)
                M[i, j] = M[j, i] = res
        if not truesort:
            sort = False
    maxval = np.max(M)
    max_idxs = np.unravel_index(M.argmax(), M.shape)
    if normalize:
        M = np.abs(M) / maxval
    return M, maxval


def kabsch_rmsd(P, Q):
    """
    Rotate matrix P unto Q using Kabsch algorithm and calculate the RMSD.

    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    rmsd : float
        root-mean squared deviation
    Q : array
        (N,D) matrix, where N is points and D is dimension,
        rotated
    """
    Q = kabsch_rotate(Q, P)
    return rmsd(P, Q), Q


def rmsd(V, W):
    """
    Calculate Root-mean-square deviation from two sets of vectors V and W.

    Parameters
    ----------
    V : array
        (N,D) matrix, where N is points and D is dimension.
    W : array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    rmsd : float
        Root-mean-square deviation

    """
    return np.sum((V - W) ** 2) / len(V)


def kabsch_rotate(P, Q):
    """
    Rotate matrix P unto matrix Q using Kabsch algorithm.

    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    P : array
        (N,D) matrix, where N is points and D is dimension,
        rotated

    """
    U = kabsch(P, Q)

    # Rotate P
    P = np.dot(P, U)
    return P


def kabsch(P, Q):
    """
    The optimal rotation matrix U is calculated and then used to rotate matrix
    P unto matrix Q so the minimum root-mean-square deviation (RMSD) can be
    calculated.

    Using the Kabsch algorithm with two sets of paired point P and Q, centered
    around the centroid. Each vector set is represented as an NxD
    matrix, where D is the the dimension of the space.

    The algorithm works in three steps:
    - a translation of P and Q
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U

    http://en.wikipedia.org/wiki/Kabsch_algorithm

    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    U : matrix
        Rotation matrix (D,D)

    """

    # Computation of the covariance matrix
    C = np.dot(np.transpose(P), Q)
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U = np.dot(V, W)

    return U


def quaternion_rmsd(P, Q):
    """
    Rotate matrix P unto Q and calculate the RMSD

    based on doi:10.1016/1049-9660(91)90036-O

    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    P : array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    rmsd : float
    """
    rot = quaternion_rotate(P, Q)
    P = np.dot(P, rot)
    return rmsd(P, Q)


def quaternion_transform(r):
    """
    Get optimal rotation
    note: translation will be zero when the centroids of each molecule are the
    same
    """
    Wt_r = makeW(*r).T
    Q_r = makeQ(*r)
    rot = Wt_r.dot(Q_r)[:3, :3]
    return rot


def makeW(r1, r2, r3, r4=0):
    """
    matrix involved in quaternion rotation
    """
    W = np.asarray(
        [[r4, r3, -r2, r1], [-r3, r4, r1, r2], [r2, -r1, r4, r3], [-r1, -r2, -r3, r4]]
    )
    return W


def makeQ(r1, r2, r3, r4=0):
    """
    matrix involved in quaternion rotation
    """
    Q = np.asarray(
        [[r4, -r3, r2, r1], [r3, r4, -r1, r2], [-r2, r1, r4, r3], [-r1, -r2, -r3, r4]]
    )
    return Q


def quaternion_rotate(X, Y):
    """
    Calculate the rotation

    Parameters
    ----------
    X : array
        (N,D) matrix, where N is points and D is dimension.
    Y: array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    rot : matrix
        Rotation matrix (D,D)
    """
    N = X.shape[0]
    W = np.asarray([makeW(*Y[k]) for k in range(N)])
    Q = np.asarray([makeQ(*X[k]) for k in range(N)])
    Qt_dot_W = np.asarray([np.dot(Q[k].T, W[k]) for k in range(N)])
    W_minus_Q = np.asarray([W[k] - Q[k] for k in range(N)])
    A = np.sum(Qt_dot_W, axis=0)
    eigen = np.linalg.eigh(A)
    r = eigen[1][:, eigen[0].argmax()]
    rot = quaternion_transform(r)
    return rot


def centroid(X):
    """
    Calculate the centroid from a vectorset X.

    https://en.wikipedia.org/wiki/Centroid
    Centroid is the mean position of all the points in all of the coordinate
    directions.

    C = sum(X)/len(X)

    Parameters
    ----------
    X : array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    C : float
        centeroid

    """
    C = X.mean(axis=0)
    return C


def reorder_hungarian(
    p_atoms,
    q_atoms,
    p_coord,
    q_coord,
):
    """
    Re-orders the input atom list and xyz coordinates using the Hungarian
    method (using optimized column results)
    Parameters
    ----------
    p_atoms : array
        (N,1) matrix, where N is points holding the atoms' names
    q_atoms : array
        (N,1) matrix, where N is points holding the atoms' names
    p_coord : array
        (N,D) matrix, where N is points and D is dimension
    q_coord : array
        (N,D) matrix, where N is points and D is dimension
    Returns
    -------
    view_reorder : array
             (N,1) matrix, reordered indexes of atom alignment based on the
             coordinates of the atoms
    """

    # hview = reorder_hungarian(
    #    mols[i].atoms,
    #    mols[j].atoms,
    #    mols[i].coordinates,
    #    mols[j].coordinates,
    # )
    # mols[j].update(mols[j].atoms[hview], mols[j].coordinates[hview])

    # Find unique atoms
    unique_atoms = np.unique(p_atoms)

    # generate full view from q shape to fill in atom view on the fly
    view_reorder = np.zeros(q_atoms.shape, dtype=int)
    view_reorder -= 1

    for atom in unique_atoms:
        (p_atom_idx,) = np.where(p_atoms == atom)
        (q_atom_idx,) = np.where(q_atoms == atom)

        A_coord = p_coord[p_atom_idx]
        B_coord = q_coord[q_atom_idx]

        view = hungarian(A_coord, B_coord)
        view_reorder[p_atom_idx] = q_atom_idx[view]

    return view_reorder


def hungarian(A, B):
    """
    Hungarian reordering.
    Assume A and B are coordinates for atoms of SAME type only
    """

    distances = cdist(A, B, "euclidean")

    # Perform Hungarian analysis on distance matrix between atoms of 1st
    # structure and trial structure
    indices_a, indices_b = linear_sum_assignment(distances)

    return indices_b


def reorder_distance(
    p_atoms,
    q_atoms,
    p_coord,
    q_coord,
):
    """
    Re-orders the input atom list and xyz coordinates by atom type and then by
    distance of each atom from the centroid.
    Parameters
    ----------
    atoms : array
        (N,1) matrix, where N is points holding the atoms' names
    coord : array
        (N,D) matrix, where N is points and D is dimension
    Returns
    -------
    atoms_reordered : array
        (N,1) matrix, where N is points holding the ordered atoms' names
    coords_reordered : array
        (N,D) matrix, where N is points and D is dimension (rows re-ordered)
    """

    # Find unique atoms
    unique_atoms = np.unique(p_atoms)

    # generate full view from q shape to fill in atom view on the fly
    view_reorder = np.zeros(q_atoms.shape, dtype=int)

    for atom in unique_atoms:
        (p_atom_idx,) = np.where(p_atoms == atom)
        (q_atom_idx,) = np.where(q_atoms == atom)

        A_coord = p_coord[p_atom_idx]
        B_coord = q_coord[q_atom_idx]

        # Calculate distance from each atom to centroid
        A_norms = np.linalg.norm(A_coord, axis=1)
        B_norms = np.linalg.norm(B_coord, axis=1)

        reorder_indices_A = np.argsort(A_norms)
        reorder_indices_B = np.argsort(B_norms)

        # Project the order of P onto Q
        translator = np.argsort(reorder_indices_A)
        view = reorder_indices_B[translator]
        view_reorder[p_atom_idx] = q_atom_idx[view]

    return view_reorder
