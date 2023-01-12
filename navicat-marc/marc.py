#!/usr/bin/env python

from __future__ import absolute_import

import sys

import numpy as np

from .clustering import (
    affprop_clustering,
    agglomerative_clustering,
    kmeans_clustering,
    plot_dendrogram,
)
from .da import da_matrix
from .erel import erel_matrix
from .exceptions import InputError
from .helpers import processargs
from .molecule import Molecule
from .rmsd import rmsd_matrix

if __name__ == "__main__" or __name__ == "marc.marc":
    (
        basename,
        molecules,
        dof,
        c,
        m,
        n_clusters,
        ewin,
        mine,
        plotmode,
        verb,
    ) = processargs(sys.argv[1:])
else:
    exit(1)

# Fill in molecule data
l = len(molecules)
if verb > 0:
    if n_clusters is not None:
        print(
            f"marc has detected {l} molecules in input.\n Will select {n_clusters} most representative conformers using {c} clustering and {m} as metric."
        )
    elif n_clusters is None:
        print(
            f"marc has detected {l} molecules in input.\n Will automatically select a set of representative conformers using {c} clustering and {m} as metric."
        )
    if ewin is not None:
        print(f"An energy window of {ewin} in energy units will be applied.")


# Generate the desired metric matrix
if m in ["rmsd", "ewrmsd", "mix"]:
    rmsd_matrix, max = rmsd_matrix(molecules)
    if plotmode > 1:
        plot_dendrogram(rmsd_matrix * max, "RMSD", verb)
    A = rmsd_matrix

if m in ["erel", "ewrmsd", "ewda", "mix"]:
    energies = [molecule.energy for molecule in molecules]
    if None in energies:
        if verb > 2:
            print(f"Energies are: {energies}")
        raise InputError(
            """One or more molecules do not have an associated energy. Cannot use
             energy metrics. Exiting."""
        )
    erel_matrix, max = erel_matrix(molecules)
    if plotmode > 1:
        plot_dendrogram(erel_matrix * max, "E_rel", verb)
    A = erel_matrix

if m in ["da", "ewda", "mix"]:
    da_matrix = da_matrix(molecules, mode="dfs")
    if plotmode > 1:
        plot_dendrogram(da_matrix, "Dihedral", verb)
    A = da_matrix

# Mix the metric matrices if desired
if m == ["ewrmsd"]:
    A = rmsd_matrix * erel_matrix

if m == ["ewrda"]:
    A = da_matrix * erel_matrix

if m == ["mix"]:
    A = da_matrix * erel_matrix * rmsd_matrix

# Time to cluster after scaling the matrix. Scaling choice is not innocent.

if c == "kmeans":
    indices, clusters = kmeans_clustering(n_clusters, A, dof, verb)

if c == "agglomerative":
    indices, clusters = agglomerative_clustering(n_clusters, A, dof, verb)

if c == "affprop":
    indices, clusters = affprop_clustering(A, verb)

# Resample clusters based on energies if requested

if mine:
    if None in energies:
        if verb > 2:
            print(f"Energies are: {energies}")
        raise InputError(
            """One or more molecules do not have an associated energy. Cannot use
             the mine option. Exiting."""
        )
    for i, (index, cluster) in enumerate(zip(indices, clusters)):
        if verb > 2:
            print(
                f"Resampling {list(cluster)} for which representative {indices[i]} has been selected."
            )
        idx_mine = cluster[
            np.argmin(np.array([energies[j] for j in cluster], dtype=float))
        ]
        if index != idx_mine:
            if verb > 2:
                print(f"Minimal energy representative {idx_mine} was selected instead.")
            indices[i] = idx_mine

# If requested, prune again based on average cluster energies

if ewin is not None:
    rejected = np.zeros((len(indices)))
    energies = [molecule.energy for molecule in molecules]
    if None in energies:
        if verb > 2:
            print(f"Energies are: {energies}")
        raise InputError(
            """One or more molecules do not have an associated energy. Cannot use
             the ewin option. Exiting."""
        )
    avgs = np.zeros(len(clusters), dtype=float)
    stds = np.zeros(len(clusters), dtype=float)
    for i, cluster in enumerate(clusters):
        if verb > 2:
            print(
                f"Going through {list(cluster)} for which representative {indices[i]} was selected."
            )
        energies = np.array(
            [molecule.energy for molecule in molecules[cluster]], dtype=float
        )
        avgs[i] = energies.mean()
        stds[i] = energies.std() * 0.5
    lowest = np.min(avgs)
    accepted = list(np.where(avgs < ewin + lowest + stds)[0])
    if verb > 2:
        print(f"Accepted indices are {accepted}.")
    for i, idx in enumerate(indices):
        if i in accepted:
            if verb > 1:
                print(
                    f"Accepting selected conformer number {idx} due to energy threshold."
                )
        else:
            if verb > 1:
                print(
                    f"Removed selected conformer number {idx} due to energy threshold."
                )
            rejected[i] = 1

# Write the indices (representative molecules) that were accepted and rejected

if ewin is not None:
    for i, idx in enumerate(indices):
        if rejected[i]:
            molecules[idx].write(f"{basename}_rejected_{idx:02}")
        if not rejected[i]:
            molecules[idx].write(f"{basename}_selected_{idx:02}")
else:
    for i, idx in enumerate(indices):
        molecules[idx].write(f"{basename}_selected_{idx:02}")
