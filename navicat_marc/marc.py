#!/usr/bin/env python

from __future__ import absolute_import

import sys

import numpy as np

from navicat_marc.clustering import (
    affprop_clustering,
    agglomerative_clustering,
    kmeans_clustering,
    plot_dendrogram,
)
from navicat_marc.da import da_matrix
from navicat_marc.erel import erel_matrix
from navicat_marc.exceptions import InputError
from navicat_marc.helpers import processargs
from navicat_marc.molecule import Molecule
from navicat_marc.rmsd import rmsd_matrix


def main():
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
        rmsd_m, max = rmsd_matrix(molecules)
        if plotmode > 1:
            plot_dendrogram(rmsd_m * max, "RMSD", verb)
        A = rmsd_m

    if m in ["erel", "ewrmsd", "ewda", "mix"]:
        energies = [molecule.energy for molecule in molecules]
        if None in energies:
            if verb > 2:
                print(f"Energies are: {energies}")
            raise InputError(
                """One or more molecules do not have an associated energy. Cannot use
                 energy metrics. Exiting."""
            )
        erel_m, max = erel_matrix(molecules)
        if plotmode > 1:
            plot_dendrogram(erel_m * max, "E_rel", verb)
        A = erel_m

    if m in ["da", "ewda", "mix"]:
        da_m = da_matrix(molecules, mode="dfs")
        if plotmode > 1:
            plot_dendrogram(da_m, "Dihedral", verb)
        A = da_m

    # Mix the metric matrices if desired
    if m == ["ewrmsd"]:
        A = rmsd_m * erel_m

    if m == ["ewrda"]:
        A = da_m * erel_m

    if m == ["mix"]:
        A = da_m * erel_m * rmsd_m

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
                    print(
                        f"Minimal energy representative {idx_mine} was selected instead."
                    )
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


if __name__ == "__main__" or __name__ == "navicat_marc.marc":
    main()
    exit()
