#!/usr/bin/env python

import sys

import numpy as np
import scipy

from navicat_marc.clustering import (
    affprop_clustering,
    agglomerative_clustering,
    kmeans_clustering,
    plot_dendrogram,
    plot_tsne,
    unique_nm,
)
from navicat_marc.da import da_matrix
from navicat_marc.distatis import run_distatis
from navicat_marc.erel import erel_matrix
from navicat_marc.exceptions import InputError
from navicat_marc.helpers import processargs
from navicat_marc.molecule import Molecule
from navicat_marc.rmsd import rmsd_matrix

from navicat_marc import helpers

def run_marc():
    (
        basename,
        molecules,
        dof,
        c,
        m,
        n_clusters,
        ewin,
        mine,
        sort,
        truesort,
        plotmode,
        verb,
        is_write
    ) = processargs(sys.argv[1:])

    indices, clusters = run_marc_from_args(basename = basename, 
                                           molecules = molecules, 
                                           dof = dof, 
                                           c = c, 
                                           m = m, 
                                           n_clusters = n_clusters,
                                           ewin = ewin,
                                           mine = mine,
                                           sort = sort,
                                           truesort = truesort,
                                           plotmode = plotmode,
                                           verb = verb,
                                           is_write = is_write
                                           )

    return

def run_marc_from_args(basename, 
                       molecules, 
                       dof: int = None, 
                       c: str = 'kmeans', 
                       m:str = 'avg', 
                       n_clusters: int = None, 
                       ewin: float = None, 
                       mine: bool = False, 
                       sort: bool = False, 
                       truesort: bool = False, 
                       plotmode: int = 0, 
                       verb: int = 1,
                       is_write: bool = False
                       ):
    """
        Clusters conformer data into conformer ensembles. 
        Details are found in the github folder: https://github.com/lcmd-epfl/marc
        and the paper: https://doi.org/10.1021/acs.jpclett.4c01657

        Args:
        -----
            basename: 

            molecules:

            dof: (Optional) Integer
                Default: None
                Degrees of freedom.
                If not specified, calculated from the given molecules.

            c: (Optional) String
                Default: kmeans
                Clustering algorithm used.
                Options: kmeans, agglomerative, affprop

            m: (Optional) String
                Default: avg
                Metric used to define distance between clusters.
                Options: rmsd, erel, da, ewrmsd, ewda, mix, avg

            n_clusters: (Optional) Integer
                Default: None
                Number of representatitive conformers to select. Gap method is used by default.

            ewin: (Optional) Float
                Default: None
                Energy window for the conformers to be accepted in kcal/mol.

            mine: (Optional) Boolean
                Default: False
                If set, the minimum energy conformer per sample will be taken instead of the centermost one.
                True if energies are available for all conformers.

            sort: (Optional) Boolean
                Default: False
                If True, will atempt to sort molecular geometries within isomorphisms w.r.t. the first structure.
                Time consuming.

            truesort: (Optional) Boolean
                Default: False

            plotmode: (Optional) Integer
                Default: 0
                Set to 1 to generate agglomerative dendrograms and to 2 to also generate TSNE plots.

            verb: (Optional) Integer
                Default: 1 
                Verbosity level of the code, Higher is more verbose and vice versa.

            is_write: (Optional) Boolean
                Default: False
                Indicates if the run should change filenames of the used files.

        Returns:
        --------
            indices: List
                Indices of the representative molecule in a cluster.
                Correlates to cluster with the same index.

            clusters: List of lists
                Clustered conformer ensembles.
                Representative molecule has same index in indices list.

    """
    if dof == None: # If degrees of freedom is not specified, then calculate.
        dof, _ = helpers.get_dof(molecules, verb = verb)


    # Fill in molecule data
    l = len(molecules)
    energies = [molecule.energy for molecule in molecules]  # Retrieve the energy from every molecule in the list if applicable.

    if verb > 0:
        if n_clusters is not None:
            print(
                f"marc has detected {l} molecules in input.\n Will select {n_clusters} most representative conformers using {c} clustering and {m} as metric."
            )
        elif n_clusters is None:
            print(
                f"marc has detected {l} molecules in input.\n Will automatically select a set of representative conformers using {c} clustering and {m} as metric."
            )
        
        if not None in energies:
            print(
                f"Energies for each conformer have been provided.\n mine will be set to True."
            )
            # mine = True
        if ewin is not None:
            print(f"An energy window of {ewin} kcal/mol will be applied.")

    # ------------------------------------------------------------------------------------------ # 
    # Generate the desired metric matrix
    # ---------------------------------- #

    # ---------------------------------- #
    # RMSD matrix 
    # ---------------------------------- # 

    if m in ["rmsd", "ewrmsd", "mix", "avg"]:
        rmsd_m, rmsd_max = rmsd_matrix(molecules, sort=sort, truesort=truesort)
        if plotmode > 0:
            plot_dendrogram(rmsd_m * rmsd_max, "RMSD", verb)
        A = rmsd_m
        if verb > 4:
            print("\n The rmsd dissimilarity matrix is :\n")
            print(np.array_str(A, precision=2, suppress_small=True))
            print(
                f"Before normalization, largest dissimilarity was {rmsd_max} angstrom."
            )

    # ---------------------------------- #
    # Relative energy matrix 
    # ---------------------------------- # 

    if m in ["erel", "ewrmsd", "ewda", "mix", "avg"] or (ewin is not None) or mine:
        if None in energies:
            if verb > 2:
                print(f"Energies are: {energies}")
            raise InputError(
                """One or more molecules do not have an associated energy. Cannot use
                 energy metrics, mine or ewin. Exiting."""
            )
        erel_m, erel_max = erel_matrix(molecules)
        if plotmode > 0:
            plot_dendrogram(erel_m * erel_max, "E_rel", verb)
        A = erel_m
        if verb > 4:
            print("\n The relative energy dissimilarity matrix is :\n")
            print(np.array_str(A, precision=2, suppress_small=True))
            print(
                f"Before normalization, largest dissimilarity was {erel_max} kcal/mol."
            )

    # ---------------------------------- #
    # Dihedral angles 
    # ---------------------------------- # 

    if m in ["da", "ewda", "mix", "avg"]:
        da_m, da_max = da_matrix(molecules, mode="dfs")
        if plotmode > 0:
            plot_dendrogram(da_m * da_max, "Dihedral", verb)
        A = da_m
        if verb > 4:
            print("\n The dihedral angle dissimilarity matrix is :\n")
            print(np.array_str(A, precision=2, suppress_small=True))
            print(
                f"Before normalization, largest dissimilarity was {da_max} adimensional units (kernel of dihedral angle vectors)."
            )

    # ---------------------------------- #
    # Mix the metric matrices if desired
    # ---------------------------------- # 

    if m == "ewrmsd":
        A = run_distatis([erel_m, rmsd_m], verb)

    if m == "ewda":
        A = run_distatis([erel_m, da_m], verb)

    if m == "mix":
        A = run_distatis([da_m, erel_m, rmsd_m], verb)

    if m == "avg":
        A = (erel_m + da_m + rmsd_m) / 3

    if verb > 3:
        print("\n The current dissimilarity matrix is :\n")
        print(np.array_str(A, precision=2, suppress_small=True))
        print(
            f"Writing dissimilarity matrix (with full precision) to dm.npy in the working directory."
        )
        np.save("dm.npy", A)

    # ------------------------------------------------------------------------------------------ # 
    # Cluster using the dissimilarity matrix of choice
    # ------------------------------------------------ # 

    if c == "kmeans":
        indices, clusters = kmeans_clustering(n_clusters, A, dof, verb)

    if c == "agglomerative":
        indices, clusters = agglomerative_clustering(n_clusters, A, dof, verb)

    if c == "affprop":
        indices, clusters = affprop_clustering(A, verb)

    # ------------------------------------------------ # 
    # Removal of duplicates.
    # ------------------------------------------------ # 

    curr_n = len(indices)
    effA = A[indices, :][:, indices]
    if verb > 3:
        print(
            f"\n The dissimilarity matrix of the selected conformers {indices} is :\n"
        )
        print(np.array_str(effA, precision=2, suppress_small=True))
        print(
            f"Writing dissimilarity matrix of selected conformers (with full precision) to edm.npy in the working directory."
        )
        np.save("edm.npy", effA)

    n, umask = unique_nm(effA, verb)
    if n < curr_n:
        print(
            f"Reduced the number of selected conformers by {curr_n - n} due to uniqueness."
        )
        if n < n_clusters:
            raise InputError(
                """The desired number of clusters is larger than the number of unique entries according to the chosen similarity matrix. Exiting."""
            )
        indices = list(np.array(indices, dtype=int)[umask])
        clusters = list(np.array(clusters, dtype=object)[umask])

    # ------------------------------------------------ # 
    # Resample clusters based on energies if requested
    # ------------------------------------------------ # 

    if mine:
        if None in energies:
            if verb > 2:
                print(f"Energies are: {energies}")
            raise InputError(
                """One or more molecules do not have an associated energy. Cannot use
                 the mine option. Exiting."""
            )
        gmine = np.min(np.array(energies, dtype=float))
        for i, (index, cluster) in enumerate(zip(indices, clusters)):
            if verb > 2:
                print(
                    f"Resampling {list(cluster)} for which representative {indices[i]} has been selected."
                )
            idx_mine = cluster[
                np.argmin(np.array([energies[j] for j in cluster], dtype=float))
            ]
            if verb > 3:
                cenergies = np.around(
                    np.array([energies[j] for j in cluster]) - gmine, decimals=4
                )
                print(f"The corresponding relative energies were: {list(cenergies)}")
            if index != idx_mine:
                if verb > 2:
                    print(
                        f"Minimal energy representative {idx_mine} was selected instead."
                    )
                indices[i] = idx_mine

    # ------------------------------------------------ # 
    # If requested, prune again based on average 
    # cluster energies
    # ------------------------------------------------ # 

    if ewin is not None:
        rejected = np.zeros((len(indices)))
        if None in energies:
            if verb > 2:
                print(f"Energies are: {energies}")
            raise InputError(
                """One or more molecules do not have an associated energy. Cannot use
                 the ewin option. Exiting."""
            )
        avgs = np.zeros(len(clusters), dtype=float)
        stds = np.zeros(len(clusters), dtype=float)
        repes = np.zeros(len(clusters), dtype=float)
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
            repes[i] = molecules[indices[i]].energy
        lowest = np.min(avgs)
        accepted = list(np.where(avgs < ewin + lowest + stds)[0])
        quasiaccepted = list(np.where(repes < ewin + lowest)[0])
        if verb > 2:
            print(f"Accepted indices are {accepted}.")
        for i, idx in enumerate(indices):
            if i in accepted:
                if verb > 1:
                    print(
                        f"Accepting selected conformer number {idx} due to energy threshold for the entire cluster."
                    )
            elif i in quasiaccepted:
                if verb > 1:
                    print(
                        f"Accepting selected conformer number {idx} due to energy threshold for the lowest conformer (in spite of the cluster being high in energy)."
                    )
            else:
                if verb > 1:
                    print(
                        f"Removed selected conformer number {idx} due to energy threshold."
                    )
                rejected[i] = 1

    # ------------------------------------------------------------------------------------------ # 
    # Output name generation
    # ------------------------ # 

    outnames = [molecule.name for molecule in molecules]
    if None in outnames:
        format_string = f"0{max(int(np.floor(np.log10(l)) + 1), 2)}d"
        outnames = [f"{basename}_{idx:{format_string}}" for idx in range(l)]

    # ------------------------ #
    # Plot tsne
    # ------------------------ #

    if plotmode > 1:
        plot_tsne(A, indices, clusters, outnames)

    # ------------------------ #
    # Write the indices (representative molecules) that were accepted and rejected
    # ------------------------ #

    if is_write:
        if ewin is not None:
            for i, idx in enumerate(indices):
                if rejected[i]:
                    molecules[idx].write(f"{outnames[idx]}_rejected")
                if not rejected[i]:
                    molecules[idx].write(f"{outnames[idx]}_accepted")
        else:
            for i, idx in enumerate(indices):
                molecules[idx].write(f"{outnames[idx]}_accepted")

    return indices, clusters
