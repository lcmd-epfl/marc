#!/usr/bin/env python

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import scipy.cluster.hierarchy
import scipy.cluster.vq
from scipy.spatial.distance import euclidean, squareform
from sklearn.cluster import DBSCAN, AffinityPropagation, AgglomerativeClustering, KMeans
from sklearn.manifold import MDS
from sklearn.neighbors import NearestCentroid

from navicat_marc.exceptions import UniqueError


def plot_dendrogram(m: np.ndarray, label: str, verb=0):
    if verb > 0:
        print(f"Max pairwise {label}: {np.max(m)} in {label} units.")
        if verb > 2:
            print(f"Distance matrix is:\n {m}")
    reduced_distances = squareform(m, force="tovector")
    linkage = scipy.cluster.hierarchy.linkage(reduced_distances, method="single")
    plt.title(f"{label} average linkage hierarchical clustering")
    dn = scipy.cluster.hierarchy.dendrogram(
        linkage, no_labels=True, count_sort="descendent"
    )
    if verb > 0:
        print(f"Saving dendrogram plot as {label}_dendrogram.png in working directory.")
    plt.savefig(f"{label}_dendrogram.png")
    plt.close()


def kmeans_clustering(n_clusters, m: np.ndarray, rank=5, verb=0):
    mds = MDS(
        dissimilarity="precomputed",
        n_components=rank,
        n_init=100,
        normalized_stress="auto",
    )
    x = mds.fit_transform(m)

    if verb > 5:
        print("The current feature matrix is :\n")
        with np.printoptions(threshold=np.inf):
            print(np.array_str(x, precision=2, suppress_small=True))
        print(
            f"Writing feature matrix (with full precision) to fm.npy in the working directory."
        )
        np.save("fm.npy", x)

    if n_clusters is None:
        nm = m.shape[0]
        percentages = sorted(
            list(
                set(
                    [
                        min(max(int(nm * percentage), 2), nm - 1)
                        for percentage in [
                            0.01,
                            0.05,
                            0.1,
                            0.15,
                            0.2,
                            0.25,
                            0.3,
                            0.35,
                            0.4,
                            0.45,
                            0.5,
                        ]
                    ]
                )
            )
        )
        gaps = gap(x, nrefs=min(nm, 50), ks=percentages, verb=verb)
        n_clusters = max(percentages[np.argmax(gaps)], 2)
        n_unique, rank = unique_nr(m, verb=verb)
        n_clusters = min(n_unique, n_clusters)
    km = KMeans(n_clusters=n_clusters, n_init=100)
    cm = km.fit_predict(x)
    u, c = np.unique(cm, return_counts=True)
    closest_pt_idx = []
    clusters = []
    if verb > 1:
        print(
            f"{u[-1]+1} unique clusters found: {u} \nWith counts: {c} \nAdding up to {np.sum(c)}"
        )
    for iclust in range(u.size):

        # get all points assigned to each cluster:
        clusters.append(np.where(km.labels_ == iclust)[0])

        # get all indices of points assigned to this cluster:
        cluster_pts_indices = np.where(km.labels_ == iclust)[0]

        cluster_cen = km.cluster_centers_[iclust]
        min_idx = np.argmin(
            [euclidean(x[idx], cluster_cen) for idx in cluster_pts_indices]
        )

        # Testing:
        if verb > 1:
            print(
                f"Closest index of point to cluster {iclust} center has index {cluster_pts_indices[min_idx]:02}"
            )
        closest_pt_idx.append(cluster_pts_indices[min_idx])
    return closest_pt_idx, clusters


def affprop_clustering(m, verb=0):
    m = np.ones_like(m) - m
    ap = AffinityPropagation(affinity="precomputed")
    cm = ap.fit_predict(m)
    u, c = np.unique(cm, return_counts=True)
    closest_pt_idx = []
    clusters = []
    if verb > 1:
        print(
            f"{u[-1]+1} unique clusters found: {u} \nWith counts: {c} \nAdding up to {np.sum(c)}"
        )
    for iclust in range(u.size):

        # get all points assigned to each cluster:
        cluster_pts = m[ap.labels_ == iclust]
        clusters.append(np.where(ap.labels_ == iclust)[0])

        min_idx = ap.cluster_centers_indices_[iclust]

        # Testing:
        if verb > 1:
            print(
                f"Point in {iclust} center has index {ap.cluster_centers_indices_[iclust]:02}"
            )
        closest_pt_idx.append(ap.cluster_centers_indices_[iclust])
    return closest_pt_idx, clusters


def agglomerative_clustering(n_clusters, m: np.ndarray, rank=5, verb=0):
    if n_clusters is None:
        mds = MDS(
            dissimilarity="precomputed",
            n_components=rank,
            n_init=100,
            normalized_stress="auto",
        )
        x = mds.fit_transform(m)
        nm = m.shape[0]
        percentages = sorted(
            list(
                set(
                    [
                        min(max(int(nm * percentage), 2), nm - 1)
                        for percentage in [
                            0.01,
                            0.05,
                            0.1,
                            0.15,
                            0.2,
                            0.25,
                            0.3,
                            0.35,
                            0.4,
                            0.45,
                            0.5,
                        ]
                    ]
                )
            )
        )
        gaps = gap(x, nrefs=min(nm, 50), ks=percentages, verb=verb)
        n_clusters = max(percentages[np.argmax(gaps)], 2)
        n_unique, rank = unique_nr(m, verb=verb)
        n_clusters = min(n_unique, n_clusters)
    m = np.ones_like(m) - m
    ac = AgglomerativeClustering(
        n_clusters=n_clusters, affinity="precomputed", linkage="single"
    )
    cm = ac.fit_predict(m)
    clf = NearestCentroid()
    clf.fit(m, cm)
    u, c = np.unique(cm, return_counts=True)
    closest_pt_idx = []
    clusters = []
    if verb > 1:
        print(
            f"{u[-1]+1} unique clusters found: {u} \nWith counts: {c} \nAdding up to {np.sum(c)}"
        )
    for iclust in range(u.size):

        # get all points assigned to each cluster:
        cluster_pts = m[ac.labels_ == iclust]
        clusters.append(np.where(ac.labels_ == iclust)[0])

        # get all indices of points assigned to this cluster:
        cluster_pts_indices = np.where(ac.labels_ == iclust)[0]

        cluster_cen = clf.centroids_[iclust]
        min_idx = np.argmin(
            [euclidean(m[idx], cluster_cen) for idx in cluster_pts_indices]
        )

        # Testing:
        if verb > 1:
            print(
                f"Closest index of point to cluster {iclust} center has index {cluster_pts_indices[min_idx]:02}"
            )
        closest_pt_idx.append(cluster_pts_indices[min_idx])
    return closest_pt_idx, clusters


def gap(data, refs=None, nrefs=20, ks=range(1, 11), verb=0):
    shape = data.shape
    if refs is None:
        tops = data.max(axis=0)
        bots = data.min(axis=0)
        dists = scipy.matrix(scipy.diag(tops - bots))
        rands = np.random.random_sample(size=(shape[0], shape[1], nrefs))
        for i in range(nrefs):
            rands[:, :, i] = rands[:, :, i] * dists + bots
    else:
        rands = refs
    gaps = np.zeros((len(ks),))
    for (i, k) in enumerate(ks):
        km = KMeans(n_clusters=k, n_init=5)
        cm = km.fit_predict(data)
        kmc = km.cluster_centers_
        kml = km.labels_
        disp = sum([euclidean(data[m, :], kmc[kml[m], :]) for m in range(shape[0])])
        refdisps = np.zeros((rands.shape[2],))
        for j in range(rands.shape[2]):
            km = KMeans(n_clusters=k, n_init=5)
            cm = km.fit_predict(rands[:, :, j])
            kmc = km.cluster_centers_
            kml = km.labels_
            refdisps[j] = sum(
                [euclidean(rands[m, :, j], kmc[kml[m], :]) for m in range(shape[0])]
            )
        gaps[i] = scipy.mean(scipy.log(refdisps)) - scipy.log(disp)
        if verb > 3:
            print(f"Gaps for k-values {ks} : {gaps}")
    return gaps


def unique_nr(data, verb=0):
    data = np.round(data, decimals=4)
    a, idxs = np.unique(data, axis=0, return_index=True)
    umask = np.array([x in idxs for x in range(data.shape[0])], dtype=bool)
    uidx = np.where(umask == True)[0]
    n = uidx.size
    r = np.linalg.matrix_rank(data, hermitian=True)
    if n < 2:
        raise UniqueError(
            "It seems like all the structures are the same or extremely similar in the defined metric. Check the input."
        )
    if verb > 1:
        print(
            f"{n} unique entries detected in dissimilarity matrix of rank {r}. Set {n} as upper bound of number of clusters."
        )
    return n, r


def unique_nm(data, verb=0):
    data = np.round(data, decimals=4)
    a, idxs = np.unique(data, axis=0, return_index=True)
    umask = np.array([x in idxs for x in range(data.shape[0])], dtype=bool)
    uidx = np.where(umask == True)[0]
    n = uidx.size
    if verb > 3:
        print(
            f"{n} unique entries detected in dissimilarity matrix of selected conformers."
        )
    return n, umask
