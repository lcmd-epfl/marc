#!/usr/bin/env python

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import scipy.cluster.hierarchy
from scipy.spatial.distance import euclidean, squareform
from sklearn.cluster import DBSCAN, KMeans


def plot_dendrogram(m: np.ndarray, label: str):
    if verb > 0:
        print(f"Max pairwise {label}: {np.max(m)} in {label} units.")
    assert np.all(m - m.T < 1e-6)
    reduced_distances = squareform(m)
    linkage = scipy.cluster.hierarchy.linkage(reduced_distances, method="average")
    plt.title(f"{label} average linkage hierarchical clustering")
    dn = scipy.cluster.hierarchy.dendrogram(
        linkage, no_labels=True, count_sort="descendent"
    )
    if verb > 0:
        print(f"Saving dendrogram plot as {label}_dendrogram.png in working directory.")
    plt.savefig(f"{label}_dendrogram.png")


def kmeans_clustering(n_clusters: int, m: np.ndarray, verb=0):
    km = KMeans(n_clusters=n_clusters, n_init=50)
    cm = km.fit_predict(m)
    u, c = np.unique(cm, return_counts=True)
    closest_pt_idx = []
    if verb > 0:
        print(
            f"Unique clusters found: {u} \nWith counts: {c} \nAdding up to {np.sum(c)}"
        )
    for iclust in range(km.n_clusters):

        # get all points assigned to each cluster:
        cluster_pts = m[km.labels_ == iclust]

        # get all indices of points assigned to this cluster:
        cluster_pts_indices = np.where(km.labels_ == iclust)[0]

        cluster_cen = km.cluster_centers_[iclust]
        min_idx = np.argmin(
            [euclidean(m[idx], cluster_cen) for idx in cluster_pts_indices]
        )

        # Testing:
        if verb > 1:
            print(
                f"Closest index of point to cluster {iclust} center has index {cluster_pts_indices[min_idx]}"
            )
        closest_pt_idx.append(cluster_pts_indices[min_idx])
    return closest_pt_idx


def dbscan_clustering(n_neighbors: int, m: np.ndarray, verb=0):
    dbsc = DBSCAN(eps=0.5, min_samples=n_neighbors)
    cm = dbsc.fit_predict(m)
    u, c = np.unique(cm, return_counts=True)
    closest_pt_idx = []
    if verb > 0:
        print(
            f"Unique clusters found: {u} \nWith counts: {c} \nAdding up to {np.sum(c)}"
        )
    for iclust, idx in enumerate(dbsc.core_sample_indices_):

        if verb > 1:
            print(f"Core point of cluster {iclust} center has index {idx}")
        closest_pt_idx.append(idx)
    return closest_pt_idx
