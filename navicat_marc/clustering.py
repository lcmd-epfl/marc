#!/usr/bin/env python

import matplotlib
import numpy as np

matplotlib.use("Agg")

from itertools import cycle, product

import matplotlib.pyplot as plt
import scipy.cluster.hierarchy
import scipy.cluster.vq
from scipy.spatial.distance import euclidean, squareform
from sklearn.cluster import DBSCAN, AffinityPropagation, AgglomerativeClustering, KMeans
from sklearn.manifold import MDS, TSNE
from sklearn.neighbors import NearestCentroid

from navicat_marc.exceptions import UniqueError

ref_percentages = [0.01, 0.5]


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


def beautify_ax(ax):
    # Border
    ax.spines["top"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.spines["right"].set_color("black")
    ax.get_xaxis().set_tick_params(direction="out")
    ax.get_yaxis().set_tick_params(direction="out")
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    return ax


def plot_tsne(m: np.ndarray, points, clusters, names):

    # Generate tsne plot
    tsne = TSNE(
        n_components=2,
        metric="precomputed",
        init="random",
        early_exaggeration=20.0,
        perplexity=min(int(len(points) / np.sqrt(len(clusters))), 50),
    )
    tsne_results = tsne.fit_transform(m)

    # Plot tsne results
    fig, ax = plt.subplots(
        frameon=False,
        figsize=[4.2, 4.2],
        dpi=300,
    )
    ax = beautify_ax(ax)
    col = ["b", "g", "r", "c", "m", "k", "y"]
    mar = ["o", "v", "^", "s", "p", "h", "P", "D", "*", ">", "<"]
    cy_col_mar = cycle(product(col, mar))
    cmdict = dict(zip(points, cy_col_mar))
    cmb = np.array([cmdict[i] for i in points])
    for i, indices_list in enumerate(clusters):
        ax.scatter(
            tsne_results[indices_list, 0],
            tsne_results[indices_list, 1],
            s=50,
            edgecolors="black",
            zorder=1,
            alpha=0.5,
            c=cmb[i][0],
            marker=cmb[i][1],
            label=f"Cluster {i}",
        )
    for i, index in enumerate(points):
        ax.scatter(
            tsne_results[index, 0],
            tsne_results[index, 1],
            s=30,
            edgecolors="black",
            zorder=2,
            c=cmb[i][0],
            marker="X",
            label=f"{names[index]}",
        )
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
        shadow=True,
        ncol=len(clusters),
    )
    plt.savefig("tsne_plot.png", bbox_inches="tight")
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
        percentages = range(
            min([max(int(nm * percentage), 1) for percentage in ref_percentages]),
            max(
                [
                    min(max(int(nm * percentage), 3), nm - 1)
                    for percentage in ref_percentages
                ]
            ),
        )
        if len(percentages) == 1:
            n_clusters = 2
        else:
            id_gap = gap(x, nrefs=min(nm, 5), ks=percentages, verb=verb)
            n_clusters = max(percentages[id_gap], 2)
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

        if verb > 1:
            print(
                f"Closest index of point to cluster {iclust} center has index {cluster_pts_indices[min_idx]:02}"
            )
        closest_pt_idx.append(cluster_pts_indices[min_idx])
    # plot_tsne(m, closest_pt_idx, clusters)
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

        if verb > 1:
            print(
                f"Point in {iclust} center has index {ap.cluster_centers_indices_[iclust]:02}"
            )
        closest_pt_idx.append(ap.cluster_centers_indices_[iclust])
    # plot_tsne(m, closest_pt_idx, clusters)
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
        percentages = range(
            min([max(int(nm * percentage), 1) for percentage in ref_percentages]),
            max(
                [
                    min(max(int(nm * percentage), 3), nm - 1)
                    for percentage in ref_percentages
                ]
            ),
        )
        if len(percentages) == 1:
            n_clusters = 2
        else:
            id_gap = gap(x, nrefs=min(nm, 5), ks=percentages, verb=verb)
            n_clusters = max(percentages[id_gap], 2)
        n_unique, rank = unique_nr(m, verb=verb)
        n_clusters = min(n_unique, n_clusters)
    m = np.ones_like(m) - m
    ac = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage="single"
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

        if verb > 1:
            print(
                f"Closest index of point to cluster {iclust} center has index {cluster_pts_indices[min_idx]:02}"
            )
        closest_pt_idx.append(cluster_pts_indices[min_idx])
    # plot_tsne(m, closest_pt_idx, clusters)
    return closest_pt_idx, clusters


def gaps_diff(data, refs=None, nrefs=10, ks=range(1, 11), verb=0):
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
    s = np.zeros((len(ks),))
    diff = np.zeros((len(ks) - 1,))
    for (i, k) in enumerate(ks):
        km = KMeans(n_clusters=k, n_init=5)
        _ = km.fit_predict(data)
        kmc = km.cluster_centers_
        kml = km.labels_
        disp = sum([euclidean(data[m, :], kmc[kml[m], :]) for m in range(shape[0])])
        refdisps = np.zeros((rands.shape[2],))
        for j in range(rands.shape[2]):
            km = KMeans(n_clusters=k, n_init=5)
            _ = km.fit_predict(rands[:, :, j])
            kmc = km.cluster_centers_
            kml = km.labels_
            refdisps[j] = sum(
                [euclidean(rands[m, :, j], kmc[kml[m], :]) for m in range(shape[0])]
            )
        l = np.mean(scipy.log(refdisps))
        rld = np.log(disp)
        gaps[i] = l - rld
        sdk = np.sqrt(np.mean((np.log(refdisps) - rld) ** 2.0))
        s[i] = np.sqrt(1.0 + 1.0 / rands.shape[2]) * sdk
        if verb > 5:
            print(f"Gaps for k-values {ks[i]} : {gaps[i]}")
    for i in range(len(ks) - 1):
        diff[i] = gaps[i] - gaps[i + 1] + s[i + 1]
        if verb > 4:
            print(
                f"Gap(i) - Gap(i+1) - sk(i+1) for k-value {ks[i]} : {gaps[i]} - {gaps[i+1]} - {s[i+1]} =  {diff[i]}"
            )
    if verb > 3:
        print(f"Gap(i) - Gap(i+1) = sk(i+1)  for k-values {ks} : {diff}")
    return diff


def gap(data, refs=None, nrefs=5, ks=range(1, 11), verb=0):
    diff = gaps_diff(data, refs, nrefs, ks, verb)
    return np.argmax(diff)


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
