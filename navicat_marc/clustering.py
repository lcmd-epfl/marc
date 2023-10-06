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
from sklearn.metrics import silhouette_score

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
    val = np.min(m)
    if val < 0:
        m += -val
        np.clip(m, a_min=0, a_max=np.max(m))

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
    col = [
        "b",
        "g",
        "r",
        "c",
        "m",
        "k",
        "y",
        "plum",
        "gold",
        "yellowgreen",
        "teal",
        "violet",
        "salmon",
        "sienna",
        "silver",
        "tan",
        "wheat",
        "ivory",
        "darkgreen",
        "coral",
        "darkblue",
        "orange",
        "olive",
        "lightgreen",
        "lightblue",
        "aquamarine",
        "orchid",
    ]
    mar = ["o", "v", "^", "s", "p", "h", "P", "D", "*", ">", "<"]
    cy_col_mar = cycle(product(mar, col))
    cmdict = dict(zip(points, cy_col_mar))
    cmb = np.array([cmdict[i] for i in points])
    for i, indices_list in enumerate(clusters):
        ax.scatter(
            tsne_results[indices_list, 0],
            tsne_results[indices_list, 1],
            s=50,
            edgecolors="black",
            zorder=1,
            alpha=0.75,
            c=cmb[i][1],
            marker=cmb[i][0],
            label=f"Cluster {i}",
        )
    for i, index in enumerate(points):
        ax.scatter(
            tsne_results[index, 0],
            tsne_results[index, 1],
            s=30,
            edgecolors="black",
            zorder=2,
            c=cmb[i][1],
            marker="X",
            label=f"{names[index]}",
        )
    plt.savefig("tsne_plot.png", bbox_inches="tight")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
        shadow=True,
        ncol=min(len(clusters), 10),
    )
    plt.savefig("tsne_plot_legend.png", bbox_inches="tight")
    plt.close()


def kmeans_clustering(n_clusters, m: np.ndarray, rank=5, verb=0):
    mds = MDS(
        dissimilarity="precomputed",
        n_components=rank,
        n_init=1,
        max_iter=500,
        normalized_stress="auto",
        random_state=42,
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
            n_clusters = percentages[id_gap]
        n_unique, rank = unique_nr(m, verb=verb)
        n_clusters = int(min(n_unique, n_clusters))
    centroids = naive_sharding(x, n_clusters)
    km = KMeans(n_clusters=n_clusters, n_init=1, init=centroids)
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
            n_init=10,
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
            n_clusters = percentages[id_gap]
        n_unique, rank = unique_nr(m, verb=verb)
        n_clusters = min(n_unique, n_clusters)
    if n_clusters < 2:
        raise UniqueError(
            "Agglomerative clustering requires at least 2 clusters. Currently it seems like your data only has 1 cluster, either by mistake or by request."
        )
    m = np.ones_like(m) - m
    ac = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage="complete"
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


def finder(data, refs=None, nrefs=10, ks=range(1, 11), choice="silhouette", verb=0):
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
    sc = np.zeros((len(ks),))
    bick = np.zeros((len(ks),))
    diff = np.zeros((len(ks) - 1,))
    for i, k in enumerate(ks):
        centroids = naive_sharding(data, n_clusters=k)
        km = KMeans(n_clusters=k, n_init=1, init=centroids)
        _ = km.fit_predict(data)
        kmc = km.cluster_centers_
        kml = km.labels_

        # Calculate silhouette score while at it
        if choice == "silhouette":
            sc[i] = sc_score(data, kml, metric="euclidean")

        if choice == "bic":
            # Also test subdivision for X-means style clustering
            M = np.size(data, axis=1)
            p = M + 1
            obic = np.zeros(k)
            nbic = np.zeros(k)
            for l in range(k):
                rn = np.size(np.where(kml == l))
                var = max(
                    np.sum((data[kml == l] - kmc[l]) ** 2) / max(float(rn - 1), 1), 1e-6
                )
                obic[l] = loglikelihood(rn, rn, var, M, 1) - p / 2.0 * np.log(rn)

            for l in range(k):
                sk = 2  # in principle we try to split clusters in 2
                ci = data[kml == l]
                if ci.shape[0] != 1:
                    r = np.size(np.where(kml == l))
                    centroids = naive_sharding(ci, n_clusters=sk)
                    km = KMeans(n_clusters=sk, n_init=1, init=centroids)
                    _ = km.fit_predict(ci)
                    cic = km.cluster_centers_
                    cil = km.labels_
                    for n in range(sk):
                        rn = np.size(np.where(cil == n))
                        var = max(
                            np.sum((ci[cil == n] - cic[n]) ** 2)
                            / max(float(rn - sk), 1),
                            1e-6,
                        )
                        nbic[l] += loglikelihood(r, rn, var, M, sk)
                else:
                    sk = 1
                    r = np.size(np.where(kml == l))
                    centroids = naive_sharding(ci, n_clusters=sk)
                    km = KMeans(n_clusters=sk, n_init=1, init=centroids)
                    _ = km.fit_predict(ci)
                    cic = km.cluster_centers_
                    cil = km.labels_
                    for n in range(sk):
                        rn = np.size(np.where(cil == n))
                        var = max(
                            np.sum((ci[cil == n] - cic[n]) ** 2)
                            / max(float(rn - sk), 1),
                            1e-6,
                        )
                        nbic[l] += loglikelihood(r, rn, var, M, sk)
                p = sk * (M + 1)
                nbic[l] -= p / 2.0 * np.log(rn)

            bicdiff = obic - nbic  # If obic > nbic then k is good
            bick[i] = (bicdiff > 0).sum()
            # if any(obic < nbic) : # If not, a cluster can probably be split
            #    continue

        if choice == "gap":
            # Continue with gap statistic
            disp = sum([euclidean(data[m, :], kmc[kml[m], :]) for m in range(shape[0])])
            refdisps = np.zeros((rands.shape[2],))
            for j in range(rands.shape[2]):
                centroids = naive_sharding(rands[:, :, j], n_clusters=k)
                km = KMeans(n_clusters=k, n_init=1)
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
                print(
                    f"Gaps and silhouette scores for k-values {ks[i]} : {np.around(gaps[i],2)} {np.around(sc[i],2)}"
                )
    if choice == "gap":
        for i in range(len(ks) - 1):
            diff[i] = gaps[i] - gaps[i + 1] + s[i + 1]
            if verb > 4:
                print(
                    f"Gap(i) vs. Gap(i+1) - sk(i+1) for k-value {ks[i]} : {gaps[i]} - {gaps[i+1]} + {s[i+1]} =  {diff[i]}"
                )
        if verb > 2:
            print(
                f"Gap(i) - Gap(i+1) = sk(i+1)  for k-values {ks[:len(ks)-1]} : {np.round(diff,2)}"
            )
    if choice == "silhouette" and verb > 2:
        print(f"Silhouette score  for k-values {ks} : {np.round(sc,2)}")
    if choice == "bic" and verb > 2:
        print(f"Positive BIC-per-cluster count for k-values {ks} : {bick}")
    return diff, sc, bick


def gap(data, refs=None, nrefs=5, ks=range(1, 11), choice="silhouette", verb=0):
    diff, sc, bick = finder(data, refs, nrefs, ks, choice, verb)

    if choice == "gap":
        best = np.argmax(diff > 0) - 1
        if best == 0:
            print(
                "The number of clusters according to the gap statistic is 1! Your structures and energies all must be very similar!"
            )

    if choice == "silhouette":
        best = np.argmax(sc) - 1
        if best == 0:
            print(
                "The number of clusters according to the gap statistic is 1! Your structures and energies all must be very similar!"
            )

    if choice == "bic":
        best = np.argmin(bick == 0)

    return best


def sc_score(m, labels, metric="precomputed"):
    if all(labels == 0):
        return 0
    else:
        return silhouette_score(m, labels, metric=metric)


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


def naive_sharding(ds, n_clusters):
    """
    Create cluster centroids using deterministic naive sharding algorithm.

    Parameters
    ----------
    ds : numpy array
        The dataset to be used for centroid initialization.
    n_clusters : int
        The desired number of clusters for which centroids are required.
    Returns
    -------
    centroids : numpy array
        Collection of k centroids as a numpy array.
    """
    k = n_clusters
    n = np.shape(ds)[1]
    m = np.shape(ds)[0]
    if (k == 1 or n == 1) and m > 1:
        return ds[np.random.randint(0, m - 1)].reshape(1, -1)
    if m < 1:
        return ds[0].reshape(1, -1)

    def _get_mean(sums, step):
        """Vectorizable ufunc for getting means of summed shard columns."""
        return sums / step

    centroids = np.zeros((k, n))

    composite = np.mat(np.sum(ds, axis=1))
    ds = np.append(composite.T, ds, axis=1)
    ds.sort(axis=0)

    step = max(int(np.floor(m / k)), 1)
    # print(f"Step is set to {m} / {k} = {step} ")
    vfunc = np.vectorize(_get_mean)

    for j in range(k):
        if j == k - 1:
            centroids[j:] = vfunc(np.sum(ds[j * step :, 1:], axis=0), step)
        else:
            centroids[j:] = vfunc(
                np.sum(ds[j * step : (j + 1) * step, 1:], axis=0), step
            )

    return centroids


def loglikelihood(r, rn, var, m, k):
    l1 = -rn / 2.0 * np.log(2 * np.pi)
    l2 = -rn * m / 2.0 * np.log(var)
    l3 = -(rn - k) / 2.0
    l4 = rn * np.log(rn)
    l5 = -rn * np.log(r)
    return l1 + l2 + l3 + l4 + l5
