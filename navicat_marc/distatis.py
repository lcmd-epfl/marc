#!/usr/bin/env python

import numpy as np
import scipy.spatial as sps
from sklearn.decomposition import PCA
from navicat_marc.exceptions import UniqueError


def run_distatis(list_D, verb=0):
    n = len(list_D)
    print(f"Running distatis on the {n} selected matrices.")
    allshapes = [D.shape for D in list_D]
    assert all_equal(allshapes)
    C = calc_C(list_D[0])
    list_Sn = []
    for i, D in enumerate(list_D):
        S = calc_S(D, C)
        U, s, Vt = np.linalg.svd(S, full_matrices=True, hermitian=True)
        Sn = S / np.max(s)
        if verb > 4:
            print(f"\n The {i}th normalized cross-product distance matrix is :\n")
            print(np.array_str(Sn, precision=2, suppress_small=True))
        list_Sn.append(Sn)
    allshapes = [Sn.shape for Sn in list_Sn]
    assert all_equal(allshapes)
    cm = np.ones((n, n))
    for i in range(0, len(list_D) - 1):
        for j in range(i + 1, len(list_D)):
            t = list_Sn[i]
            p = list_Sn[j]
            tt = np.einsum("ij,ji->", t.T, t)
            tp = np.einsum("ij,ji->", t.T, p)
            pp = np.einsum("ij,ji->", p.T, p)
            cos_dist = tp / np.sqrt(tt * tp)
            cm[i, j] = cm[j, i] = cos_dist
    if verb > 4:
        print("\n The cosine distance matrix in distatis is :\n")
        print(np.array_str(cm, precision=2, suppress_small=True))
    U, s, Vt = np.linalg.svd(cm, full_matrices=True, hermitian=True)
    exp_i = np.round(np.max(s) / np.sum(s), 2)
    if verb > 2:
        print(f"The explained inertia given by the compromise method is {exp_i}")
    alphas = U[0, :] / np.sum(U[0, :])
    if verb > 4:
        print(f"The assigned weights are {alphas}")
    Sp = np.zeros_like(list_D[0])
    for alpha, Sn in zip(alphas, list_Sn):
        Sp -= alpha * Sn
    return Sp


def all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)


def calc_m(D):
    m = np.ones(shape=(D.shape[0], 1)) * (1 / D.shape[0])
    return m


def calc_C(D):
    m = calc_m(D)
    C = np.identity(n=D.shape[0]) - np.full_like(D, fill_value=m)
    return C


def calc_S(D, C):
    return -0.5 * C @ D @ C.T
