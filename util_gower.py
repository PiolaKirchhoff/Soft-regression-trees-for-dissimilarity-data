# -*- coding: utf-8 -*-
"""Compute Gower-like dissimilarity matrices for the tabular DSRT experiments.

Used to construct training and test distance matrices from input features.

Created on Fri Jan 30 12:15:57 2026

"""

import numpy as np

def gower_distance_fixed_blockwise(XA, XB, cat_features, min_, range_, weights=None,block_size=256, dtype=np.float32):
    """
    Compute Gower-like dissimilarities between two sets of observations.

    Distances are computed in two-dimensional blocks to limit intermediate
    memory use. For numerical features, absolute differences are divided by
    the training-set range. Categorical features contribute 0 when the two
    values are equal and 1 when they are different. Missing values are not
    explicitly handled.

    Parameters
    ----------
    XA : array-like of shape (n_a, n_features)
        First set of observations.
    XB : array-like of shape (n_b, n_features)
        Second set of observations.
    cat_features : array-like of bool, shape (n_features,)
        True for categorical features and False for numerical features.
    min_ : array-like of shape (n_features,)
        Feature minima computed on the training set. Accepted as an input
        parameter but not used in the current calculation.
    range_ : array-like of shape (n_features,)
        Feature ranges computed on the training set. A zero range is replaced
        by 1 when calculating distances for numerical features.
    weights : array-like of shape (n_features,), optional
        Feature weights. If None, all features receive equal weight.
    block_size : int, default=256
        Number of observations processed per block along each matrix axis.
    dtype : data-type, default=np.float32
        Output matrix data type. Intermediate distance calculations use
        np.float32.

    Returns
    -------
    D : numpy.ndarray of shape (n_a, n_b)
        Pairwise dissimilarity matrix, where D[i, j] is the dissimilarity
        between XA[i] and XB[j].
    """

    XA = np.asarray(XA, dtype=object)
    XB = np.asarray(XB, dtype=object)

    nA, p = XA.shape
    nB = XB.shape[0]

    cat = np.asarray(cat_features, dtype=bool)
    num = ~cat

    if weights is None:
        weights = np.ones(p, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    wsum = float(weights.sum())

    # Indices of numerical and categorical features
    num_idx = np.where(num)[0]
    cat_idx = np.where(cat)[0]

    # Extract and convert numerical and categorical features
    XA_num = XA[:, num_idx].astype(np.float32) if num_idx.size else None
    XB_num = XB[:, num_idx].astype(np.float32) if num_idx.size else None
    XA_cat = XA[:, cat_idx] if cat_idx.size else None
    XB_cat = XB[:, cat_idx] if cat_idx.size else None

    rng_num = np.asarray(range_, dtype=np.float32)[num_idx] if num_idx.size else None
    rng_num = np.where(rng_num == 0, 1.0, rng_num) if num_idx.size else None

    w_num = weights[num_idx] if num_idx.size else None
    w_cat = weights[cat_idx] if cat_idx.size else None

    # Allocate the full output dissimilarity matrix
    D = np.empty((nA, nB), dtype=dtype)

    for i0 in range(0, nA, block_size):
        i1 = min(nA, i0 + block_size)
        for j0 in range(0, nB, block_size):
            j1 = min(nB, j0 + block_size)

            block = np.zeros((i1 - i0, j1 - j0), dtype=np.float32)

            # Numerical features: accumulate contributions without 3D arrays
            if num_idx.size:
                A = XA_num[i0:i1, :]  # (bi, fnum)
                B = XB_num[j0:j1, :]  # (bj, fnum)
                for k in range(A.shape[1]):
                    # (bi, bj) computed using 2D broadcasting
                    diff = np.abs(A[:, [k]] - B[:, [k]].T) / rng_num[k]
                    block += diff * w_num[k]

            # Categorical features: 0 for equal values, 1 otherwise
            if cat_idx.size:
                A = XA_cat[i0:i1, :]  # (bi, fcat)
                B = XB_cat[j0:j1, :]  # (bj, fcat)
                for k in range(A.shape[1]):
                    diff = (A[:, [k]] != B[:, [k]].T).astype(np.float32)
                    block += diff * w_cat[k]

            block /= wsum
            D[i0:i1, j0:j1] = block.astype(dtype, copy=False)

    return D