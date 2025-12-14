# coding:utf-8
"""utils.py
Include distance calculation for evaluation metrics
"""
import glob
import math
import os
import sys

import numpy as np
import sklearn
from scipy import integrate, stats


def _clean_1d_finite(arr):
    arr = np.asarray(arr, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _is_almost_constant(arr, rtol=1e-7, atol=1e-8):
    if arr.size == 0:
        return True
    spread = np.ptp(arr)
    return spread <= max(atol, np.max(np.abs(arr)) * rtol)


def _add_jitter(arr, scale=1e-6):
    if arr.size == 0:
        return arr
    magnitude = max(np.max(np.abs(arr)), 1.0)
    rng = np.random.default_rng(0)
    return arr + rng.normal(0.0, magnitude * scale, size=arr.shape)


def _kde_with_fallback(arr):
    if arr.size == 0:
        return None, arr

    attempts = [arr]
    if _is_almost_constant(arr):
        attempts.append(_add_jitter(arr))

    for candidate in attempts:
        try:
            return stats.gaussian_kde(candidate), candidate
        except Exception:
            continue
    return None, attempts[-1]


# Calculate overlap between the two PDF
def overlap_area(A, B):
    A = _clean_1d_finite(A)
    B = _clean_1d_finite(B)

    if _is_almost_constant(A) and _is_almost_constant(B):
        if A.size and B.size and np.isclose(np.mean(A), np.mean(B), rtol=1e-7, atol=1e-8):
            return 1.0
        return 0.0

    pdf_A, data_A = _kde_with_fallback(A)
    pdf_B, data_B = _kde_with_fallback(B)

    if pdf_A is None or pdf_B is None:
        return 0.0

    lower = np.min((np.min(data_A), np.min(data_B)))
    upper = np.max((np.max(data_A), np.max(data_B)))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
        return 0.0

    try:
        return integrate.quad(lambda x: float(min(pdf_A(x)[0], pdf_B(x)[0])), lower, upper)[0]
    except Exception:
        return 0.0


# Calculate KL distance between the two PDF
def kl_dist(A, B, num_sample=1000):
    A = _clean_1d_finite(A)
    B = _clean_1d_finite(B)

    if _is_almost_constant(A) and _is_almost_constant(B):
        if A.size and B.size and np.isclose(np.mean(A), np.mean(B), rtol=1e-7, atol=1e-8):
            return 0.0
        return 0.0

    pdf_A, data_A = _kde_with_fallback(A)
    pdf_B, data_B = _kde_with_fallback(B)

    if pdf_A is None or pdf_B is None:
        return 0.0

    min_A, max_A = np.min(data_A), np.max(data_A)
    min_B, max_B = np.min(data_B), np.max(data_B)

    try:
        sample_A = np.linspace(min_A, max_A, num_sample)
        sample_B = np.linspace(min_B, max_B, num_sample)
        return float(stats.entropy(pdf_A(sample_A), pdf_B(sample_B)))
    except Exception:
        return 0.0


def c_dist(A, B, mode='None', normalize=0):
    c_dist = np.zeros(len(B))
    for i in range(0, len(B)):
        if mode == 'None':
            c_dist[i] = np.linalg.norm(A - B[i])
        elif mode == 'EMD':
            if normalize == 1:
                A_ = sklearn.preprocessing.normalize(A.reshape(1, -1), norm='l1')[0]
                B_ = sklearn.preprocessing.normalize(B[i].reshape(1, -1), norm='l1')[0]
            else:
                A_ = A.reshape(1, -1)[0]
                B_ = B[i].reshape(1, -1)[0]

            c_dist[i] = stats.wasserstein_distance(A_, B_)

        elif mode == 'KL':
            if normalize == 1:
                A_ = sklearn.preprocessing.normalize(A.reshape(1, -1), norm='l1')[0]
                B_ = sklearn.preprocessing.normalize(B[i].reshape(1, -1), norm='l1')[0]
            else:
                A_ = A.reshape(1, -1)[0]
                B_ = B[i].reshape(1, -1)[0]

            B_[B_ == 0] = 0.00000001
            c_dist[i] = stats.entropy(A_, B_)
    return c_dist
