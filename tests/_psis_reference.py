"""plumax's NumPy Pareto smoothing, copied as the test oracle for integrix#70.

From ``src/plumax/population/_src/integrate.py`` at plumax ``feb5e86``
(``_generalized_pareto_fit`` and ``pareto_smoothed_log_weights``), unchanged
except that xtremax's ``gpd_icdf`` is replaced by its closed form,
F⁻¹(p) = −σ log(1 − p) (e^a − 1)/a with a = −k log(1 − p), so these tests
depend on neither plumax nor xtremax.
"""

from __future__ import annotations

import math

import numpy as np


def generalized_pareto_fit(exceedances: np.ndarray) -> tuple[float, float]:
    """Zhang & Stephens (2009) posterior-mean (k, σ), with the PSIS prior."""
    x = np.sort(exceedances)
    n = x.size
    prior_bs, prior_k = 3.0, 10.0
    m = 30 + int(math.sqrt(n))
    b = 1.0 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))
    b = b / (prior_bs * x[int(n / 4 + 0.5) - 1]) + 1.0 / x[-1]
    k = np.mean(np.log1p(-np.outer(b, x)), axis=1)
    profile = n * (np.log(-b / k) - k - 1.0)
    weights = 1.0 / np.sum(np.exp(np.subtract.outer(profile, profile)).T, axis=1)
    weights = weights / weights.sum()
    b_mean = float(np.sum(b * weights))
    shape = float(np.mean(np.log1p(-b_mean * x)))
    scale = -shape / b_mean
    shape = (n * shape + prior_k * 0.5) / (n + prior_k)
    return shape, scale


def gpd_icdf(levels: np.ndarray, scale: float, shape: float) -> np.ndarray:
    """F⁻¹(p) = σ((1 − p)^(−k) − 1)/k, → −σ log(1 − p) as k → 0."""
    log_exceedance = np.log1p(-levels)
    exponent = -shape * log_exceedance
    if abs(shape) < 1e-12:
        return -scale * log_exceedance
    return -scale * log_exceedance * np.expm1(exponent) / exponent


def pareto_smoothed_log_weights(log_weights: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalised Pareto-smoothed log weights and k̂, as plumax computes them."""
    log_w = np.asarray(log_weights, dtype=float).ravel()
    log_w = log_w - log_w.max()
    size = log_w.size
    tail_length = math.ceil(min(0.2 * size, 3.0 * math.sqrt(size)))
    if tail_length >= size:
        return log_w - np.logaddexp.reduce(log_w), math.nan
    order = np.argsort(log_w)
    cutoff = max(log_w[order[-tail_length - 1]], math.log(np.finfo(float).tiny))
    tail = order[log_w[order] > cutoff]
    k_hat = math.nan
    if tail.size > 4:
        exceedances = np.exp(log_w[tail]) - math.exp(cutoff)
        k_hat, scale = generalized_pareto_fit(exceedances)
        if math.isfinite(k_hat):
            levels = (np.arange(tail.size) + 0.5) / tail.size
            quantiles = gpd_icdf(levels, scale, k_hat)
            log_w[tail] = np.minimum(np.log(quantiles + math.exp(cutoff)), 0.0)
    smoothed = log_w - np.logaddexp.reduce(log_w)
    return smoothed, float(k_hat)
