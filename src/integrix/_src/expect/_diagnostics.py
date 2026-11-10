"""Importance-weight diagnostics: the effective sample size and Pareto smoothing."""

from __future__ import annotations

import math

import einx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float

from integrix.exceptions import ValidationError


__all__ = ["effective_sample_size", "pareto_smoothed_log_weights"]

# Zhang & Stephens' prior: b is scaled by the first quartile over 3, and the
# shape is shrunk towards ½ with the weight of 10 pseudo-observations
_PRIOR_QUARTILE_SCALE = 3.0
_PRIOR_SHAPE_WEIGHT = 10.0
_PRIOR_SHAPE = 0.5
# fewer exceedances than this leave no tail to fit
_MIN_EXCEEDANCES = 5


def effective_sample_size(
    log_weights: Float[ArrayLike, "*batch draw"],
) -> Float[Array, "*batch"]:
    """ESS = (∑ₛ wₛ)² / ∑ₛ wₛ² from log weights, along the last axis.

    log ESS = 2 logsumexp(log wₛ) − logsumexp(2 log wₛ), so no weight is
    exponentiated and the result is the same for weights scaled by any
    constant. It equals S for equal weights and falls towards 1 as one weight
    dominates.

    Args:
        log_weights: Unnormalised log importance weights log wₛ, the S draws
            along the last axis, any batch in front.

    Returns:
        The effective sample size, one per batch entry.

    Raises:
        ValidationError: If ``log_weights`` has no draw axis.

    Examples:
        >>> import jax.numpy as jnp
        >>> bool(jnp.allclose(effective_sample_size(jnp.zeros(8)), 8.0))
        True
        >>> # one weight e times the others: (e + 3)² / (e² + 3)
        >>> ess = effective_sample_size(jnp.array([1.0, 0.0, 0.0, 0.0]))
        >>> bool(jnp.allclose(ess, (jnp.e + 3) ** 2 / (jnp.e**2 + 3)))
        True
    """
    log_weights = _as_log_weights(log_weights)
    # 2 log ∑ₛ wₛ
    log_sum = einx.logsumexp("... s -> ...", log_weights)
    # log ∑ₛ wₛ²
    log_sum_of_squares = einx.logsumexp("... s -> ...", 2.0 * log_weights)
    # log ESS = 2 log ∑ wₛ − log ∑ wₛ²
    return jnp.exp(2.0 * log_sum - log_sum_of_squares)


def pareto_smoothed_log_weights(
    log_weights: Float[ArrayLike, "*batch draw"],
) -> tuple[Float[Array, "*batch draw"], Float[Array, "*batch"]]:
    """Pareto-smoothed log importance weights and the tail shape k̂.

    The largest M = ⌈min(S/5, 3√S)⌉ weights are replaced by the expected order
    statistics of a generalised Pareto distribution fitted to their
    exceedances over the cutoff (Zhang and Stephens' estimator, with the
    weakly informative prior on the shape), capped at the largest raw weight,
    and the weights are normalised to logsumexp 0. The quantile is
    F⁻¹(p) = σ((1 − p)^(−k) − 1)/k. k̂ > 0.7 means the importance estimate
    should not be trusted.

    Algorithm:
        1. Shift log w so its maximum is 0; with M ≥ S there is no tail, so
           normalise and return k̂ = NaN.
        2. Sort, take the top M as the tail and the next weight as the cutoff;
           weights tied at the cutoff are masked out of the fit.
        3. Fit (k, σ) to the exceedances: the Zhang–Stephens posterior mean
           over m = 30 + ⌊√M⌋ candidate values, vectorised, then shrink k
           towards ½.
        4. Replace the tail by min(log(F⁻¹((z − ½)/M) + cutoff), 0), z = 1, …, M,
           when k̂ is finite, and normalise.

    The fit sorts and selects, so it is piecewise in the weights: the tail
    replacement and k̂ are wrapped in ``jax.lax.stop_gradient``, and the
    gradient flows only through the untouched weights and the normalisation.
    Batched over any leading axes, and traceable (S is static).

    Args:
        log_weights: Unnormalised log importance weights, the S draws along
            the last axis.

    Returns:
        ``(smoothed, k̂)``: the normalised smoothed log weights (logsumexp 0
        along the last axis) and the fitted shape, NaN when fewer than five
        weights exceed the cutoff (equal weights need no smoothing).

    Raises:
        ValidationError: If ``log_weights`` has no draw axis.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> log_weights = 0.3 * jax.random.normal(jax.random.key(0), (1000,))
        >>> smoothed, k_hat = pareto_smoothed_log_weights(log_weights)
        >>> bool(jnp.allclose(jax.nn.logsumexp(smoothed), 0.0, atol=1e-5))
        True
        >>> bool(k_hat < 0.7)
        True
        >>> # equal weights: nothing to smooth, k̂ is NaN
        >>> _, k_equal = pareto_smoothed_log_weights(jnp.zeros(100))
        >>> bool(jnp.isnan(k_equal))
        True

    References:
        Vehtari, A., Simpson, D., Gelman, A., Yao, Y. & Gabry, J. Pareto
        smoothed importance sampling. arXiv:1507.02646.
        Zhang, J. & Stephens, M. A. (2009). A new and efficient estimation
        method for the generalized Pareto distribution. Technometrics 51,
        316–325.
    """
    log_weights = _as_log_weights(log_weights)
    smooth = jnp.vectorize(_smooth_one, signature="(s)->(s),()")
    return smooth(log_weights)


def _as_log_weights(log_weights: ArrayLike) -> Float[Array, "*batch draw"]:
    """Log weights as a floating array with a draw axis."""
    array = jnp.asarray(log_weights)
    if array.ndim == 0:
        raise ValidationError("'log_weights' must have a draw axis, got a scalar")
    if not jnp.issubdtype(array.dtype, jnp.floating):
        array = array.astype(jnp.result_type(float))
    return array


def _smooth_one(
    log_weights: Float[Array, " draw"],
) -> tuple[Float[Array, " draw"], Float[Array, ""]]:
    """Pareto smoothing of one vector of log weights (see the public function)."""
    dtype = log_weights.dtype
    draw_count = log_weights.shape[0]
    # log w − max log w, so the largest weight is 1
    shifted = log_weights - jnp.max(log_weights)
    tail_length = math.ceil(min(0.2 * draw_count, 3.0 * math.sqrt(draw_count)))
    not_a_number = jnp.asarray(jnp.nan, dtype)
    if tail_length >= draw_count:
        # too few weights to separate a tail: normalise, no diagnostic
        normalised = shifted - jax.nn.logsumexp(shifted)
        return normalised, not_a_number
    fixed = jax.lax.stop_gradient(shifted)
    order = jnp.argsort(fixed)
    tail_index = order[draw_count - tail_length :]
    # the weight just below the tail, kept above the smallest normal float
    tiny = jnp.log(jnp.asarray(jnp.finfo(dtype).tiny, dtype))
    cutoff = jnp.maximum(fixed[order[draw_count - tail_length - 1]], tiny)
    tail_log_weights = fixed[tail_index]
    # weights tied at the cutoff are not exceedances
    exceeds = tail_log_weights > cutoff
    exceedance_count = jnp.sum(exceeds)
    # xᵢ = wᵢ − e^cutoff, ascending, 0 where masked
    exceedances = jnp.where(exceeds, jnp.exp(tail_log_weights) - jnp.exp(cutoff), 0.0)
    shape, scale = _generalized_pareto_fit(exceedances, exceeds, tail_length)
    enough = exceedance_count >= _MIN_EXCEEDANCES
    k_hat = jnp.where(enough, shape, not_a_number)
    # pₖ = (z − ½)/M, z = 1, …, M
    levels = (jnp.arange(tail_length, dtype=dtype) + 0.5) / tail_length
    quantiles = _generalized_pareto_quantile(levels, scale, shape)
    # log(F⁻¹(pₖ) + e^cutoff), capped at the largest raw weight (log 1 = 0)
    replaced = jnp.minimum(jnp.log(quantiles + jnp.exp(cutoff)), 0.0)
    smooth_tail = enough & jnp.isfinite(shape)
    # the replacement holds the smallest quantile against the smallest
    # exceedance, in the sorted order of the tail
    new_tail = jnp.where(smooth_tail & exceeds, replaced, tail_log_weights)
    new_tail = jax.lax.stop_gradient(new_tail)
    smoothed = shifted.at[tail_index].set(new_tail)
    normalised = smoothed - jax.nn.logsumexp(smoothed)
    return normalised, jax.lax.stop_gradient(k_hat)


def _generalized_pareto_fit(
    exceedances: Float[Array, " tail"],
    valid: Bool[Array, " tail"],
    tail_length: int,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Zhang–Stephens posterior-mean (k, σ) of a GPD, with the PSIS shape prior."""
    dtype = exceedances.dtype
    whole_count = jnp.sum(valid)
    count = whole_count.astype(dtype)
    # m from the static tail length M; it equals the exceedance count n
    # unless weights tie at the cutoff
    candidate_count = 30 + int(math.sqrt(tail_length))
    # the first quartile x₍⌊n/4 + ½⌋₎ of the n valid exceedances, which sit
    # at the end of the ascending tail, and the largest exceedance
    first_valid = tail_length - whole_count
    quartile = exceedances[first_valid + jnp.floor(count / 4 + 0.5).astype(int) - 1]
    largest = exceedances[-1]
    index = jnp.arange(1, candidate_count + 1, dtype=dtype)
    # bⱼ = 1 − √(m / (j − ½)), scaled by the prior
    candidates = 1.0 - jnp.sqrt(candidate_count / (index - 0.5))
    candidates = candidates / (_PRIOR_QUARTILE_SCALE * quartile) + 1.0 / largest
    # log(1 − bⱼ xᵢ), 0 for masked exceedances
    products = einx.multiply("b, x -> b x", candidates, exceedances)
    logs = jnp.log1p(-products)
    logs = jnp.where(einx.id("x -> b x", valid, b=candidate_count), logs, 0.0)
    # kⱼ = mean over i of log(1 − bⱼ xᵢ)
    shapes = einx.sum("b x -> b", logs) / count
    # ℓⱼ = n (log(−bⱼ/kⱼ) − kⱼ − 1)
    profile = count * (jnp.log(-candidates / shapes) - shapes - 1.0)
    # wⱼ ∝ e^ℓⱼ
    profile_weights = jax.nn.softmax(profile)
    # b̄ = ∑ⱼ wⱼ bⱼ
    mean_candidate = einx.dot("b, b ->", candidates, profile_weights)
    # k = mean of log(1 − b̄ xᵢ), σ = −k/b̄
    mean_logs = jnp.where(valid, jnp.log1p(-mean_candidate * exceedances), 0.0)
    shape = jnp.sum(mean_logs) / count
    scale = -shape / mean_candidate
    # shrink k towards ½ with the prior's weight
    prior_mass = _PRIOR_SHAPE_WEIGHT * _PRIOR_SHAPE
    shrunk = (count * shape + prior_mass) / (count + _PRIOR_SHAPE_WEIGHT)
    return shrunk, scale


def _generalized_pareto_quantile(
    levels: Float[Array, " tail"], scale: Float[Array, ""], shape: Float[Array, ""]
) -> Float[Array, " tail"]:
    """F⁻¹(p) = σ((1 − p)^(−k) − 1)/k, → −σ log(1 − p) as k → 0."""
    # log(1 − p)
    log_exceedance = jnp.log1p(-levels)
    # a = −k log(1 − p)
    exponent = -shape * log_exceedance
    small = jnp.abs(exponent) < 1e-8
    safe = jnp.where(small, 1.0, exponent)
    # (eᵃ − 1)/a, 1 at a = 0
    ratio = jnp.where(small, 1.0 + 0.5 * exponent, jnp.expm1(safe) / safe)
    # F⁻¹(p) = −σ log(1 − p) (eᵃ − 1)/a
    return -scale * log_exceedance * ratio
