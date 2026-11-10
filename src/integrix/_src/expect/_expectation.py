"""Expectations 𝔼[f] with a standard error: from draws, points or a proposal.

integrix averages; sampling from a posterior stays in NumPyro and pyrox
(design decision, #26), so `monte_carlo` takes draws already made as readily
as a sampler.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from integrix._src.expect._sequences import (
    SOBOL_BITS,
    halton_points,
    sobol_integers,
)
from integrix._src.rule._pairwise import pairwise_sum
from integrix.exceptions import ValidationError


__all__ = [
    "Expectation",
    "importance",
    "monte_carlo",
    "qmc",
]


class Expectation(eqx.Module):
    """An estimate of 𝔼[f] with its standard error.

    Attributes:
        value: The estimate Î, one per output entry.
        standard_error: Its standard error σ̂, the same shape.
        n_samples: The number of integrand evaluations S.
        effective_sample_size: (∑ wₛ)² / ∑ wₛ² for importance sampling;
            ``None`` otherwise.

    Examples:
        >>> import jax.numpy as jnp
        >>> estimate = monte_carlo(lambda x: x, jnp.array([1.0, 2.0, 3.0]))
        >>> bool(estimate.value == 2.0), estimate.n_samples
        (True, 3)
    """

    value: Float[Array, "*out"]
    standard_error: Float[Array, "*out"]
    n_samples: int = eqx.field(static=True)
    effective_sample_size: Float[Array, ""] | None = None


def monte_carlo(
    fn: Callable[[PyTree], Array],
    samples_or_sampler: PyTree | Callable[[PRNGKeyArray, int], PyTree],
    n: int | None = None,
    *,
    key: PRNGKeyArray | None = None,
) -> Expectation:
    """Plain Monte Carlo: Î = (1/S) ∑ₛ f(xₛ), with standard error σ̂/√S.

    The draws xₛ ~ p are given (an array or pytree with a leading sample axis,
    NumPyro's posterior samples, say) or drawn as ``sampler(key, n)``.

    Algorithm:
        1. The draws: as given, or ``sampler(key, n)``.
        2. fₛ = f(xₛ) under `jax.vmap`.
        3. Î = (1/S) ∑ₛ fₛ and s² = ∑ₛ (fₛ − Î)² / (S − 1), both summed
           pairwise; σ̂ = s / √S.

    Args:
        fn: The integrand f, applied to one draw.
        samples_or_sampler: The draws, leading axis S ≥ 2, or a sampler
            ``(key, n) -> draws``.
        n: The number of draws for a sampler (must match the draws if given).
        key: The PRNG key for a sampler.

    Returns:
        The `Expectation`.

    Raises:
        ValidationError: If a sampler has no ``key`` or ``n``, fewer than two
            draws are given, or ``n`` disagrees with the draws.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> sampler = lambda key, n: jax.random.normal(key, (n,))
        >>> estimate = monte_carlo(jnp.square, sampler, 4096, key=jax.random.key(0))
        >>> bool(abs(estimate.value - 1.0) < 4 * estimate.standard_error)
        True
    """
    if callable(samples_or_sampler):
        if key is None or n is None:
            raise ValidationError(
                "monte_carlo needs 'key' and 'n' to draw from a sampler"
            )
        draws = samples_or_sampler(key, n)
    else:
        draws = samples_or_sampler
    count = _sample_count(draws)
    if n is not None and n != count:
        raise ValidationError(f"'n' is {n} but the draws have {count} samples")
    values = _as_float(jax.vmap(lambda draw: jnp.asarray(fn(draw)))(draws))
    value, standard_error = _mean_and_error(values)
    return Expectation(value=value, standard_error=standard_error, n_samples=count)


def qmc(
    fn: Callable[[Array], Array],
    dim: int,
    n: int,
    *,
    sequence: Literal["sobol", "halton"] = "sobol",
    n_shifts: int = 16,
    transform: Callable[[Array], PyTree] | None = None,
    key: PRNGKeyArray,
) -> Expectation:
    """Randomised quasi-Monte Carlo over [0, 1)^D, with an error from random shifts.

    Î = (1/R) ∑ᵣ Îᵣ, where Îᵣ = (1/n) ∑ᵢ f(T(uᵢ ⊕ Δᵣ)) averages over the
    low-discrepancy points uᵢ shifted by the r-th random shift Δᵣ, and
    σ̂ = sᵣ / √R from the spread of the R shifted estimates. For a smooth
    integrand the error falls faster than S^{−1/2}. An unshifted sequence has
    no error estimate, so R ≥ 2.

    Algorithm:
        1. The first n points of the Sobol (Joe–Kuo, Gray-code order, n a
           power of two is best) or Halton sequence in D dimensions.
        2. R random shifts from one key: a digital shift (XOR of 32-bit
           integers) for Sobol, a rotation (uᵢ + Δᵣ) mod 1 for Halton.
        3. Map each point through ``transform`` (to a Gaussian with
           `jax.scipy.special.ndtri`, say), evaluate f, average per shift,
           then over shifts.

    Args:
        fn: The integrand f, applied to one (transformed) point.
        dim: The dimension D ≥ 1 (Sobol: at most 64).
        n: The number of points per shift, n ≥ 1.
        sequence: ``"sobol"`` or ``"halton"``.
        n_shifts: The number of random shifts R ≥ 2.
        transform: Maps a point u ∈ [0, 1)^D to the integration space T(u);
            the identity when ``None``.
        key: The PRNG key for the shifts.

    Returns:
        The `Expectation`, with S = n R evaluations.

    Raises:
        ValidationError: If ``dim``, ``n`` or ``n_shifts`` is out of range, or
            ``sequence`` is unknown.

    References:
        Owen, A. B. (2013). Monte Carlo theory, methods and examples, ch. 17
        (randomised quasi-Monte Carlo).

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> cosine = lambda u: jnp.prod(jnp.cos(u))
        >>> estimate = qmc(cosine, 3, 256, key=jax.random.key(0))
        >>> bool(
        ...     abs(estimate.value - jnp.sin(1.0) ** 3)
        ...     < 4 * estimate.standard_error
        ... )
        True
    """
    for name, value, smallest in (
        ("dim", dim, 1),
        ("n", n, 1),
        ("n_shifts", n_shifts, 2),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < smallest:
            raise ValidationError(
                f"'{name}' must be an integer of at least {smallest}, got {value!r}"
            )
    dtype = jnp.result_type(float)
    if sequence == "sobol":
        integers = jnp.asarray(sobol_integers(n, dim))
        # a digital shift per replicate: uᵢ ⊕ Δᵣ, bit by bit
        shifts = jax.random.bits(key, (n_shifts, dim), dtype=jnp.uint32)
        tiled_points = einx.id("i d -> r i d", integers, r=n_shifts)
        tiled_shifts = einx.id("r d -> r i d", shifts, i=n)
        shifted = jnp.bitwise_xor(tiled_points, tiled_shifts)
        # keep p leading bits (23 in float32, 32 in float64) so the midpoint
        # of the top 2^{−p} cell, 1 − 2^{−p−1}, is a float below 1: no point is
        # 0 or 1
        precision = min(SOBOL_BITS, jnp.finfo(dtype).nmant)
        leading = shifted >> jnp.uint32(SOBOL_BITS - precision)
        points = (leading.astype(dtype) + 0.5) / 2.0**precision
    elif sequence == "halton":
        base = jnp.asarray(halton_points(n, dim), dtype=dtype)
        shifts = jax.random.uniform(key, (n_shifts, dim), dtype=dtype)
        # a Cranley–Patterson rotation: (uᵢ + Δᵣ) mod 1
        rotated = jnp.mod(einx.add("i d, r d -> r i d", base, shifts), 1.0)
        # (u + Δ) can round onto 1, which mod sends to 0: keep inside (0, 1)
        half_eps = 0.5 * jnp.finfo(dtype).eps
        points = jnp.clip(rotated, half_eps, 1.0 - half_eps)
    else:
        raise ValidationError(
            f"'sequence' must be 'sobol' or 'halton', got {sequence!r}"
        )

    def evaluate(point: Array) -> Array:
        mapped = point if transform is None else transform(point)
        return jnp.asarray(fn(mapped))

    values = _as_float(jax.vmap(jax.vmap(evaluate))(points))
    # Îᵣ = (1/n) ∑ᵢ fᵣᵢ, one estimate per shift
    per_shift = jax.vmap(lambda shift_values: pairwise_sum(shift_values) / n)(values)
    value, standard_error = _mean_and_error(per_shift)
    return Expectation(
        value=value, standard_error=standard_error, n_samples=n * n_shifts
    )


def importance(
    fn: Callable[[PyTree], Array],
    proposal_sample: Callable[[PRNGKeyArray, int], PyTree],
    proposal_log_prob: Callable[[PyTree], Array],
    target_log_prob: Callable[[PyTree], Array],
    n: int,
    *,
    key: PRNGKeyArray,
) -> Expectation:
    """Self-normalised importance sampling: Î = ∑ₛ wₛ f(xₛ) / ∑ₛ wₛ, xₛ ~ q.

    The weights wₛ = p(xₛ)/q(xₛ) need p only up to a constant. The standard
    error is the delta-method one, σ̂² = ∑ₛ w̃ₛ² (fₛ − Î)², with w̃ the
    normalised weights, and the effective sample size (∑ wₛ)²/∑ wₛ² is
    reported: heavy-tailed weights make the estimate unreliable long before
    σ̂ looks large.

    Args:
        fn: The integrand f, applied to one draw.
        proposal_sample: ``(key, n) -> draws`` from the proposal q.
        proposal_log_prob: log q(x) for one draw.
        target_log_prob: log p(x) for one draw, up to an additive constant.
        n: The number of draws S ≥ 2.
        key: The PRNG key for the proposal.

    Returns:
        The `Expectation`, with ``effective_sample_size``.

    Raises:
        ValidationError: If ``n`` is below 2.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import jax.scipy.stats as stats
        >>> estimate = importance(
        ...     jnp.square,
        ...     lambda key, n: 2.0 * jax.random.normal(key, (n,)),
        ...     lambda x: stats.norm.logpdf(x, scale=2.0),
        ...     lambda x: stats.norm.logpdf(x),
        ...     4096,
        ...     key=jax.random.key(0),
        ... )
        >>> bool(abs(estimate.value - 1.0) < 4 * estimate.standard_error)
        True
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValidationError(f"'n' must be an integer of at least 2, got {n!r}")
    draws = proposal_sample(key, n)
    log_target = jax.vmap(target_log_prob)(draws)
    log_proposal = jax.vmap(proposal_log_prob)(draws)
    log_weights = log_target - log_proposal
    # w̃ₛ = wₛ / ∑ wₛ, from log wₛ = log p(xₛ) − log q(xₛ), stably
    normalised = jax.nn.softmax(log_weights)
    values = _as_float(jax.vmap(lambda draw: jnp.asarray(fn(draw)))(draws))
    normalised = normalised.astype(jnp.result_type(values.dtype, normalised.dtype))
    # Î = ∑ₛ w̃ₛ fₛ
    value = pairwise_sum(einx.multiply("s, s ... -> s ...", normalised, values))
    deviation = values - value
    # σ̂² = ∑ₛ w̃ₛ² (fₛ − Î)²
    variance = pairwise_sum(
        einx.multiply("s, s ... -> s ...", normalised**2, deviation**2)
    )
    # ESS = 1 / ∑ₛ w̃ₛ²
    effective = 1.0 / pairwise_sum(normalised**2)
    # a target with no support on any draw leaves no weight at all
    value = eqx.error_if(
        value,
        ~jnp.any(jnp.isfinite(log_weights)),
        "importance: the target has zero density at every proposal draw",
    )
    return Expectation(
        value=value,
        standard_error=jnp.sqrt(variance),
        n_samples=n,
        effective_sample_size=effective,
    )


def _sample_count(draws: PyTree) -> int:
    """The shared leading sample count S ≥ 2 of the draws."""
    leaves = jax.tree.leaves(draws)
    if not leaves:
        raise ValidationError("the draws must hold at least one array")
    if any(jnp.ndim(leaf) == 0 for leaf in leaves):
        raise ValidationError("every leaf of the draws needs a leading sample axis")
    counts = {int(jnp.shape(leaf)[0]) for leaf in leaves}
    if len(counts) != 1:
        raise ValidationError(
            f"every leaf of the draws must share one sample count, got {sorted(counts)}"
        )
    (count,) = counts
    if count < 2:
        raise ValidationError(f"a standard error needs at least two draws, got {count}")
    return count


def _as_float(values: Array) -> Array:
    """Integer or boolean integrand values in the default float, so sums count."""
    if jnp.issubdtype(values.dtype, jnp.inexact):
        return values
    return values.astype(jnp.result_type(float))


def _mean_and_error(values: Array) -> tuple[Array, Array]:
    """The mean over the leading axis and its standard error s/√S, summed pairwise."""
    count = values.shape[0]
    # Î = (1/S) ∑ₛ fₛ
    mean = pairwise_sum(values) / count
    # s² = ∑ₛ (fₛ − Î)² / (S − 1)
    variance = pairwise_sum((values - mean) ** 2) / (count - 1)
    return mean, jnp.sqrt(variance / count)
