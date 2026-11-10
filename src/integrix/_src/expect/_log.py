"""Log-space expectations: Gauss–Hermite in log Q and Simpson in log q."""

from __future__ import annotations

import math
import operator
from collections.abc import Callable
from typing import Any

import einx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float

from integrix._src.rule._log import log_integrate
from integrix.exceptions import ValidationError


__all__ = ["log_gauss_hermite", "log_simpson"]


def log_gauss_hermite(
    log_fn: Callable[[Array], Array],
    mean: ArrayLike,
    std: ArrayLike,
    *,
    n: int = 32,
) -> Float[Array, "*out"]:
    """log 𝔼[f(Q)] for a lognormal Q, by Gauss–Hermite quadrature in log Q.

    log Q is normal with mean μ and variance σ². With qₖ = exp(μ + √2 σ xₖ)
    at the Hermite nodes xₖ of weight e^(−x²), log 𝔼[f(Q)] ≈ logsumexp over k
    of (log wₖ − ½ log π + log f(qₖ)), so f never leaves log space. The rule
    is exact when f is a polynomial in log q of degree at most 2n − 1, and
    gives log 𝔼[Qᵃ] = aμ + a²σ²/2 to rounding for moderate aσ.

    Algorithm:
        1. The Hermite nodes xₖ and weights wₖ on the host (static n).
        2. The rates qₖ = exp(μ + √2 σ xₖ), node axis first, batched over the
           broadcast shape of μ and σ.
        3. `integrix.rule.log_integrate` with weights wₖ/√π.

    Args:
        log_fn: log f, applied to the rates (node axis first) and returning
            the node axis first.
        mean: μ, the mean of log Q; broadcasts against ``std``.
        std: σ > 0, the standard deviation of log Q.
        n: The number of Hermite nodes, at least 1.

    Returns:
        log 𝔼[f(Q)], with the broadcast shape of μ and σ followed by any
        output axes of f.

    Raises:
        ValidationError: If ``n`` is not an integer of at least 1.

    Examples:
        >>> import jax.numpy as jnp
        >>> # log 𝔼[Q] = μ + σ²/2
        >>> value = log_gauss_hermite(jnp.log, 0.2, 0.5)
        >>> bool(jnp.allclose(value, 0.2 + 0.5 * 0.5**2))
        True

        A batch of (μ, σ) at once:

        >>> means = jnp.array([0.0, 1.0])
        >>> log_gauss_hermite(jnp.log, means, 0.3).shape
        (2,)

    References:
        Golub, G. H. & Welsch, J. H. (1969). Calculation of Gauss quadrature
        rules. Math. Comp. 23, 221–230.
    """
    count = _count(n, smallest=1)
    reference_nodes, reference_weights = np.polynomial.hermite.hermgauss(count)
    # μ and σ broadcast to one batch shape
    mean, std = jnp.broadcast_arrays(jnp.asarray(mean), jnp.asarray(std))
    dtype = jnp.result_type(mean, std, float)
    # xₖ and wₖ/√π, in the inputs' precision
    nodes = jnp.asarray(reference_nodes, dtype=dtype)
    weights = jnp.asarray(reference_weights / math.sqrt(math.pi), dtype=dtype)
    # √2 σ
    spread = math.sqrt(2.0) * std
    # √2 σ xₖ, node axis first
    offsets = einx.multiply("n, ... -> n ...", nodes, spread)
    # uₖ = μ + √2 σ xₖ
    log_rates = einx.add("n ..., ... -> n ...", offsets, mean)
    # qₖ = e^uₖ
    rates = jnp.exp(log_rates)
    # log 𝔼[f(Q)] = logsumexp over k of (log(wₖ/√π) + log f(qₖ))
    return log_integrate(weights, log_fn(rates), axis=0)


def log_simpson(
    log_fn: Callable[[Array], Array],
    lower: float,
    upper: float,
    *,
    n: int = 2001,
) -> Float[Array, "*out"]:
    """log ∫ₐᵇ f(q) dq by composite Simpson on a uniform grid in u = log q.

    ∫ₐᵇ f(q) dq = ∫ from log a to log b of f(eᵘ) eᵘ du, and with the Simpson
    weights wₖ on a uniform u grid, log ∫ ≈ logsumexp over k of
    (log wₖ + log f(e^uₖ) + uₖ). A uniform grid in log q resolves a lognormal
    body and a power-law tail alike, which a uniform grid in q cannot.

    Algorithm:
        1. A uniform grid uₖ on [log a, log b] with an odd number of nodes,
           and Simpson's weights (h/3)(1, 4, 2, …, 4, 1) on the host.
        2. `integrix.rule.log_integrate` with those weights on
           log f(e^uₖ) + uₖ.

    Args:
        log_fn: log f, applied to the rates e^uₖ (node axis first) and
            returning the node axis first.
        lower: a > 0, the lower limit.
        upper: b > a, the upper limit.
        n: The number of nodes, at least 3; an even count is raised by one,
            since Simpson's rule needs an even number of intervals.

    Returns:
        log ∫ₐᵇ f(q) dq, with any output axes of f.

    Raises:
        ValidationError: If ``n`` is not an integer of at least 3, or the
            limits are not 0 < a < b.

    Examples:
        >>> import jax.numpy as jnp
        >>> # ∫₁^∞ 3 q⁻⁴ dq = 1, truncated at 10⁶
        >>> value = log_simpson(lambda q: jnp.log(3.0) - 4.0 * jnp.log(q), 1.0, 1e6)
        >>> bool(jnp.abs(value) < 1e-6)
        True
        >>> # ∫₁ᵉ dq/q = 1
        >>> bool(jnp.allclose(log_simpson(lambda q: -jnp.log(q), 1.0, jnp.e), 0.0))
        True
    """
    count = _count(n, smallest=3)
    # Simpson needs an even number of intervals
    count = count if count % 2 == 1 else count + 1
    lower, upper = float(lower), float(upper)
    if not (0.0 < lower < upper and math.isfinite(upper)):
        raise ValidationError(
            f"'lower' and 'upper' must satisfy 0 < lower < upper < ∞, got "
            f"({lower}, {upper})"
        )
    # uₖ, uniform in log q
    grid = np.linspace(math.log(lower), math.log(upper), count)
    step = grid[1] - grid[0]
    # (h/3)(1, 4, 2, …, 4, 1)
    pattern = np.ones(count)
    pattern[1:-1:2] = 4.0
    pattern[2:-1:2] = 2.0
    weights = jnp.asarray(pattern * step / 3.0, dtype=jnp.result_type(float))
    log_grid = jnp.asarray(grid, dtype=weights.dtype)
    # log f(e^uₖ) + uₖ: the Jacobian dq = eᵘ du
    log_values = einx.add("n ..., n -> n ...", log_fn(jnp.exp(log_grid)), log_grid)
    return log_integrate(weights, log_values, axis=0)


def _count(n: Any, *, smallest: int) -> int:
    """``n`` as a Python int of at least ``smallest``, or `ValidationError`."""
    message = f"'n' must be an integer of at least {smallest}, got {n!r}"
    if isinstance(n, bool):
        raise ValidationError(message)
    try:
        count = operator.index(n)
    except TypeError as err:
        raise ValidationError(message) from err
    if count < smallest:
        raise ValidationError(message)
    return count
