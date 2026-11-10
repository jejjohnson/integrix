"""Tanh–sinh quadrature: endpoint singularities and infinite limits.

The substitution x = tanh(π/2 sinh t) (and exp–sinh or sinh–sinh for
infinite limits) makes the integrand decay double-exponentially in t, so the
trapezoid rule in t converges fast even when f is singular at an endpoint.
Every level of step halving is computed (static shapes), and the first whose
change meets the tolerance is returned, so the result differentiates
directly, with no custom rule.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer
from jaxtyping import Array, ArrayLike, ScalarLike

from integrix._src.adapt._levels import first_converged
from integrix._src.rule._pairwise import pairwise_sum
from integrix._src.rule._rule import Integral
from integrix.exceptions import ValidationError


__all__ = [
    "TanhSinh",
]


# The t range of each map, in whole units so every level's grid aligns. At
# t = ±4 a finite endpoint is e^{−π sinh 4} ≈ 5e−38 (relative) away and an
# infinite map reaches e^{π/2 sinh 4} ≈ 4e18. Exp–sinh runs to t = −5 on the
# side of its finite end, e^{−(π/2) sinh 5} ≈ 4e−51, so an integrable
# singularity there (x^{−1/2} at 0) leaves no truncation the levels miss.
_T_RANGE = {"finite": (-4, 4), "upper": (-5, 4), "lower": (-5, 4), "both": (-4, 4)}


class TanhSinh(eqx.Module):
    """Tanh–sinh (double-exponential) quadrature, for singular ends and infinite limits.

    I = ∫ₐᵇ f(x) dx with x = c + d tanh(u), u = (π/2) sinh t, c = (a + b)/2,
    d = (b − a)/2, so dx = d (π/2) cosh t sech²(u) dt decays like e^{−(π/2)e^{|t|}}
    and the trapezoid rule in t converges double-exponentially, even for
    f(x) = log x or x^{−1/2} at an endpoint. For [a, ∞) the map is exp–sinh,
    x = a + e^{u}; for (−∞, ∞) it is sinh–sinh, x = sinh(u).

    Algorithm:
        1. Evaluate f once on the finest grid t = k h, h = 2^{−L}, |t| ≤ 4.
           Near a finite endpoint the distance (b − a)/(1 + e^{2|u|}) is
           formed directly, so x never rounds onto the endpoint; a node that
           still does gets weight zero.
        2. For each level ℓ = 0, …, L, the trapezoid sum Tℓ on every
           2^{L−ℓ}-th node; the error of level ℓ is |Tℓ − Tℓ₋₁|.
        3. Return the first level within max(atol, rtol |I|), or the last,
           with ``converged``.

    Attributes:
        rtol: The relative tolerance, ≥ 0.
        atol: The absolute tolerance, ≥ 0.
        max_levels: The number of step halvings L ≥ 1; the finest grid has
            8 · 2^L + 1 nodes (9 · 2^L + 1 for a half-infinite interval).
        min_levels: The coarsest level ℓ ≥ 1 that may be accepted, against a
            coarse grid whose nodes all land on the integrand's zeros.

    The integrand sees x, not the distance to the end: near a singular
    endpoint at a ≠ 0, x − a is quantised to the spacing of floats at a, so
    f(x) = (x − a)^{−1/2} loses accuracy there. Shift the variable so the
    singularity sits at 0.

    Raises:
        ValidationError: If a tolerance is negative or not finite, or
            ``max_levels`` or ``min_levels`` is out of range.

    References:
        Takahasi, H. and Mori, M. (1974). Double exponential formulas for
        numerical integration. Publ. RIMS Kyoto Univ. 9, 721–741.

    Examples:
        >>> import jax.numpy as jnp
        >>> result = TanhSinh(rtol=1e-6)(jnp.log, 0.0, 1.0)
        >>> bool(jnp.allclose(result.value, -1.0)), bool(result.converged)
        (True, True)
        >>> result = TanhSinh(rtol=1e-6)(lambda x: jnp.exp(-x), 0.0, float("inf"))
        >>> bool(jnp.allclose(result.value, 1.0))
        True
    """

    rtol: float = eqx.field(static=True, default=1e-10)
    atol: float = eqx.field(static=True, default=0.0)
    max_levels: int = eqx.field(static=True, default=6)
    min_levels: int = eqx.field(static=True, default=3)

    def __check_init__(self) -> None:
        check_tolerances(self.rtol, self.atol)
        check_levels(self.max_levels, self.min_levels)

    def __call__(
        self,
        fn: Callable[..., ArrayLike],
        lower: ScalarLike,
        upper: ScalarLike,
        *,
        args: Any = None,
    ) -> Integral:
        """Integrate ``fn`` from ``lower`` to ``upper``.

        Args:
            fn: The integrand f(x), or f(x, args) when ``args`` is given, for a
                scalar x; it may return an array.
            lower: The lower limit a; ``-inf`` (a Python or NumPy float) for
                an infinite one.
            upper: The upper limit b; ``inf`` for an infinite one.
            args: Parameters passed to ``fn``, differentiable.

        Returns:
            An `integrix.rule.Integral` with the value, the change from the
            previous level as the error, the number of evaluations and
            ``converged``.

        Raises:
            ValidationError: If a limit is not a scalar or is NaN.
        """
        sign, kind, start, stop = _orient(lower, upper)
        call = (lambda x: fn(x)) if args is None else (lambda x: fn(x, args))
        dtype = jnp.result_type(start, stop, float)
        if kind == "empty":
            out = jax.eval_shape(lambda x: jnp.asarray(call(x)), jnp.zeros((), dtype))
            zeros = jnp.zeros(out.shape, out.dtype)
            return Integral(
                value=zeros,
                error=zeros,
                n_evaluations=jnp.asarray(0),
                converged=jnp.asarray(True),
            )
        t_low, t_high = _T_RANGE[kind]
        per_unit = 2**self.max_levels
        # t = k h on the finest grid, k = t_low/h, …, t_high/h
        steps = jnp.arange(t_low * per_unit, t_high * per_unit + 1, dtype=dtype)
        n_nodes = (t_high - t_low) * per_unit + 1
        t = steps * jnp.asarray(2.0**-self.max_levels, dtype=dtype)
        points, jacobian, valid = _substitute(kind, start, stop, t, dtype)
        safe_points = jnp.where(valid, points, _anchor(kind, start, stop, dtype))
        samples = jax.vmap(lambda x: jnp.asarray(call(x)))(safe_points)
        weights = jnp.where(valid, jacobian, 0.0).astype(samples.dtype)
        # wₖ f(xₖ), zero where the node is invalid (no inf · 0)
        weighted = einx.multiply("k, k ... -> k ...", weights, samples)
        terms = einx.where("k, k ..., -> k ...", valid, weighted, 0.0)
        estimates = []
        for level in range(self.max_levels + 1):
            stride = 2 ** (self.max_levels - level)
            step = 2.0**-level
            # Tℓ = hℓ ∑ₖ w(k hℓ) f(x(k hℓ))
            estimates.append(step * pairwise_sum(terms[::stride]))
        estimates = jnp.stack(estimates)
        errors = jnp.concatenate(
            [
                jnp.full_like(estimates[:1], jnp.inf),
                jnp.abs(estimates[1:] - estimates[:-1]),
            ]
        )
        value, error, _, converged = first_converged(
            estimates,
            errors,
            rtol=self.rtol,
            atol=self.atol,
            min_level=self.min_levels,
        )
        value = sign * value
        if kind == "finite":
            # a traced limit cannot be checked on the host: inf or NaN at run
            # time raises here instead of returning NaN
            finite = jnp.isfinite(jnp.asarray(start)) & jnp.isfinite(jnp.asarray(stop))
            value = eqx.error_if(
                value,
                ~finite,
                "TanhSinh: a JAX-array limit must be finite; give infinite "
                "limits as Python or NumPy floats",
            )
        return Integral(
            value=value,
            error=error,
            n_evaluations=jnp.asarray(n_nodes),
            converged=converged,
        )


def check_tolerances(rtol: float, atol: float) -> None:
    """Raise `ValidationError` unless both tolerances are finite and ≥ 0."""
    for name, value in (("rtol", rtol), ("atol", atol)):
        if not (math.isfinite(value) and value >= 0.0):
            raise ValidationError(
                f"'{name}' must be finite and non-negative, got {value!r}"
            )


def check_levels(max_levels: int, min_levels: int) -> None:
    """Raise `ValidationError` unless 1 ≤ ``min_levels`` ≤ ``max_levels``, integers."""
    for name, value in (("max_levels", max_levels), ("min_levels", min_levels)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValidationError(f"'{name}' must be a positive integer, got {value!r}")
    if min_levels > max_levels:
        raise ValidationError(
            f"'min_levels' ({min_levels}) must not exceed 'max_levels' ({max_levels})"
        )


def _orient(lower: Any, upper: Any) -> tuple[float, str, Any, Any]:
    """The sign, the map ("finite", "upper", "lower" or "both"), and the limits."""
    for name, value in (("lower", lower), ("upper", upper)):
        if np.ndim(value) != 0:
            raise ValidationError(f"'{name}' must be a scalar, got {np.shape(value)}")
        if not isinstance(value, Tracer) and np.isnan(np.asarray(value)):
            raise ValidationError(f"'{name}' must not be NaN")
    # a concrete limit (Python, NumPy or an eager JAX array) is read on the host
    lower_inf = not isinstance(lower, Tracer) and np.isinf(np.asarray(lower))
    upper_inf = not isinstance(upper, Tracer) and np.isinf(np.asarray(upper))
    if not lower_inf and not upper_inf:
        return 1.0, "finite", lower, upper
    if lower_inf and upper_inf and float(lower) == float(upper):
        # ∫ from ∞ to ∞ (or −∞ to −∞) is an empty interval
        return 1.0, "empty", 0.0, 0.0
    sign = 1.0
    # orient so that any infinite limit is +∞ above or −∞ below
    if (lower_inf and float(lower) > 0) or (upper_inf and float(upper) < 0):
        lower, upper, sign = upper, lower, -1.0
        lower_inf, upper_inf = upper_inf, lower_inf
    if lower_inf and upper_inf:
        return sign, "both", 0.0, 0.0
    if upper_inf:
        return sign, "upper", lower, 0.0
    return sign, "lower", 0.0, upper


def _substitute(kind: str, start: Any, stop: Any, t: Array, dtype: Any):
    """x(t), dx/dt and the validity of each node, for the chosen map."""
    half_pi = jnp.asarray(math.pi / 2, dtype=dtype)
    # u = (π/2) sinh t
    u = half_pi * jnp.sinh(t)
    outer = half_pi * jnp.cosh(t)
    if kind == "finite":
        a = jnp.asarray(start, dtype=dtype)
        b = jnp.asarray(stop, dtype=dtype)
        width = b - a
        decay = jnp.exp(-2.0 * jnp.abs(u))
        # the distance to the nearer end, (b − a)/(1 + e^{2|u|}), formed directly
        distance = width * decay / (1.0 + decay)
        points = jnp.where(t <= 0, a + distance, b - distance)
        # dx/dt = (b − a)/2 · (π/2) cosh t · sech² u
        #       = 2(b − a)(π/2) cosh t e^{−2|u|}/(1 + e^{−2|u|})²
        jacobian = 2.0 * width * outer * decay / (1.0 + decay) ** 2
        valid = (points != a) & (points != b) & (jacobian != 0)
        return points, jacobian, valid
    if kind == "both":
        # x = sinh u, dx/dt = cosh u (π/2) cosh t
        points = jnp.sinh(u)
        jacobian = jnp.cosh(u) * outer
    else:
        # x = a + e^{u} (or b − e^{u}), dx/dt = e^{u} (π/2) cosh t
        growth = jnp.exp(u)
        jacobian = growth * outer
        if kind == "upper":
            end = jnp.asarray(start, dtype=dtype)
            points = end + growth
        else:
            end = jnp.asarray(stop, dtype=dtype)
            points = end - growth
        # a node that rounds onto the finite end is dropped, as for a finite map
        valid = jnp.isfinite(points) & jnp.isfinite(jacobian) & (jacobian != 0)
        return points, jacobian, valid & (points != end)
    valid = jnp.isfinite(points) & jnp.isfinite(jacobian) & (jacobian != 0)
    return points, jacobian, valid


def _anchor(kind: str, start: Any, stop: Any, dtype: Any) -> Array:
    """A safe point inside the interval, for nodes that are not evaluated."""
    if kind == "finite":
        return 0.5 * (jnp.asarray(start, dtype=dtype) + jnp.asarray(stop, dtype=dtype))
    if kind == "upper":
        return jnp.asarray(start, dtype=dtype) + 1.0
    if kind == "lower":
        return jnp.asarray(stop, dtype=dtype) - 1.0
    return jnp.zeros((), dtype=dtype)
