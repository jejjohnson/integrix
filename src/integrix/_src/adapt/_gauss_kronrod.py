"""Adaptive Gauss–Kronrod quadrature on an interval, under jit, with a custom JVP.

Written on quadax's design, with no dependency on it (design decision 5):
quadax and `scipy.integrate.quad` are test references only. The search runs
in a `jax.lax.while_loop` over a static buffer of subintervals, so it traces
with static shapes; reverse mode cannot run through that loop, so the
derivative is the derivative of the final Kronrod rule on the final
partition, held fixed (a custom JVP).
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import Any

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float, Inexact, Int, ScalarLike

from integrix._src.adapt._gauss_kronrod_tables import GAUSS, KRONROD, NODES
from integrix._src.rule._pairwise import pairwise_sum
from integrix._src.rule._rule import Integral
from integrix.exceptions import ValidationError


__all__ = [
    "GaussKronrod",
]


class GaussKronrod(eqx.Module):
    """Adaptive Gauss–Kronrod quadrature to a tolerance, with an error estimate.

    I = ∫ₐᵇ f(x) dx. On each subinterval the Kronrod rule K₂ₙ₊₁ gives the value
    and its embedded Gauss rule Gₙ the error, E ≈ |K₂ₙ₊₁(f) − Gₙ(f)|, for no
    extra evaluations (the Gauss nodes are a subset of the Kronrod ones).
    Endpoint singularities need tanh–sinh, not Gauss–Kronrod.

    Algorithm:
        1. Evaluate the pair on [a, b].
        2. While the summed error of any output entry exceeds
           max(atol, rtol |I|) and the buffer of ``max_intervals``
           subintervals has room, bisect the subinterval with the largest
           error and evaluate the pair on both halves.
        3. Return the summed value and error, the number of evaluations, and
           ``converged`` (never raised: under jit it is a flag to check).

    The derivative, forward or reverse, is that of the final Kronrod rule on
    the final partition held fixed: in the integrand's parameters (through
    ``args`` or closed-over arrays) and in the limits a and b.

    Attributes:
        order: The number of Kronrod points 2n + 1: 15, 21, 31, 41, 51 or 61.
        rtol: The relative tolerance on the summed error, ≥ 0.
        atol: The absolute tolerance on the summed error, ≥ 0.
        max_intervals: The size of the static subinterval buffer, ≥ 1.

    Raises:
        ValidationError: If ``order`` has no table, a tolerance is negative or
            not finite, or ``max_intervals`` is not a positive integer.

    References:
        Piessens, R. et al. (1983). QUADPACK: A Subroutine Package for
        Automatic Integration. Springer.

    Examples:
        >>> import jax.numpy as jnp
        >>> result = GaussKronrod(rtol=1e-6)(jnp.exp, 0.0, 1.0)
        >>> bool(jnp.allclose(result.value, jnp.e - 1.0)), bool(result.converged)
        (True, True)

        A peaked integrand takes more subintervals:

        >>> peak = lambda x, width: 1.0 / (width**2 + (x - 0.3) ** 2)
        >>> result = GaussKronrod(rtol=1e-6)(peak, 0.0, 1.0, args=0.01)
        >>> bool(result.converged), int(result.n_evaluations) > 15
        (True, True)
    """

    order: int = eqx.field(static=True, default=15)
    rtol: float = eqx.field(static=True, default=1e-8)
    atol: float = eqx.field(static=True, default=0.0)
    max_intervals: int = eqx.field(static=True, default=64)

    def __check_init__(self) -> None:
        if self.order not in NODES:
            raise ValidationError(
                f"'order' must be one of {sorted(NODES)}, got {self.order!r}"
            )
        for name in ("rtol", "atol"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0.0):
                raise ValidationError(
                    f"'{name}' must be finite and non-negative, got {value!r}"
                )
        if (
            isinstance(self.max_intervals, bool)
            or not isinstance(self.max_intervals, int)
            or self.max_intervals < 1
        ):
            raise ValidationError(
                f"'max_intervals' must be a positive integer, got "
                f"{self.max_intervals!r}"
            )

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
                scalar x; it may return an array (each entry is integrated).
            lower: The lower limit a, finite.
            upper: The upper limit b, finite (b < a gives −∫ᵇₐ).
            args: Parameters passed to ``fn``, differentiable.

        Returns:
            An `integrix.rule.Integral` with the value, the summed error
            estimate, the number of evaluations and ``converged``.

        Raises:
            ValidationError: If a limit is not a scalar, or a host limit is
                not finite.
        """
        lower, upper = _limits(lower, upper)
        args = _as_arrays(args)
        integrand = _bind(fn, args)
        # closed-over arrays become leaves of the converted function, so they
        # are differentiated and never cached globally
        converted = eqx.filter_closure_convert(integrand, lower, args)
        # the arrays are differentiated; the traced program rides along static
        leaves, program = eqx.partition(converted, eqx.is_array)
        value, error, n_evaluations, converged = _solve(
            self, program, lower, upper, args, leaves
        )
        return Integral(
            value=value,
            error=error,
            n_evaluations=n_evaluations,
            converged=converged,
        )


def _limits(lower: ScalarLike, upper: ScalarLike) -> tuple[Array, Array]:
    """The limits as scalar float arrays of one dtype; host values must be finite."""
    for name, value in (("lower", lower), ("upper", upper)):
        if np.ndim(value) != 0:
            raise ValidationError(f"'{name}' must be a scalar, got {np.shape(value)}")
        if not isinstance(value, jax.Array) and not np.isfinite(np.asarray(value)):
            raise ValidationError(
                f"'{name}' must be finite (tanh–sinh handles infinite limits), "
                f"got {value!r}"
            )
    dtype = jnp.result_type(lower, upper, float)
    return jnp.asarray(lower, dtype=dtype), jnp.asarray(upper, dtype=dtype)


def _as_arrays(args: Any) -> Any:
    """``args`` with every float leaf as a JAX array (ints and bools kept as they are).

    The closure-converted integrand records which of its arguments are arrays;
    a Python float would be static there but arrive as an array under a JVP,
    so floats are made arrays up front.
    """

    def convert(leaf: Any) -> Any:
        if isinstance(leaf, (bool, int, np.integer, np.bool_)):
            return leaf
        if isinstance(leaf, (float, complex, np.floating, np.complexfloating)):
            return jnp.asarray(leaf)
        if isinstance(leaf, np.ndarray):
            return jnp.asarray(leaf)
        return leaf

    return jax.tree.map(convert, args)


def _bind(fn: Callable[..., ArrayLike], args: Any) -> Callable[[Array, Any], Array]:
    """``fn`` as f(x, args), whether or not it takes ``args``."""
    if args is None:
        return lambda x, _args: jnp.asarray(fn(x))
    return lambda x, args: jnp.asarray(fn(x, args))


def _rule(order: int, dtype: Any) -> tuple[Array, Array, Array]:
    """The Kronrod nodes on [−1, 1], their weights, and the Gauss weights."""
    nodes = jnp.asarray(NODES[order], dtype=dtype)
    kronrod = jnp.asarray(KRONROD[order], dtype=dtype)
    gauss = jnp.asarray(GAUSS[order], dtype=dtype)
    return nodes, kronrod, gauss


def _pair(
    config: GaussKronrod,
    fn: Callable[..., Array],
    lower: Array,
    upper: Array,
    start: Float[Array, " interval"],
    stop: Float[Array, " interval"],
    args: Any,
) -> tuple[Inexact[Array, " interval *out"], Float[Array, " interval *out"]]:
    """The Kronrod value and the |K − G| error on each subinterval [s₀, s₁]."""
    nodes, kronrod, gauss = _rule(config.order, lower.dtype)
    width = upper - lower
    # x = a + (b − a) (s₀ + (s₁ − s₀)(1 + ξ)/2)
    half = 0.5 * (stop - start)
    fractions = einx.add(
        "i, i k -> i k", start + half, einx.multiply("i, k -> i k", half, nodes)
    )
    points = lower + width * fractions
    samples = jax.vmap(jax.vmap(lambda x: fn(x, args)))(points)
    # the Jacobian of s ↦ x on each subinterval: (b − a)(s₁ − s₀)/2
    scale = width * half
    value = einx.dot("i k ..., k -> i ...", samples, kronrod)
    embedded = einx.dot("i k ..., k -> i ...", samples, gauss)
    value = einx.multiply("i ..., i -> i ...", value, scale.astype(value.dtype))
    embedded = einx.multiply("i ..., i -> i ...", embedded, scale.astype(value.dtype))
    # Eᵢ = |Kᵢ − Gᵢ|, entry by entry
    error = jnp.abs(value - embedded)
    return value, error


def _search(
    config: GaussKronrod,
    fn: Callable[..., Array],
    lower: Array,
    upper: Array,
    args: Any,
) -> tuple[Any, ...]:
    """Bisect the worst subinterval until the tolerance is met or the buffer fills."""
    size = config.max_intervals
    dtype = lower.dtype
    first_value, first_error = _pair(
        config, fn, lower, upper, jnp.zeros(1, dtype), jnp.ones(1, dtype), args
    )
    out_shape = first_value.shape[1:]
    starts = jnp.zeros(size, dtype).at[0].set(0.0)
    stops = jnp.zeros(size, dtype).at[0].set(1.0)
    values = jnp.zeros((size, *out_shape), first_value.dtype).at[0].set(first_value[0])
    errors = jnp.zeros((size, *out_shape), first_error.dtype).at[0].set(first_error[0])

    def totals(state):
        values, errors, count = state[2], state[3], state[4]
        active = jnp.arange(size) < count
        masked = einx.where("i, i ..., -> i ...", active, values, 0.0)
        total = pairwise_sum(masked)
        total_error = pairwise_sum(
            einx.where("i, i ..., -> i ...", active, errors, 0.0)
        )
        return total, total_error

    def within_tolerance(total, total_error):
        # every entry: E ≤ max(atol, rtol |I|)
        bound = jnp.maximum(config.atol, config.rtol * jnp.abs(total))
        return jnp.all(total_error <= bound)

    def keep_going(state):
        total, total_error = totals(state)
        return (state[4] < size) & ~within_tolerance(total, total_error)

    def bisect(state):
        starts, stops, values, errors, count = state
        active = jnp.arange(size) < count
        # the subinterval whose largest entry error is largest
        largest = einx.max("i ... -> i", errors)
        worst = jnp.argmax(jnp.where(active, largest, -jnp.inf))
        middle = 0.5 * (starts[worst] + stops[worst])
        halves_start = jnp.stack([starts[worst], middle])
        halves_stop = jnp.stack([middle, stops[worst]])
        half_values, half_errors = _pair(
            config, fn, lower, upper, halves_start, halves_stop, args
        )
        # the left half replaces the worst slot, the right half takes a new one
        starts = starts.at[worst].set(halves_start[0]).at[count].set(halves_start[1])
        stops = stops.at[worst].set(halves_stop[0]).at[count].set(halves_stop[1])
        values = values.at[worst].set(half_values[0]).at[count].set(half_values[1])
        errors = errors.at[worst].set(half_errors[0]).at[count].set(half_errors[1])
        return starts, stops, values, errors, count + 1

    state = (starts, stops, values, errors, jnp.asarray(1))
    state = jax.lax.while_loop(keep_going, bisect, state)
    total, total_error = totals(state)
    count = state[4]
    converged = within_tolerance(total, total_error)
    n_evaluations = config.order * (2 * count - 1)
    return total, total_error, n_evaluations, converged, state[0], state[1], count


def _final_rule(
    config: GaussKronrod,
    fn: Callable[..., Array],
    lower: Array,
    upper: Array,
    args: Any,
    starts: Array,
    stops: Array,
    count: Int[Array, ""],
) -> Inexact[Array, "*out"]:
    """The Kronrod sum on a fixed partition, differentiable in its inputs."""
    values, _ = _pair(config, fn, lower, upper, starts, stops, args)
    active = jnp.arange(starts.shape[0]) < count
    masked = einx.where("i, i ..., -> i ...", active, values, 0.0)
    return pairwise_sum(masked)


@functools.partial(jax.custom_jvp, nondiff_argnums=(0, 1))
def _solve(config, program, lower, upper, args, leaves):
    """The adaptive integral; its JVP is the final rule's on the final partition.

    The integrand is ``eqx.combine(leaves, program)``: its closed-over arrays
    (``leaves``) are differentiated like ``args``, and its traced program is
    static.
    """
    fn = eqx.combine(leaves, program)
    total, error, n_evaluations, converged, *_ = _search(
        config, fn, lower, upper, _as_arrays(args)
    )
    return total, error, n_evaluations, converged


@_solve.defjvp
def _solve_jvp(config, program, primals, tangents):
    lower, upper, args, leaves = primals
    # under a JVP the primal of a float argument is the caller's Python float
    args = _as_arrays(args)
    primals = (lower, upper, args, leaves)
    total, error, n_evaluations, converged, starts, stops, count = _search(
        config, eqx.combine(leaves, program), lower, upper, args
    )
    partition = jax.lax.stop_gradient((starts, stops, count))

    def final(lower, upper, args, leaves):
        fn = eqx.combine(leaves, program)
        return _final_rule(config, fn, lower, upper, args, *partition)

    _, total_dot = jax.jvp(final, primals, tangents)
    zero_count = np.zeros(np.shape(n_evaluations), dtype=jax.dtypes.float0)
    zero_flag = np.zeros(np.shape(converged), dtype=jax.dtypes.float0)
    outputs = (total, error, n_evaluations, converged)
    return outputs, (total_dot, jnp.zeros_like(error), zero_count, zero_flag)
