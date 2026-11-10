"""Integrate in log space: log ∑ₙ wₙ fₙ from log fₙ, without leaving log space."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import einx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from integrix._src._axes import (
    check_node_shape,
    flatten,
    nodes_first,
    normalise_axes,
    working_dtype,
)
from integrix._src.rule._rule import Rule, _kept_samples, _read_rule
from integrix._typing import Axes
from integrix.exceptions import ValidationError


__all__ = ["log_integrate"]


def log_integrate(
    rule: Rule,
    log_values_or_fn: ArrayLike | Callable[..., Any],
    *,
    axis: Axes | None = None,
    args: Any = None,
    where: ArrayLike | None = None,
) -> Float[Array, "*batch"]:
    """log I from log f: the log-sum-exp twin of `integrate`.

    log I = log ∑ₙ wₙ fₙ = logsumexp over n of (log wₙ + log fₙ), so an
    integrand whose values underflow (log fₙ ≈ −800, a heavy-tailed density
    far from its mode) still gives a finite log-integral. The weights must be
    positive; a mask leaves nodes out by making their term −∞.

    Algorithm:
        1. Read the weights w as `integrate` does, and check that they are
           positive.
        2. Get log f: the log-values as given, or ``log_f(nodes)`` (with
           ``args``, ``log_f(nodes, args)``), node axes first.
        3. Form tₙ = log wₙ + log fₙ, with −∞ for nodes ``where`` drops (their
           log fₙ replaced by 0 first, so neither a NaN nor its gradient
           leaks).
        4. Return logsumexp over the node axes.

    Args:
        rule: A rule with positive ``weights``, a tuple of 1-D rules, or a
            positive weights array w, as in `integrate`.
        log_values_or_fn: The log-samples log fₙ, or a callable log f
            evaluated at the rule's ``nodes`` and returning its node axes
            first.
        axis: The axes to integrate over, as in `integrate`.
        args: Extra arguments passed to the callable as ``log_f(nodes, args)``.
        where: A boolean mask that broadcasts against the log-samples; nodes
            where it is false are left out. With no node kept the result is
            −∞, the log of an empty sum.

    Returns:
        log I, with the shape of the log-samples without the integrated axes.

    Raises:
        ValidationError: As `integrate` (an unknown rule, a bad axis, the
            wrong number of samples or a bad mask), if the log-samples are
            complex, or if a weight is not positive (checked under ``jit``
            too).

    Examples:
        >>> import jax.numpy as jnp
        >>> weights = jnp.array([0.25, 0.75])
        >>> log_total = log_integrate(weights, jnp.log(jnp.array([2.0, 4.0])))
        >>> bool(jnp.allclose(log_total, jnp.log(3.5)))
        True

        Values far below the smallest float still have a log-integral:

        >>> tiny = jnp.full(2, -800.0)
        >>> bool(jnp.allclose(log_integrate(jnp.array([0.5, 0.5]), tiny), -800.0))
        True

        A mask drops a node:

        >>> keep = jnp.array([True, False])
        >>> masked = log_integrate(
        ...     weights, jnp.log(jnp.array([2.0, 4.0])), where=keep
        ... )
        >>> bool(jnp.allclose(masked, jnp.log(0.5)))
        True
    """
    weights, nodes = _read_rule(rule)
    if callable(log_values_or_fn):
        if nodes is None:
            raise ValidationError(
                "'rule' must have 'nodes' to evaluate a callable integrand"
            )
        if axis is not None:
            raise ValidationError(
                "'axis' must be None for a callable integrand, whose node axes "
                "come first"
            )
        log_fn = log_values_or_fn
        evaluated = log_fn(nodes) if args is None else log_fn(nodes, args)
        log_values = jnp.asarray(evaluated)
        axes = tuple(range(weights.ndim))
    else:
        log_values = jnp.asarray(log_values_or_fn)
        axes = normalise_axes(axis, log_values.ndim, weights.ndim)
    check_node_shape(log_values, axes, weights.shape)
    if jnp.issubdtype(log_values.dtype, jnp.complexfloating):
        raise ValidationError(
            f"'log_values_or_fn' must be real, got dtype {log_values.dtype}"
        )
    keep = _kept_samples(log_values, where, "propagate")

    dtype = working_dtype(log_values.dtype, weights.dtype)
    log_samples = nodes_first(log_values.astype(dtype), axes)
    flat_weights = flatten(weights).astype(dtype)
    flat_weights = eqx.error_if(
        flat_weights,
        jnp.any(flat_weights <= 0),
        "'rule' must have positive weights for log_integrate",
    )
    # log wₙ
    log_weights = jnp.log(flat_weights)
    if keep is not None:
        kept = nodes_first(keep, axes) > 0
        zero = jnp.zeros((), dtype)
        # 0 in place of every dropped log fₙ, so its NaN cannot leak
        log_samples = jnp.where(kept, log_samples, zero)
    # tₙ = log wₙ + log fₙ
    terms = einx.add("n ..., n -> n ...", log_samples, log_weights)
    if keep is not None:
        minus_infinity = jnp.asarray(-jnp.inf, dtype)
        # a dropped node adds e^(−∞) = 0 to the sum
        terms = jnp.where(kept, terms, minus_infinity)
    # log I = logsumexp over n of tₙ
    return einx.logsumexp("n ... -> ...", terms)
