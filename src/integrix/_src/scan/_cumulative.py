"""The running integral of samples along one axis, forward or reversed."""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Inexact

from integrix._src._axes import (
    as_inexact,
    check_node_shape,
    move_back,
    move_first,
    normalise_axes,
    working_dtype,
)
from integrix._typing import RuleLike
from integrix.exceptions import ValidationError


__all__ = [
    "cumulative",
]


def cumulative(
    rule: RuleLike,
    values: ArrayLike,
    *,
    reverse: bool = False,
    axis: int = -1,
    initial: float = 0.0,
) -> Inexact[Array, "*shape"]:
    """The running trapezoid integral of samples along one axis.

    F₀ = F_init and Fₖ = Fₖ₋₁ + ½ (xₖ − xₖ₋₁)(fₖ₋₁ + fₖ), so Fₖ ≈ F_init +
    ∫ from x₀ to xₖ of f. With ``reverse=True`` the integral runs from the far
    end, Fₖ ≈ F_init + ∫ from xₖ to x_{N−1} of f (optical depth from the top
    of a column).

    Algorithm:
        1. Form the interval increments Δₖ = ½ (xₖ − xₖ₋₁)(fₖ₋₁ + fₖ) for
           k = 1, …, N − 1, from the rule's nodes.
        2. Prefix-sum them with `jax.lax.associative_scan` (O(log N) depth),
           from the far end when ``reverse`` is set: a reversed scan reverses
           the increments, never the nodes, so every width keeps its sign.
        3. Prepend (or, reversed, append) a zero for the starting node and add
           F_init.

    The increments are the trapezoid rule on the rule's nodes (the cumulative
    form of functax's ``trapezoid``), so the last entry equals
    `integrix.rule.integrate` with a trapezoid rule on the same nodes; the
    rule's own weights are not used. The gradient of a prefix sum is a
    reversed prefix sum, which JAX differentiates without a custom rule.

    Args:
        rule: A rule with ``nodes`` x, N ≥ 1 of them, ascending or descending.
        values: The samples f, with N entries along ``axis``.
        reverse: Integrate from the last node towards the first.
        axis: The axis of ``values`` to integrate along.
        initial: The value F_init at the starting node.

    Returns:
        The running integral F, with the shape of the samples, in their dtype
        (complex when the nodes are; integer samples take the nodes' dtype).

    Raises:
        ValidationError: If the rule has no 1-D ``nodes``, ``axis`` is out of
            range, or the samples do not have one entry per node along it.

    Examples:
        >>> import equinox as eqx
        >>> import jax.numpy as jnp
        >>> class Trapezoid(eqx.Module):
        ...     nodes: jnp.ndarray
        ...     weights: jnp.ndarray
        >>> nodes = jnp.array([0.0, 1.0, 3.0])
        >>> rule = Trapezoid(nodes, jnp.array([0.5, 1.5, 1.0]))
        >>> values = jnp.array([1.0, 2.0, 3.0])
        >>> bool(jnp.allclose(cumulative(rule, values), jnp.array([0.0, 1.5, 6.5])))
        True
        >>> from_top = cumulative(rule, values, reverse=True)
        >>> bool(jnp.allclose(from_top, jnp.array([6.5, 5.0, 0.0])))
        True
        >>> cumulative(rule, jnp.ones((4, 3))).shape
        (4, 3)
    """
    rule_nodes = getattr(rule, "nodes", None) if isinstance(rule, RuleLike) else None
    if rule_nodes is None:
        raise ValidationError(
            "'rule' must be a rule with 'nodes' (and 'weights', as functax's "
            "Rule has) to form a running integral"
        )
    nodes = jnp.asarray(rule_nodes)
    if nodes.ndim != 1 or nodes.shape[0] == 0:
        raise ValidationError(
            f"'rule' must have 1-D nodes, at least one, got shape {nodes.shape}"
        )
    values = jnp.asarray(values)
    (position,) = normalise_axes(axis, values.ndim, 1)
    check_node_shape(values, (position,), nodes.shape)

    dtype = working_dtype(values.dtype, as_inexact(nodes).dtype)
    samples = move_first(values, position).astype(dtype)
    start_value = jnp.asarray(initial, dtype=dtype)
    if nodes.shape[0] == 1:
        # one node, no interval: F₀ = F_init
        return move_back(jnp.full_like(samples, start_value), position)
    # widths in the nodes' own precision, then in the working dtype
    widths = (nodes[1:] - nodes[:-1]).astype(dtype)
    # Δₖ = ½ (xₖ − xₖ₋₁)(fₖ₋₁ + fₖ)
    pair_sums = samples[:-1] + samples[1:]
    increments = 0.5 * einx.multiply("n ..., n -> n ...", pair_sums, widths)
    # Fₖ = ∑ⱼ≤ₖ Δⱼ, or ∑ⱼ>ₖ Δⱼ from the far end
    running = jax.lax.associative_scan(jnp.add, increments, reverse=reverse)
    start = jnp.zeros((1, *samples.shape[1:]), dtype=samples.dtype)
    pieces = [running, start] if reverse else [start, running]
    integral = jnp.concatenate(pieces) + start_value
    return move_back(integral, position)
