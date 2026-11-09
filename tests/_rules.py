"""Test-only rules, standing in for functax's until functax#44 lands.

integrix reads a rule's ``nodes`` and ``weights`` structurally (`RuleLike`),
so these small modules exercise the same code path functax's ``Rule`` will.
The weights follow the formulas in functax#44 and are checked against NumPy
and SciPy here, so a test that uses them tests integrix, not the stand-in.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


class StandInRule(eqx.Module):
    """Nodes and weights only, as functax's ``Rule`` is."""

    nodes: Float[Array, " node"]
    weights: Float[Array, " node"]


def trapezoid_rule(nodes: Float[Array, " node"]) -> StandInRule:
    """w₀ = (x₁ − x₀)/2, wₖ = (xₖ₊₁ − xₖ₋₁)/2, w_N = (x_N − x_{N−1})/2."""
    nodes = jnp.asarray(nodes)
    widths = jnp.diff(nodes)
    zero = jnp.zeros(1, dtype=nodes.dtype)
    left = jnp.concatenate([widths, zero])
    right = jnp.concatenate([zero, widths])
    return StandInRule(nodes=nodes, weights=0.5 * (left + right))


def gauss_legendre_rule(
    n_nodes: int, interval: tuple[float, float] = (-1.0, 1.0)
) -> StandInRule:
    """The n-point Gauss–Legendre rule on [a, b], mapped from NumPy's on [−1, 1]."""
    reference_nodes, reference_weights = np.polynomial.legendre.leggauss(n_nodes)
    lower, upper = interval
    # x = (a + b)/2 + (b − a)/2 ξ, w = (b − a)/2 ω
    half_width = 0.5 * (upper - lower)
    nodes = 0.5 * (lower + upper) + half_width * reference_nodes
    return StandInRule(
        nodes=jnp.asarray(nodes), weights=jnp.asarray(half_width * reference_weights)
    )
