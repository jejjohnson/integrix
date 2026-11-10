"""Rules for the tests: functax's, under the names the tests use, and a bare one.

`trapezoid_rule` and `gauss_legendre_rule` are functax.quad's `trapezoid` and
`gauss_legendre` (functax#44, in functax 0.0.6), which replace the stand-ins the
tests used before functax had rules. `StandInRule` stays as a bare module with
``nodes`` and ``weights`` only: integrix reads a rule structurally (`RuleLike`),
and some tests need a rule functax's `Rule` would reject (2-D weights inside a
tensor rule) or one with weights that are not a quadrature rule's.
"""

from __future__ import annotations

import equinox as eqx
import functax.quad as fx_quad
from jaxtyping import Array, ArrayLike, Float


class StandInRule(eqx.Module):
    """Nodes and weights only, the `RuleLike` protocol and nothing else."""

    nodes: Float[Array, " node *rest"]
    weights: Float[Array, " node *rest"]


def trapezoid_rule(nodes: Float[ArrayLike, " node"]) -> fx_quad.Rule:
    """functax's composite trapezoid rule on the given nodes."""
    return fx_quad.trapezoid(nodes)


def gauss_legendre_rule(
    n_nodes: int, interval: tuple[float, float] = (-1.0, 1.0)
) -> fx_quad.Rule:
    """functax's n-point Gauss–Legendre rule on [a, b] (host-side float64)."""
    return fx_quad.gauss_legendre(n_nodes, interval=interval)
