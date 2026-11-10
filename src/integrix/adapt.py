"""Adaptive quadrature: integrals to a tolerance, with an error estimate.

`GaussKronrod` integrates on a finite interval to a tolerance, under jit with
a static buffer of subintervals, and differentiates through a custom JVP. It
returns an `integrix.rule.Integral` whose ``error`` and ``converged`` are
filled in. Written on quadax's design, with no dependency on quadax.
"""

from __future__ import annotations

from integrix._src.adapt._gauss_kronrod import GaussKronrod


__all__ = [
    "GaussKronrod",
]
