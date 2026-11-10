"""Adaptive quadrature: integrals to a tolerance, with an error estimate.

`GaussKronrod` integrates on a finite interval to a tolerance, under jit with
a static buffer of subintervals, and differentiates through a custom JVP.
`TanhSinh` handles endpoint singularities and infinite limits, `Romberg`
smooth integrands, and `cumulative_adaptive` the running integral to each
node. Each returns an `integrix.rule.Integral` whose ``error`` and
``converged`` are filled in. Written on quadax's design, with no dependency on quadax.
"""

from __future__ import annotations

from integrix._src.adapt._cumulative import cumulative_adaptive
from integrix._src.adapt._gauss_kronrod import GaussKronrod
from integrix._src.adapt._romberg import Romberg
from integrix._src.adapt._tanh_sinh import TanhSinh


__all__ = [
    "GaussKronrod",
    "Romberg",
    "TanhSinh",
    "cumulative_adaptive",
]
