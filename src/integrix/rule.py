"""Rules: integrals as weights against samples, I ≈ ∑ₙ wₙ f(xₙ).

`integrate` contracts values (or an integrand evaluated at a rule's nodes)
with quadrature weights over one axis or several, batched over the rest;
`weighted_mean` divides by the integral of a weight function, as a band
radiance does. The rules themselves (nodes and weights) come from functax, or
are a plain weights array such as a spectral response or cell areas.
"""

from __future__ import annotations

from integrix._src.rule._rule import Integral, integrate, weighted_mean


__all__ = [
    "Integral",
    "integrate",
    "weighted_mean",
]
