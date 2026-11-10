"""Expectations: 𝔼[f] by Monte Carlo, randomised QMC and importance sampling.

Each estimator returns an `Expectation` with a standard error. `monte_carlo`
averages draws already made (NumPyro's posterior samples, say) or drawn from
a sampler; `qmc` averages over randomly shifted Sobol or Halton points;
`importance` reweights draws from a proposal. integrix averages; sampling
from a posterior stays in NumPyro and pyrox.
"""

from __future__ import annotations

from integrix._src.expect._expectation import (
    Expectation,
    importance,
    monte_carlo,
    qmc,
)


__all__ = [
    "Expectation",
    "importance",
    "monte_carlo",
    "qmc",
]
