"""Expectations: 𝔼[f] by Monte Carlo, randomised QMC and importance sampling.

Each estimator returns an `Expectation` with a standard error. `monte_carlo`
averages draws already made (NumPyro's posterior samples, say) or drawn from
a sampler; `qmc` averages over randomly shifted Sobol or Halton points;
`importance` reweights draws from a proposal. `log_gauss_hermite` and
`log_simpson` return log-integrals for heavy-tailed integrands that underflow.
integrix averages; sampling from a posterior stays in NumPyro and pyrox.
"""

from __future__ import annotations

from integrix._src.expect._expectation import (
    Expectation,
    importance,
    monte_carlo,
    qmc,
)
from integrix._src.expect._log import log_gauss_hermite, log_simpson


__all__ = [
    "Expectation",
    "importance",
    "log_gauss_hermite",
    "log_simpson",
    "monte_carlo",
    "qmc",
]
