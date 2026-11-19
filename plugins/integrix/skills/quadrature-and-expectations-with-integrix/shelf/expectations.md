# Expectations

Role: ② strategy (`role-ladder`, `references/strategy.md`).
Unit: integrals.

Open this when the integral is an average over draws: a posterior mean from samples, an expectation in a few dimensions on a low-discrepancy set, or a reweighting from a proposal.

## Names

All of them live in `integrix.expect`.

| Name | What it is |
|---|---|
| `monte_carlo(fn, samples_or_sampler, n=None, *, key=None)` | (1/S) ∑ₛ f(xₛ) over draws you hold (leading axis S ≥ 2), or drawn by `sampler(key, n)`. |
| `qmc(fn, dim, n, *, sequence="sobol", n_shifts=16, transform=None, key)` | Randomised quasi-Monte Carlo over [0, 1)^D with Sobol (D ≤ 64) or Halton points; the error comes from the random shifts. |
| `importance(fn, proposal_sample, proposal_log_prob, target_log_prob, n, *, diagnostics=False, key)` | Self-normalised importance sampling, ∑ₛ wₛ f(xₛ) / ∑ₛ wₛ with xₛ from the proposal. |
| `Expectation` | The result: `value`, `standard_error`, `n_samples`, and for importance sampling `effective_sample_size` and (with `diagnostics=True`) `pareto_k`. |
| `effective_sample_size(log_weights)` | (∑ wₛ)² / ∑ wₛ² from log weights along the last axis, with no weight exponentiated. |
| `pareto_smoothed_log_weights(log_weights)` | The Pareto-smoothed log weights and the tail shape k̂. |
| `log_gauss_hermite(log_fn, mean, std, *, n=32)` | log 𝔼[f(Q)] for a lognormal Q, by Gauss–Hermite quadrature in log Q. |
| `log_simpson(log_fn, lower, upper, *, n=2001)` | log ∫ f(q) dq by Simpson's rule on a uniform grid in log q, for 0 < lower < upper. |

## Use it

```python
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats

from integrix.expect import (
    Expectation,
    effective_sample_size,
    importance,
    log_gauss_hermite,
    log_simpson,
    monte_carlo,
    pareto_smoothed_log_weights,
    qmc,
)

key = jax.random.key(0)

# draws you already hold, such as posterior samples: 𝔼[x²] = 1
draws = jax.random.normal(key, (4096,))
estimate = monte_carlo(jnp.square, draws)
assert isinstance(estimate, Expectation) and estimate.n_samples == 4096
assert abs(estimate.value - 1.0) < 4 * estimate.standard_error

# quasi-Monte Carlo on the unit cube: ∫ ∏ cos(uᵢ) du = sin(1)³
cube = qmc(lambda u: jnp.prod(jnp.cos(u)), 3, 256, key=key)
assert abs(cube.value - jnp.sin(1.0) ** 3) < 4 * cube.standard_error
assert float(cube.standard_error) < float(estimate.standard_error)

# importance sampling from a wider proposal, with the tail diagnostic
weighted = importance(
    jnp.square,
    lambda key, n: 2.0 * jax.random.normal(key, (n,)),
    lambda x: stats.norm.logpdf(x, scale=2.0),
    lambda x: stats.norm.logpdf(x),
    4096,
    diagnostics=True,
    key=key,
)
assert abs(weighted.value - 1.0) < 4 * weighted.standard_error
assert 1.0 < float(weighted.effective_sample_size) < 4096.0
assert float(weighted.pareto_k) < 0.7  # above 0.7, do not trust the estimate

# the diagnostics on their own
log_weights = 0.3 * jax.random.normal(key, (1000,))
assert float(effective_sample_size(log_weights)) < 1000.0
smoothed, k_hat = pareto_smoothed_log_weights(log_weights)
assert smoothed.shape == (1000,) and float(k_hat) < 0.7

# log-expectations that would underflow: log 𝔼[Q] = μ + σ²/2 for a lognormal Q
assert jnp.allclose(log_gauss_hermite(jnp.log, 0.2, 0.5), 0.2 + 0.5 * 0.5**2, rtol=1e-5)
# log ∫₁ᵉ dq/q = log 1 = 0
assert jnp.allclose(log_simpson(lambda q: -jnp.log(q), 1.0, jnp.e), 0.0, atol=1e-5)
```

## Rules

- `fn` is applied to one draw; the estimator maps it over the draws. A key is required whenever something is drawn.
- Report the standard error with the value. For `qmc` it is the spread over `n_shifts` random shifts, so `n_shifts` is at least 2.
- `transform=` maps a point of the unit cube to the integration space, such as an inverse CDF.
- `target_log_prob` may be unnormalised: the weights are self-normalised.
- `pareto_k` above 0.7 says the importance weights have too heavy a tail to trust the estimate. The estimate itself is not smoothed.
- `log_fn` in `log_gauss_hermite` and `log_simpson` takes the nodes with the node axis first and returns it first.
- integrix averages. Sampling from a posterior stays in NumPyro and pyrox, and Gaussian expectations with a closed rule (unscented, sigma points) are not here.
