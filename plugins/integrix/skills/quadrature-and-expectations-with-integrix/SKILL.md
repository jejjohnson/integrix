---
name: quadrature-and-expectations-with-integrix
description: Integrate in JAX with integrix — weights against samples or an integrand over one axis or several with masks and NaN handling, band-weighted means, log-integrals, running integrals forward and reverse, adaptive Gauss–Kronrod, tanh–sinh and Romberg quadrature with an error estimate under jit, and Monte Carlo, quasi-Monte Carlo and importance-sampling expectations with a standard error and weight diagnostics. Use when a task sums samples against quadrature weights, integrates over a spectrum, a column, an area or a sphere, needs a cumulative integral such as optical depth, integrates a function to a tolerance, or averages over draws, in a project that uses (or could use) integrix.
---

# Quadrature and expectations with integrix

integrix computes one integral, I = ∫ f dμ. This skill covers the two ways that do not step in time: a rule sums weights against samples, I ≈ ∑ₙ wₙ f(xₙ), and a scan builds the running integral Fₖ.
Adaptive rules and expectations are the same sum with the nodes chosen for you: to a tolerance, or by drawing.

It covers the primitives and strategies of integrix's quadrature, in the vocabulary of the `role-ladder` skill.
Time stepping, where the integrand depends on the solution, is in `integrix:time-stepping-with-integrix`.

## The contract

- A rule is data, nodes and weights, and comes from functax (`functax.quad`); a tuple of 1-D rules is a tensor rule, and a plain array is taken as the weights.
- `integrate(rule, values_or_fn)` returns an `Integral`: read `.value`; `.error` is `None` and `.converged` is true for a plain rule.
- An adaptive rule is called as `rule(fn, lower, upper, args=...)` and returns an `Integral` with `.error` and `.converged` filled in. It never raises on a missed tolerance.
- An expectation returns an `Expectation` with `.value` and `.standard_error`.
- Every call is batched over the axes it does not integrate, and runs under `jit`, `vmap` and `grad`. The library never jits: you do, once, at the top.

## Which one

| You have | You want | Use |
|---|---|---|
| Samples on nodes | the integral | `integrix.rule.integrate` |
| Samples and a weight function | the weighted mean | `integrix.rule.weighted_mean` |
| Log-values that underflow | the log-integral | `integrix.rule.log_integrate` |
| Samples along an axis | the integral up to every node | `integrix.scan.cumulative` |
| An integrand, finite interval | a tolerance | `integrix.adapt.GaussKronrod` |
| An integrand, singular end or infinite limit | a tolerance | `integrix.adapt.TanhSinh` |
| A smooth integrand | a tolerance | `integrix.adapt.Romberg` |
| Draws from the distribution | 𝔼[f] | `integrix.expect.monte_carlo` |
| A few dimensions on the unit cube | 𝔼[f], lower variance | `integrix.expect.qmc` |
| Draws from another distribution | 𝔼[f] under the target | `integrix.expect.importance` |

## Worked example

```python
import jax
import jax.numpy as jnp
from functax.quad import trapezoid

from integrix.adapt import GaussKronrod
from integrix.rule import integrate, weighted_mean
from integrix.scan import cumulative

# a column of absorber on 40 levels, and its spectrum on 200 wavelengths
height = jnp.linspace(0.0, 20.0, 40)  # km
wavelength = jnp.linspace(1.6, 1.7, 200)  # µm
density = jnp.exp(-height / 8.0)  # (40,)
cross_section = 0.05 * (1.0 + jnp.cos(200.0 * wavelength) ** 2)  # (200,)
levels = trapezoid(height)

# the column amount: one integral over height
column = integrate(levels, density).value
assert jnp.allclose(column, 8.0 * (1.0 - jnp.exp(-2.5)), rtol=1e-3)

# optical depth from the top down to every level, for every wavelength
extinction = cross_section[:, None] * density[None, :]  # (200, 40)
tau = cumulative(levels, extinction, reverse=True)  # (200, 40)
assert jnp.allclose(tau[:, 0], cross_section * column, rtol=1e-5)
transmittance = jnp.exp(-tau[:, 0])  # (200,)

# what an instrument band sees: the mean transmittance under its response
response = jnp.exp(-0.5 * ((wavelength - 1.65) / 0.01) ** 2)
band = weighted_mean(trapezoid(wavelength), transmittance, weight=response)
assert 0.0 < float(band) < 1.0


# the gradient with respect to the amount of absorber flows through all three
def band_transmittance(scale):
    tau = cumulative(levels, scale * extinction, reverse=True)[:, 0]
    return weighted_mean(trapezoid(wavelength), jnp.exp(-tau), weight=response)


assert float(jax.grad(band_transmittance)(1.0)) < 0.0

# the same column from the density as a function, to a tolerance
adaptive = GaussKronrod(rtol=1e-6)(lambda z: jnp.exp(-z / 8.0), 0.0, 20.0)
assert bool(adaptive.converged)
assert jnp.allclose(adaptive.value, 8.0 * (1.0 - jnp.exp(-2.5)), rtol=1e-5)
```

## Don't write it

| Don't write | Use |
|---|---|
| `jnp.trapezoid(f, x)` or `jnp.sum(w * f, axis=...)` | `integrate(rule, f).value` |
| `jnp.sum(w * R * L) / jnp.sum(w * R)` | `weighted_mean(rule, L, weight=R)` |
| `jnp.nansum` or `jnp.where(mask, f, 0.0)` before a sum | `where=` and `nan_policy="omit"` |
| `jax.scipy.special.logsumexp(jnp.log(w) + log_f)` | `log_integrate(rule, log_f)` |
| `jnp.cumsum` of trapezoid increments, or a flipped `cumsum` | `cumulative(rule, f, reverse=...)` |
| `scipy.integrate.quad`, `quadax.quadgk` in a traced model | `GaussKronrod()(fn, a, b)` |
| `jnp.mean(f(draws))` with a hand-made standard error | `monte_carlo(fn, draws)` |
| Hand-coded Gauss nodes or `cos(lat)` weights | a rule from `functax.quad` |

## The shelf

Each entry is one file with the names it covers, a worked example and the rules.
Open the one the task needs; do not read them all.

<!-- shelf:begin -->
| Role | Entry | Covers | Names |
|---|---|---|---|
| ① primitive | [Sums against a rule](shelf/rules-and-weights.md) | an integral as weights against samples over one axis or several, the mean under a weight function, the log-integral, and the result they return | 4 |
| ① primitive | [Running integrals](shelf/running-integrals.md) | the cumulative integral along one axis, from either end, from samples or from an integrand to a tolerance | 2 |
| ② strategy | [Adaptive quadrature](shelf/adaptive-quadrature.md) | an integrand integrated to a tolerance with an error estimate: peaked, smooth, singular at an end, or on an infinite interval | 3 |
| ② strategy | [Expectations](shelf/expectations.md) | an expectation with its standard error by Monte Carlo, quasi-Monte Carlo or importance sampling, the diagnostics of importance weights, and log-expectations of heavy-tailed integrands | 8 |
| setup | [Errors, the version and what to install](shelf/errors-and-installs.md) | the two exceptions, which checks still run under jit, the version string and the packages integrix brings with it | 5 |
<!-- shelf:end -->

## Before you finish

- Does every weighted sum go through `integrate`, `weighted_mean` or `log_integrate`, with the rule from functax?
- Is `.value` read from each `Integral`, and `.converged` checked after each adaptive rule?
- Is each mask passed as `where=`, not multiplied in?
- Is each estimate from draws reported with its standard error?

<!-- skill-tree:begin · generated by geostack scripts/skill_tree.py; do not edit by hand -->
## Builds on

Each card below says what a library under this skill guarantees.
A card is enough to call it.
Load a skill only when its card's last line applies.

### functax · base dependency · 0 of 8 units

It uses fewer than 3 names of each unit, so no card. The front door is `functax:function-approximation-with-functax`.

### gaussx · base dependency · 0 of 7 units

It uses fewer than 3 names of each unit, so no card. The front door is `gaussx:structured-gaussian-linalg`.

Used by (pointers, not skills to load): finitevolX, plumax, sesmax, vardax.

<!-- skill-tree:end -->
