---
numbering:
  enumerator: "10.%s"
---

# Expectations

## An emission rate is a distribution

A plume inversion does not return one emission rate. It returns a posterior over rates, $p(Q \mid \vect{y})$, usually as a set of samples from an MCMC run, and what we report (a mean, an interval, the probability that the rate exceeds a reporting threshold) is an integral against that posterior. A weather or ocean ensemble is the same thing in a different dress: each member is a draw, and the ensemble mean and spread are expectations over the forecast distribution.

So an expectation is an integral with a probability density for its measure,

$$\mathbb{E}_{p(\vect{x})}[f(\vect{x})] = \int f(\vect{x}) p(\vect{x}) \mathrm{d}\vect{x}$$ (eq-expectation)

with $\vect{x} \in \mathbb{R}^{D}$ and $D$ the dimension of the space we integrate over. The difference from the earlier chapters is that $D$ can be large (the dimension of a state, or of a parameter vector), and that we often can sample $p$ but not evaluate it. Both push us away from the rules of chapters 2 to 5 and toward random nodes. In integrix this is `integrix.expect`: [`monte_carlo`](xref:api#integrix.expect.monte_carlo), [`qmc`](xref:api#integrix.expect.qmc) and [`importance`](xref:api#integrix.expect.importance), each returning an [`Expectation`](xref:api#integrix.expect.Expectation) with its standard error.

## Monte Carlo: random nodes, equal weights

Monte Carlo is a weighted sum like every other rule, with nodes drawn from $p$ and equal weights. With $N_\mathcal{E}$ draws $\vect{x}^{(e)} \sim p$, $e \in \mathcal{E}$,

$$\hat{I} = \frac{1}{N_\mathcal{E}} \sum_{e \in \mathcal{E}} f(\vect{x}^{(e)}), \quad \hat{\sigma} = \frac{s}{\sqrt{N_\mathcal{E}}}$$ (eq-monte-carlo)

where $s^2$ is the sample variance of the $f(\vect{x}^{(e)})$ and $\hat{\sigma}$ the standard error of $\hat{I}$. The estimate is random, so it comes with an error bar, and the error bar shrinks like $N_\mathcal{E}^{-1/2}$.

We take a log-normal posterior for the emission rate, $\log Q \sim \mathcal{N}(\log 500, 0.3^2)$ with $Q$ in kg h⁻¹, whose mean is $500 e^{0.3^2 / 2} \approx 523$ kg h⁻¹, and estimate it from 1000 draws.

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
from integrix import expect, rule

# Shapes: N_E draws of a scalar emission rate
key = jax.random.key(0)
log_mean, log_std = np.log(500.0), 0.3


def posterior_draws(key, n):
    """Q⁽ᵉ⁾ = exp(log 500 + 0.3 ε⁽ᵉ⁾), ε⁽ᵉ⁾ ~ 𝒩(0, 1)."""
    # user
    return jnp.exp(log_mean + log_std * jax.random.normal(key, (n,)))


# today (integrix)
# Î = (1/N_E) ∑ₑ Q⁽ᵉ⁾, σ̂ = s/√N_E
rate = expect.monte_carlo(lambda q: q, posterior_draws, 1000, key=key)
exact_rate = 500.0 * np.exp(log_std**2 / 2.0)
print(
    f"E[Q] = {float(rate.value):.1f} +/- {float(rate.standard_error):.1f} kg/h "
    f"(exact {exact_rate:.1f})"
)
```

```text
E[Q] = 520.1 +/- 5.2 kg/h (exact 523.0)
```

The estimate is within one standard error of the exact mean, which is what the standard error is for: it says how far off we should expect to be, without knowing the answer.

## Global and local in many dimensions

The rate $N_\mathcal{E}^{-1/2}$ looks slow next to the spectral and algebraic rates of chapter 3. Its strength is that it does not depend on $D$. A tensor-product rule ([](#eq-tensor-rule)) with $n$ nodes per axis needs $n^D$ evaluations, so even a modest $n = 10$ in an 8-dimensional parameter space costs $10^8$ evaluations, and its error per axis falls only like $n^{-q}$ for a local rule of order $q$. Every node of a tensor rule sits on a grid that fills the whole box, so it is global in the worst way: refining one axis multiplies the cost of all the others. Random nodes do not have a grid, and their error depends only on the variance of $f$, not on how many axes it has.

## Quasi-Monte Carlo

Random nodes are clumpy: some regions of the box get more points than they should and some get none. Quasi-Monte Carlo replaces them with low-discrepancy sequences (Sobol, Halton) that fill the unit cube $[0, 1)^D$ evenly by construction. The Koksma–Hlawka inequality bounds the error by the variation of $f$ times the star discrepancy of the points, and for these sequences the discrepancy falls like $(\log N_\mathcal{E})^D / N_\mathcal{E}$, so for a smooth integrand in moderate dimension the error falls close to $N_\mathcal{E}^{-1}$ instead of $N_\mathcal{E}^{-1/2}$ (Owen, 2013). A deterministic sequence has no error bar, so integrix randomises it with $R$ random shifts and takes the spread of the $R$ shifted estimates as the standard error.

We compare the two on a smooth product in $D = 8$, $f(\vect{u}) = \prod_{d=1}^{8} (1 + \frac{1}{2} (u_d - \frac{1}{2}))$ over the unit cube, whose exact mean is 1, and fit the slope of the standard error against the number of evaluations.

```python
# Shapes: D = 8; N_E = 16 shifts × n points
dimension = 8
mc_key, qmc_key = jax.random.split(key)


def smooth_product(point):
    """f(u) = ∏_d (1 + (u_d − ½)/2), with 𝔼[f] = 1 on [0, 1)⁸."""
    # user
    return jnp.prod(1.0 + 0.5 * (point - 0.5))


def uniform_draws(key, n):
    """N_E points uniform on [0, 1)⁸."""
    # user
    return jax.random.uniform(key, (n, dimension))


evaluations, mc_errors, qmc_errors = [], [], []
for n_points in (256, 1024, 4096, 16384):
    n_total = 16 * n_points
    # today (integrix)
    plain = expect.monte_carlo(smooth_product, uniform_draws, n_total, key=mc_key)
    # today (integrix)
    shifted = expect.qmc(smooth_product, dimension, n_points, n_shifts=16, key=qmc_key)
    evaluations.append(n_total)
    mc_errors.append(float(plain.standard_error))
    qmc_errors.append(float(shifted.standard_error))
    print(f"N_E = {n_total:6d}: MC {mc_errors[-1]:.1e}, QMC {qmc_errors[-1]:.1e}")
mc_slope = np.polyfit(np.log(evaluations), np.log(mc_errors), 1)[0]
qmc_slope = np.polyfit(np.log(evaluations), np.log(qmc_errors), 1)[0]
print(f"slope: MC {mc_slope:.2f}, QMC {qmc_slope:.2f}")
```

```text
N_E =   4096: MC 6.7e-03, QMC 5.3e-04
N_E =  16384: MC 3.3e-03, QMC 8.9e-05
N_E =  65536: MC 1.7e-03, QMC 3.2e-05
N_E = 262144: MC 8.3e-04, QMC 6.6e-06
slope: MC -0.50, QMC -1.03
```

Monte Carlo follows $N_\mathcal{E}^{-1/2}$ to two digits and quasi-Monte Carlo falls about twice as fast, so at the largest size the same budget buys an error more than a hundred times smaller. The advantage shrinks as $D$ grows or $f$ gets rougher, since the $(\log N_\mathcal{E})^D$ factor and the variation both grow, but for the smooth, moderate-dimensional integrals of a sensitivity study or an uncertainty propagation it is a large saving for no extra code.

## Importance sampling

Often we cannot sample the posterior directly but can evaluate it up to a constant, as prior times likelihood. Importance sampling draws from a proposal $q$ we can sample, and corrects with weights,

$$\hat{I} = \frac{\sum_e w^{(e)} f(\vect{x}^{(e)})}{\sum_e w^{(e)}}, \quad w^{(e)} = \frac{\tilde{p}(\vect{x}^{(e)})}{q(\vect{x}^{(e)})}, \quad N_\text{eff} = \frac{\left( \sum_e w^{(e)} \right)^2}{\sum_e (w^{(e)})^2}$$ (eq-importance)

with $\vect{x}^{(e)} \sim q$, $\tilde{p}$ the unnormalised target, and $N_\text{eff}$ the effective sample size. Dividing by the sum of the weights is what lets $\tilde{p}$ go without its normalising constant (self-normalisation). The weights are the quadrature weights here, now random and unequal, and $N_\text{eff}$ says how many equal-weight draws they are worth. A proposal that misses the posterior's mass shows up as a small $N_\text{eff}$ long before the standard error looks alarming.

We update a log-normal prior on the emission rate, $\log Q \sim \mathcal{N}(\log 400, 0.5^2)$, with one mass-balance estimate of 520 kg h⁻¹ and a 10 % error, using the prior as the proposal. The reference is the same expectation on a fine grid with [`integrate`](xref:api#integrix.rule.integrate), since this toy posterior is one-dimensional.

```python
# Shapes: N_E = 4000 draws of x = log Q
prior_mean, prior_std = np.log(400.0), 0.5


def prior_log_density(log_rate):
    """log q(x) = log 𝒩(x; log 400, 0.5²)."""
    # user
    return stats.norm.logpdf(log_rate, prior_mean, prior_std)


def target_log_density(log_rate):
    """log p̃(x) = log q(x) + log 𝒩(eˣ; 520, 52²)."""
    # user
    return prior_log_density(log_rate) + stats.norm.logpdf(
        jnp.exp(log_rate), 520.0, 52.0
    )


def prior_draws(key, n):
    """x⁽ᵉ⁾ ~ 𝒩(log 400, 0.5²)."""
    # user
    return prior_mean + prior_std * jax.random.normal(key, (n,))


# today (integrix)
# Î = ∑ₑ w⁽ᵉ⁾ eˣ⁽ᵉ⁾ / ∑ₑ w⁽ᵉ⁾, w⁽ᵉ⁾ = p̃/q
posterior_rate = expect.importance(
    jnp.exp, prior_draws, prior_log_density, target_log_density, 4000, key=key
)
# reference: the same ratio of integrals on a fine grid in x
grid = jnp.linspace(prior_mean - 8 * prior_std, prior_mean + 8 * prior_std, 20001)
spacing = grid[1] - grid[0]
grid_weights = jnp.full(20001, spacing).at[jnp.array([0, -1])].set(spacing / 2)
log_target = target_log_density(grid)
density = jnp.exp(log_target - jnp.max(log_target))
# today (integrix)
# 𝔼[Q] = ∫ eˣ p̃(x) dx / ∫ p̃(x) dx
reference = rule.weighted_mean(grid_weights, jnp.exp(grid), weight=density)
print(
    f"E[Q | y] = {float(posterior_rate.value):.1f} "
    f"+/- {float(posterior_rate.standard_error):.1f} kg/h "
    f"(grid {float(reference):.1f})"
)
print(
    f"effective sample size: {float(posterior_rate.effective_sample_size):.0f} of 4000"
)
```

```text
E[Q | y] = 508.6 +/- 1.2 kg/h (grid 509.8)
effective sample size: 992 of 4000
```

The estimate agrees with the grid within about one standard error, and the effective sample size says that the 4000 prior draws are worth about a quarter as many posterior draws, because the likelihood is narrower than the prior and most draws land where it is small.

## Gaussian expectations with a closed rule

When $p$ is Gaussian there is a third option: a deterministic rule built for the Gaussian measure. Gauss–Hermite is the Gauss rule of chapter 3 for the weight $e^{-x^2}$, exact for polynomials up to degree $2n - 1$ in each axis, and the unscented transform uses $2 D + 1$ sigma points that match the first two moments. These are global rules again, with the strengths and the $n^D$ cost of chapter 3 (Gauss–Hermite is a tensor rule; the unscented transform avoids the cost by matching moments only). They live in gaussx (`GaussHermiteIntegrator`, `UnscentedIntegrator`), because their natural home is next to the Gaussian solves and filters that use them, and integrix covers the expectations that have no closed rule.
