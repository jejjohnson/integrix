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
from jaxtyping import Array, Float, PRNGKeyArray

# Shapes: N_E draws of a scalar emission rate
key = jax.random.key(0)
# μ = log 500, σ = 0.3
log_mean, log_std = np.log(500.0), 0.3


# user
def posterior_draws(
    key: PRNGKeyArray,
    n: int,
) -> Float[Array, "draw"]:
    """Q⁽ᵉ⁾ = exp(log 500 + 0.3 ε⁽ᵉ⁾), ε⁽ᵉ⁾ ~ 𝒩(0, 1)."""
    # εₑ ~ 𝒩(0, 1)
    noise = jax.random.normal(key, (n,))                                # → (N_E,)
    # Qₑ = exp(μ + σ εₑ)
    return jnp.exp(log_mean + log_std * noise)


# today (integrix)
# Î = (1/N_E) ∑ₑ Q⁽ᵉ⁾, σ̂ = s/√N_E
rate = expect.monte_carlo(lambda q: q, posterior_draws, 1000, key=key)  # → ()
# 𝔼[Q] = exp(μ + σ²/2)
exact_rate = 500.0 * np.exp(log_std**2 / 2.0)                           # → ()
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
# Shapes: D = 8 dimensions; N_E = 16 shifts × n points
dimension = 8
# one independent key per estimator
mc_key, qmc_key = jax.random.split(key)


# user
def smooth_product(
    point: Float[Array, "dim"],
) -> Float[Array, ""]:
    """f(u) = ∏_d (1 + (u_d − ½)/2), with 𝔼[f] = 1 on [0, 1)⁸."""
    # u_d − ½
    centred = point - 0.5                                              # → (D,)
    # t_d = 1 + (u_d − ½)/2
    factors = 1.0 + 0.5 * centred                                      # → (D,)
    # f = ∏_d t_d
    return jnp.prod(factors)


# user
def uniform_draws(
    key: PRNGKeyArray,
    n: int,
) -> Float[Array, "draw dim"]:
    """N_E points uniform on [0, 1)⁸."""
    # uₑ ~ 𝒰[0, 1)ᴰ
    points = jax.random.uniform(key, (n, dimension))                   # → (N_E, D)
    return points


evaluations, mc_errors, qmc_errors = [], [], []
for n_points in (256, 1024, 4096, 16384):
    # N_E = 16 n
    n_total = 16 * n_points                                            # → ()
    # today (integrix)
    # Î = (1/N_E) ∑ₑ f(uₑ), σ̂ = s/√N_E
    plain = expect.monte_carlo(                                        # → ()
        smooth_product, uniform_draws, n_total, key=mc_key
    )
    # today (integrix)
    # Î = (1/16) ∑ₛ (1/n) ∑ᵢ f(uᵢ ⊕ Δₛ), σ̂ from the 16 shifts
    shifted = expect.qmc(                                              # → ()
        smooth_product, dimension, n_points, n_shifts=16, key=qmc_key
    )
    evaluations.append(n_total)
    mc_errors.append(float(plain.standard_error))
    qmc_errors.append(float(shifted.standard_error))
    print(f"N_E = {n_total:6d}: MC {mc_errors[-1]:.1e}, QMC {qmc_errors[-1]:.1e}")
# log σ̂ = slope · log N_E + c, fitted for MC
mc_slope = np.polyfit(np.log(evaluations), np.log(mc_errors), 1)[0]    # → ()
# log σ̂ = slope · log N_E + c, fitted for QMC
qmc_slope = np.polyfit(np.log(evaluations), np.log(qmc_errors), 1)[0]  # → ()
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
# Shapes: N_E = 4000 draws of x = log Q; N = 20,001 grid nodes; *B any batch shape
# μ_q = log 400, σ_q = 0.5
prior_mean, prior_std = np.log(400.0), 0.5


# user
def prior_log_density(
    log_rate: Float[Array, "*batch"],
) -> Float[Array, "*batch"]:
    """log q(x) = log 𝒩(x; log 400, 0.5²)."""
    # log q(x) = log 𝒩(x; μ_q, σ_q²)
    log_q = stats.norm.logpdf(log_rate, prior_mean, prior_std)       # → (*B,)
    return log_q


# user
def target_log_density(
    log_rate: Float[Array, "*batch"],
) -> Float[Array, "*batch"]:
    """log p̃(x) = log q(x) + log 𝒩(eˣ; 520, 52²)."""
    # log q(x)
    log_q = prior_log_density(log_rate)                              # → (*B,)
    # Q = eˣ
    rate = jnp.exp(log_rate)                                         # → (*B,)
    # log 𝒩(Q; 520, 52²)
    log_like = stats.norm.logpdf(rate, 520.0, 52.0)                  # → (*B,)
    # log p̃(x) = log q(x) + log 𝒩(eˣ; 520, 52²)
    return log_q + log_like


# user
def prior_draws(
    key: PRNGKeyArray,
    n: int,
) -> Float[Array, "draw"]:
    """x⁽ᵉ⁾ ~ 𝒩(log 400, 0.5²)."""
    # εₑ ~ 𝒩(0, 1)
    noise = jax.random.normal(key, (n,))                             # → (N_E,)
    # xₑ = μ_q + σ_q εₑ
    return prior_mean + prior_std * noise


# today (integrix)
# Î = ∑ₑ w⁽ᵉ⁾ eˣ⁽ᵉ⁾ / ∑ₑ w⁽ᵉ⁾, w⁽ᵉ⁾ = p̃/q
posterior_rate = expect.importance(                                  # → ()
    jnp.exp, prior_draws, prior_log_density, target_log_density, 4000, key=key
)
# reference: the same ratio of integrals on a fine grid in x
# x_min = μ_q − 8σ_q
lower = prior_mean - 8 * prior_std                                   # → ()
# x_max = μ_q + 8σ_q
upper = prior_mean + 8 * prior_std                                   # → ()
# xᵢ evenly spaced from x_min to x_max
grid = jnp.linspace(lower, upper, 20001)                             # → (N,)
# Δx = x₁ − x₀
spacing = grid[1] - grid[0]                                          # → ()
# indices of the two end nodes, which get half weight
ends = jnp.array([0, -1])
# wᵢ = Δx, with Δx/2 at both ends (trapezoid)
grid_weights = jnp.full(20001, spacing).at[ends].set(spacing / 2)    # → (N,)
# log p̃(xᵢ)
log_target = target_log_density(grid)                                # → (N,)
# m = maxᵢ log p̃(xᵢ)
peak = jnp.max(log_target)                                           # → ()
# p̃ᵢ ∝ exp(log p̃(xᵢ) − m)
density = jnp.exp(log_target - peak)                                 # → (N,)
# Qᵢ = exp(xᵢ)
rates = jnp.exp(grid)                                                # → (N,)
# today (integrix)
# 𝔼[Q] = ∫ eˣ p̃(x) dx / ∫ p̃(x) dx
reference = rule.weighted_mean(grid_weights, rates, weight=density)  # → ()
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

## When importance weights lie

The effective sample size says how many equal-weight draws the weights are worth, but it is itself estimated from the same weights, and when their distribution has a heavy right tail it can look healthy while the estimate is badly wrong. A heavy tail means that the few largest weights, the ones that decide the answer, are not yet represented in the sample. Pareto-smoothed importance sampling makes this measurable (Vehtari et al.): it fits a generalised Pareto distribution to the largest $M = \lceil \min(N_\mathcal{E}/5, 3\sqrt{N_\mathcal{E}}) \rceil$ weights and reports the fitted tail shape $\hat{k}$. The weight variance is finite only for $k < 1/2$, and above about $\hat{k} = 0.7$ the estimate should not be trusted whatever the effective sample size says. In integrix, [`effective_sample_size`](xref:api#integrix.expect.effective_sample_size) and [`pareto_smoothed_log_weights`](xref:api#integrix.expect.pareto_smoothed_log_weights) compute the two diagnostics from log weights, and `importance(..., diagnostics=True)` reports $\hat{k}$ as `pareto_k`.

We estimate $\mathbb{E}[X^2] = 1$ for $X \sim \mathcal{N}(0, 1)$ with proposals $\mathcal{N}(0, s^2)$ that get narrower. For this pair the weights $w \propto e^{x^2 (1/s^2 - 1)/2}$ have a Pareto tail with shape $k = 1 - s^2$ when $s < 1$, so we know the right answer for $\hat{k}$ too.

```python
# Shapes: N_E = 4000 draws of a scalar x
draws_key = jax.random.key(0)
print(" s    estimate   std err    ESS    k_hat   k = 1 - s^2")
for proposal_scale in (2.0, 1.0, 0.8, 0.6, 0.4, 0.3):

    # user
    def proposal_draws(
        key: PRNGKeyArray, n: int, scale: float = proposal_scale
    ) -> Float[Array, "draw"]:
        """x⁽ᵉ⁾ ~ 𝒩(0, s²)."""
        # x = s ε, ε ~ 𝒩(0, 1)
        return scale * jax.random.normal(key, (n,))

    # user
    def proposal_log_density(
        x: Float[Array, ""], scale: float = proposal_scale
    ) -> Float[Array, ""]:
        """log q(x) = log 𝒩(x; 0, s²)."""
        # log 𝒩(x; 0, s²)
        return stats.norm.logpdf(x, scale=scale)

    # today (integrix)
    # Î = ∑ₑ w⁽ᵉ⁾ x⁽ᵉ⁾² / ∑ₑ w⁽ᵉ⁾, with N_eff and k̂ from the same log weights
    estimate = expect.importance(                          # → value ()
        jnp.square,
        proposal_draws,
        proposal_log_density,
        stats.norm.logpdf,
        4000,
        diagnostics=True,
        key=draws_key,
    )
    # k = 1 − s² for s < 1, no tail otherwise
    true_k = 1.0 - proposal_scale**2 if proposal_scale < 1.0 else float("nan")
    print(
        f"{proposal_scale:3.1f}  {float(estimate.value):9.4f}  "
        f"{float(estimate.standard_error):8.4f}  "
        f"{float(estimate.effective_sample_size):6.0f}  "
        f"{float(estimate.pareto_k):6.2f}  {true_k:6.2f}"
    )
```

```text
 s    estimate   std err    ESS    k_hat   k = 1 - s^2
2.0     0.9844    0.0176    2653   -1.78     nan
1.0     0.9913    0.0221    4000     nan     nan
0.8     0.9679    0.0371    3456    0.31    0.36
0.6     0.8488    0.0604    1540    0.53    0.64
0.4     0.5478    0.0562     456    0.68    0.84
0.3     0.3526    0.0397     282    0.74    0.91
```

A wide proposal ($s = 2$) is safe: the weights are bounded, $\hat{k}$ is negative, and the estimate is within a standard error of 1. At $s = 1$ the proposal is the target, every weight is equal, and there is no tail to fit, so $\hat{k}$ is NaN. As the proposal narrows the estimate drifts away from 1 while its own standard error stays small: at $s = 0.3$ it reports about $0.35 \pm 0.04$, some sixteen standard errors from the truth, with an effective sample size still in the hundreds. $\hat{k}$ is the number that warns: it passes $1/2$ (infinite weight variance) at $s = 0.6$ and $0.7$ at $s = 0.3$. It follows $k = 1 - s^2$ but underestimates it at this sample size, the more so the heavier the tail, so a $\hat{k}$ near the threshold is already a reason to change the proposal.

## Heavy tails in log space

The methane plumes of an inventory follow a heavy-tailed size distribution: most sources are small, and a few super-emitters carry a large share of the total. Combining that distribution with an instrument's detection probability gives the integrals of an inventory likelihood, such as the fraction of sources an instrument detects,

$$\bar{P} = \int_0^\infty P_d(q) f(q) \mathrm{d}q$$ (eq-detected-fraction)

where $q$ is an emission rate in kg h⁻¹, $f$ its density and $P_d$ the detection probability. Two things go wrong in linear $q$. A grid uniform in $q$ cannot resolve a density whose mass sits at tens of kg h⁻¹ and whose tail reaches $10^5$, and a likelihood built from many such factors underflows long before it is summed. Both are fixed by working in $u = \log q$. The Gauss–Hermite rule of the last section in $\log q$ suits a lognormal $f$, and Simpson's rule on a uniform grid in $u$ resolves a lognormal body and a power-law tail alike (`log_simpson` uses $\mathrm{d}q = e^u \mathrm{d}u$). The sum itself is a log-sum-exp, $\log \sum_n w_n f_n = \mathrm{logsumexp}_n (\log w_n + \log f_n)$, which [`log_integrate`](xref:api#integrix.rule.log_integrate) computes from $\log f$ without ever leaving log space.

We take a lognormal $f$ with median 100 kg h⁻¹ and $\sigma = 1.5$ in $\log q$, and a detection probability that reaches one half at 200 kg h⁻¹, $P_d(q) = 1 - 2^{-q/200}$. A 200-node Gauss–Hermite rule is the reference.

```python
# Shapes: N = 2001 nodes; K = 32 or 200 Hermite nodes
# μ = log 100, σ = 1.5 (in log q); q₅₀ = 200 kg/h
log_median, log_spread, half_detection = np.log(100.0), 1.5, 200.0


# user
def log_detection(rate: Float[Array, "*batch"]) -> Float[Array, "*batch"]:
    """log P_d(q) = log(1 − 2^(−q/q₅₀))."""
    # 2^(−q/q₅₀) = exp(−q log 2 / q₅₀)
    exponent = -np.log(2.0) * rate / half_detection             # → (*B,)
    # log(1 − e^exponent)
    return jnp.log(-jnp.expm1(exponent))


# user
def log_density(rate: Float[Array, "*batch"]) -> Float[Array, "*batch"]:
    """log f(q), the lognormal density with median e^μ and spread σ."""
    # log q
    log_rate = jnp.log(rate)                                    # → (*B,)
    # (log q − μ)² / 2σ²
    misfit = (log_rate - log_median) ** 2 / (2.0 * log_spread**2)
    # log f = −log q − log(σ √(2π)) − (log q − μ)²/2σ²
    return -log_rate - np.log(log_spread * np.sqrt(2.0 * np.pi)) - misfit


# today (integrix)
# log P̄ by Gauss–Hermite in log q, 200 nodes
reference = expect.log_gauss_hermite(log_detection, log_median, log_spread, n=200)
# today (integrix)
# log P̄ by Gauss–Hermite in log q, the default 32 nodes
hermite = expect.log_gauss_hermite(log_detection, log_median, log_spread)


# user
def log_integrand(rate: Float[Array, "*batch"]) -> Float[Array, "*batch"]:
    """log P_d(q) + log f(q)."""
    # log(P_d f)
    return log_detection(rate) + log_density(rate)


# today (integrix)
# log P̄ by Simpson on a uniform grid in log q over [10⁻³, 10⁷]
simpson = expect.log_simpson(log_integrand, 1e-3, 1e7)
# P̄ = e^(log P̄)
exact = float(jnp.exp(reference))
print(f"reference P = {exact:.10f}")
print(f"Gauss-Hermite (32), log q:  error {abs(float(jnp.exp(hermite)) - exact):.1e}")
print(f"Simpson (2001), log q:      error {abs(float(jnp.exp(simpson)) - exact):.1e}")
for upper in (1e4, 1e5):
    # qₙ uniform on [0, q_max], kg/h, and the trapezoid weights
    rates = jnp.linspace(0.0, upper, 2001)                      # → (N,)
    # h = q₁ − q₀
    spacing = rates[1] - rates[0]
    # wₙ = h inside, h/2 at the ends
    weights = jnp.full(2001, spacing).at[jnp.array([0, -1])].set(spacing / 2)
    # q kept away from 0, where log q is −∞
    positive = jnp.maximum(rates, 1e-300)                       # → (N,)
    # P_d(qₙ) f(qₙ), 0 at q = 0
    values = jnp.where(rates > 0, jnp.exp(log_integrand(positive)), 0.0)
    # today (integrix)
    # P̄ ≈ ∑ₙ wₙ P_d(qₙ) f(qₙ) on the linear grid
    linear = rule.integrate(weights, values).value              # → ()
    print(f"trapezoid (2001), q to {upper:.0e}: error {abs(float(linear) - exact):.1e}")
```

```text
reference P = 0.3845823341
Gauss-Hermite (32), log q:  error 9.2e-07
Simpson (2001), log q:      error 7.7e-15
trapezoid (2001), q to 1e+04: error 1.1e-03
trapezoid (2001), q to 1e+05: error 6.8e-03
```

The two log-space rules agree with the reference to about $10^{-6}$ (32 Hermite nodes) and $10^{-14}$ (Simpson), while the same 2001 nodes spread uniformly in $q$ miss by 0.3 %, and by almost 2 % when the grid reaches $10^5$, because each extra decade of range takes nodes away from the body. The underflow half of the story needs no grid at all: an inventory likelihood multiplies such a fraction into hundreds of per-source terms, and once their log passes about $-745$ (the smallest positive float64 is near $e^{-745}$), the linear product is zero.

```python
# Shapes: scalars
# log L = log P̄ + ∑ of the other log terms, here −1200
other_terms = -1200.0
# log L = log P̄ + the other terms: finite
log_likelihood = float(reference) + other_terms
# L = P̄ · e^(other terms), formed in linear space: e^(−1200) is 0 in float64
linear_likelihood = float(jnp.exp(reference) * jnp.exp(other_terms))
print(f"log space: log L = {log_likelihood:.4f}")
print(f"linear:    L = {linear_likelihood}, log L = {np.log(linear_likelihood)}")
```

```text
log space: log L = -1200.9556
linear:    L = 0.0, log L = -inf
```

## An integral to infinity

The last example has an infinite limit. For a point process with intensity $\lambda(t)$ (methane plumes seen by a satellite, say), the number of events in a window is Poisson with mean $\Lambda(t_0, t_0 + s) = \int_{t_0}^{t_0 + s} \lambda(\tau) \mathrm{d}\tau$, so the probability of no event in the next $s$ hours is $e^{-\Lambda(t_0, t_0 + s)}$, and the expected wait for the next event is

$$\mathbb{E}[\Delta t \mid t_0] = \int_0^\infty \exp(-\Lambda(t_0, t_0 + s)) \mathrm{d}s$$ (eq-expected-wait)

with $\Delta t$ the wait in hours. For a constant rate $\lambda$, $\Lambda = \lambda s$ and the wait is $1/\lambda$. A truncated rule would have to guess where the integrand has decayed, which depends on $\lambda$; the tanh–sinh rule of chapter 8 maps $[0, \infty)$ itself onto a finite grid and needs no guess. We check it on a constant rate of 0.3 h⁻¹ and then use it on a rate with a daily cycle, $\lambda(t) = 0.3 (1 + \frac{1}{2} \sin(2 \pi t / 24))$ h⁻¹ from $t_0 = 6$ h, whose $\Lambda$ is closed form; a Gauss–Kronrod integral to 400 h, where the integrand is below $10^{-50}$, is the reference.

```python
from integrix import adapt

# Shapes: scalars; s in hours
mean_rate, start_hour = 0.3, 6.0


# user
def survival_constant(wait: Float[Array, ""]) -> Float[Array, ""]:
    """e^(−λ s), no event in the next s hours at a constant rate."""
    # e^(−λ s)
    return jnp.exp(-mean_rate * wait)


# user
def integrated_rate(wait: Float[Array, ""]) -> Float[Array, ""]:
    """Λ(t₀, t₀ + s) for λ(t) = λ₀ (1 + ½ sin(2πt/24))."""
    # ω = 2π/24 h⁻¹
    angular = 2.0 * np.pi / 24.0
    # cos ω(t₀ + s) − cos ωt₀
    cycle = jnp.cos(angular * (start_hour + wait)) - np.cos(angular * start_hour)
    # Λ = λ₀ (s − (cos ω(t₀ + s) − cos ωt₀) / 2ω)
    return mean_rate * (wait - cycle / (2.0 * angular))


# user
def survival_daily(wait: Float[Array, ""]) -> Float[Array, ""]:
    """e^(−Λ(t₀, t₀ + s)) for the daily cycle."""
    # e^(−Λ)
    return jnp.exp(-integrated_rate(wait))


# today (integrix)
# 𝔼[Δt] = ∫₀^∞ e^(−λ s) ds = 1/λ
constant = adapt.TanhSinh(rtol=1e-8)(survival_constant, 0.0, jnp.inf)
# today (integrix)
# 𝔼[Δt | t₀] = ∫₀^∞ e^(−Λ) ds
daily = adapt.TanhSinh(rtol=1e-8)(survival_daily, 0.0, jnp.inf)
# today (integrix)
# the reference: ∫₀^400 e^(−Λ) ds, with e^(−Λ(400)) < 10⁻⁵⁰
truncated = adapt.GaussKronrod(rtol=1e-12)(survival_daily, 0.0, 400.0)
print(f"constant rate: wait {float(constant.value):.10f} h, 1/lambda = "
      f"{1.0 / mean_rate:.10f} h, converged {bool(constant.converged)}")
print(f"daily cycle:   wait {float(daily.value):.10f} h, reference "
      f"{float(truncated.value):.10f} h, {int(daily.n_evaluations)} evaluations")
```

```text
constant rate: wait 3.3333333333 h, 1/lambda = 3.3333333333 h, converged True
daily cycle:   wait 2.5665939158 h, reference 2.5665939158 h, 577 evaluations
```

The constant rate gives $1/\lambda$ to every digit printed. The daily cycle starts at 6 h, while the rate is rising towards its peak at 12 h, so the expected wait is shorter than the $1/\lambda = 3.33$ h of the mean rate, and tanh–sinh finds it with no truncation point to choose.

## Gaussian expectations with a closed rule

When $p$ is Gaussian there is a third option: a deterministic rule built for the Gaussian measure. Gauss–Hermite is the Gauss rule of chapter 3 for the weight $e^{-x^2}$, exact for polynomials up to degree $2n - 1$ in each axis, and the unscented transform uses $2 D + 1$ sigma points that match the first two moments. These are global rules again, with the strengths and the $n^D$ cost of chapter 3 (Gauss–Hermite is a tensor rule; the unscented transform avoids the cost by matching moments only). They live in gaussx (`GaussHermiteIntegrator`, `UnscentedIntegrator`), because their natural home is next to the Gaussian solves and filters that use them, and integrix covers the expectations that have no closed rule.
