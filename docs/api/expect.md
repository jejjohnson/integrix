# Expectations

An expectation is an integral over a probability space, $\mathbb{E}[f] = \int f(x) p(x) \mathrm{d}x$, and each estimator here returns it with a standard error. `monte_carlo` averages draws $x_s \sim p$, given (a NumPyro posterior, say) or drawn from a sampler, with $\hat\sigma / \sqrt{S}$. `qmc` averages over $R$ randomly shifted copies of a Sobol (Joe–Kuo) or Halton point set, and takes its error from the spread of the shifted estimates. On a smooth integrand that error falls faster than $S^{-1/2}$. `importance` reweights draws from a proposal $q$ by $w_s = p(x_s) / q(x_s)$, self-normalised, so $p$ is needed only up to a constant. It reports the effective sample size $(\sum_s w_s)^2 / \sum_s w_s^2$, because heavy-tailed weights make the estimate unreliable long before its standard error looks large.

integrix averages; sampling from a posterior stays in NumPyro and pyrox.

The theory is in the [Expectations](../expectations/) chapter of the mathematical reference.

For integrands that underflow, two conveniences return log-integrals through `integrix.rule.log_integrate`. `log_gauss_hermite` computes $\log \mathbb{E}[f(Q)]$ for a lognormal $Q$ by Gauss–Hermite quadrature in $\log Q$, and `log_simpson` computes $\log \int_a^b f(q) \mathrm{d}q$ by Simpson's rule on a uniform grid in $\log q$, which resolves a lognormal body and a power-law tail alike.

::: integrix.expect
