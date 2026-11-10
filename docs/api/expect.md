# Expectations

An expectation is an integral over a probability space, $\mathbb{E}[f] = \int f(x) p(x) \mathrm{d}x$, and each estimator here returns it with a standard error. `monte_carlo` averages draws $x_s \sim p$, given (a NumPyro posterior, say) or drawn from a sampler, with $\hat\sigma / \sqrt{S}$. `qmc` averages over $R$ randomly shifted copies of a Sobol (Joe–Kuo) or Halton point set, and takes its error from the spread of the shifted estimates. On a smooth integrand that error falls faster than $S^{-1/2}$. `importance` reweights draws from a proposal $q$ by $w_s = p(x_s) / q(x_s)$, self-normalised, so $p$ is needed only up to a constant. It reports the effective sample size $(\sum_s w_s)^2 / \sum_s w_s^2$, because heavy-tailed weights make the estimate unreliable long before its standard error looks large.

integrix averages; sampling from a posterior stays in NumPyro and pyrox.

::: integrix.expect
