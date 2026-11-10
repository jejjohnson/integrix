# Adaptive quadrature

Adaptive quadrature computes $I = \int_a^b f(x) \mathrm{d}x$ to a tolerance, with an error estimate. On each subinterval the Kronrod rule $K_{2n+1}$ gives the value and its embedded Gauss rule $G_n$ the error, $E \approx \lvert K_{2n+1}(f) - G_n(f) \rvert$, for no extra evaluations; the subinterval with the largest error is bisected until $E \le \max(\text{atol}, \text{rtol} \lvert I \rvert)$ or the static buffer of subintervals is full. The search runs inside `jax.lax.while_loop` with static shapes, so it traces under `jit` and `vmap`, and `converged` is returned rather than raised. Reverse mode cannot run through that loop, so the derivative is that of the final rule on the final partition, held fixed (a custom JVP), in the integrand's parameters and in the limits.

The nodes and weights are QUADPACK's (Piessens et al., 1983). The design follows quadax, with no dependency on it; quadax and `scipy.integrate.quad` are test references only.

::: integrix.adapt
