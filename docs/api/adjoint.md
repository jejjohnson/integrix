# Adjoints

Adjoints through time: the backward sweeps of a forward run. A truncated rollout cuts the gradient through the carry before the last $k$ steps, so reverse mode keeps $O(k)$ of the history, and for chaotic dynamics it regularises, since the exact adjoint grows like $e^{\lambda_{\max} T}$. A checkpointed rollout stores every $k$-th state on the forward run and recomputes the $k$ steps between two of them on the backward run: memory $O(T / k + k)$, minimised near $k \approx \sqrt{T}$, for about one extra forward pass. `Checkpointed` and `Truncated` are the `adjoint=` options of `integrix.march.run`, and `checkpointed_scan` and `truncated_scan` their function forms for any `jax.lax.scan` rollout. Forward values never depend on the adjoint. `to_diffrax_adjoint` resolves pipekit-cycle's adjoint specs, matched structurally, to diffrax adjoints. Adjoints at an optimum (implicit, $k$-step) are argminax's.

`truncated_scan` and `to_diffrax_adjoint` moved from pipekit-jax at `7d02d8e`, whose tests run against them.

::: integrix.adjoint
