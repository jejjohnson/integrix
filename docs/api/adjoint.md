# Adjoints

Adjoints through time: the backward sweeps of a forward run. A truncated rollout cuts the gradient through the carry before the last $k$ steps, so reverse mode keeps $O(k)$ of the history, and for chaotic dynamics it regularises, since the exact adjoint grows like $e^{\lambda_{\max} T}$. `to_diffrax_adjoint` resolves pipekit-cycle's adjoint specs, matched structurally, to diffrax adjoints. Adjoints at an optimum (implicit, $k$-step) are argminax's.

Moved from pipekit-jax at `7d02d8e`, whose tests run against them.

::: integrix.adjoint
