# Adjoints through a march

Role: ② strategy (`role-ladder`, `references/strategy.md`).
Unit: marches.

Open this when you take a gradient through a long rollout and the stored states do not fit in memory, or when you want the gradient of only the last k steps.

## Names

All of them live in `integrix.adjoint`.

| Name | What it is |
|---|---|
| `Checkpointed(every=50)` | Store every k-th state and recompute between: the exact gradient in O(N/k + k) memory. An option for `integrix.march.run`. |
| `Truncated(k=1)` | Gradients flow through the carry of the last k steps only. An option for `run` with an `AbstractStepper`. |
| `checkpointed_scan(f, init, xs, *, every, length=None)` | `jax.lax.scan` that stores the carry only every k steps for reverse mode. |
| `truncated_scan(f, init, xs, *, k=1, length=None)` | `jax.lax.scan` with the carry's gradient stopped before the last k steps. |
| `to_diffrax_adjoint(spec)` | Resolves an adjoint spec, matched by class name, or a diffrax adjoint, to a `diffrax.AbstractAdjoint`. |

## Use it

```python
import diffrax as dfx
import jax
import jax.numpy as jnp

from integrix.adjoint import (
    Checkpointed,
    Truncated,
    checkpointed_scan,
    to_diffrax_adjoint,
    truncated_scan,
)
from integrix.march import AB2Solver, RK4Classic, run

rhs = lambda t, y, rate: -rate * y


def loss(rate, stepper, adjoint):
    return run(stepper, rhs, 0.0, jnp.array(1.0), 0.01, 100, args=rate, adjoint=adjoint)


# checkpointing changes the memory, not the gradient
plain = jax.grad(loss)(1.0, AB2Solver(), None)
checkpointed = jax.grad(loss)(1.0, AB2Solver(), Checkpointed(every=10))
assert jnp.allclose(checkpointed, plain, rtol=1e-5)
assert jnp.allclose(plain, -jnp.exp(-1.0), rtol=1e-2)

# a diffrax solver takes Checkpointed too
solver_grad = jax.grad(loss)(1.0, RK4Classic(), Checkpointed(every=10))
assert jnp.allclose(solver_grad, -jnp.exp(-1.0), rtol=1e-3)

# truncation keeps the forward value and cuts the gradient
assert jnp.allclose(
    loss(1.0, AB2Solver(), Truncated(k=5)), loss(1.0, AB2Solver(), None)
)
short = jax.grad(loss)(1.0, AB2Solver(), Truncated(k=5))
assert abs(float(short)) < abs(float(plain))


# the scan forms, for a rollout that is not a march
def step(carry, x):
    new = jnp.sin(carry) + x
    return new, new


xs = jnp.linspace(0.0, 1.0, 10)
_, plain_ys = jax.lax.scan(step, 0.5, xs)
_, ys = checkpointed_scan(step, 0.5, xs, every=3)
assert jnp.allclose(ys, plain_ys)
_, ys = truncated_scan(step, 0.5, xs, k=2)
assert jnp.allclose(ys, plain_ys)


# a spec from another library, matched by its class name
class RecursiveCheckpointAdjoint:
    checkpoints = 8


adjoint = to_diffrax_adjoint(RecursiveCheckpointAdjoint())
assert isinstance(adjoint, dfx.RecursiveCheckpointAdjoint) and adjoint.checkpoints == 8
```

## Rules

- Forward values never depend on the adjoint.
- `Checkpointed(every=k)` with k ≈ √N minimises the memory. For a diffrax solver it becomes `diffrax.RecursiveCheckpointAdjoint(checkpoints=⌈N / k⌉)`.
- `Truncated` is for an `AbstractStepper` only, and a diffrax adjoint is for a diffrax solver only; the other pairing raises `ValidationError`. `Truncated(k)` with k ≥ N is the exact adjoint.
- A truncated gradient is biased: it drops the dependence through earlier steps. Each step's own output still receives gradients from that step's inputs.
- `to_diffrax_adjoint` accepts `RecursiveCheckpointAdjoint`, `DirectAdjoint`, `BacksolveAdjoint` and `ImplicitAdjoint` specs and passes a diffrax adjoint through. A `TruncatedAdjoint` spec raises: truncation belongs to the rollout, not to one solve.
- A continuous (backsolve) adjoint diverges for chaotic or stiff dynamics, so it is never a default.
- Derivatives at an optimum or a fixed point (implicit, k-step) are argminax's; backward sweeps through a forward run are integrix's.
