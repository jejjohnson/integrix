# The march and its history steppers

Role: ④ algorithm (`role-ladder`, `references/algorithm.md`).
Unit: marches.

Open this when you step a state through N constant steps, need every state along the way, or use a scheme that remembers earlier steps.

## Names

All of them live in `integrix.march`.

| Name | What it is |
|---|---|
| `run(stepper, rhs, t0, y0, dt, n_steps, *, args=None, saveat=None, adjoint=None)` | N constant steps of h from t₀. Returns y_N, or with `saveat=diffrax.SaveAt(steps=True)` the states y₁, …, y_N stacked on a new leading axis. |
| `AbstractStepper` | The base class of a fixed-step scheme whose history lives in the loop carry: `order`, `init(rhs, t0, y0, dt, args)` and `step(rhs, t, y, dt, state, args)`. |
| `AB2Solver()` | Second-order Adams–Bashforth. |
| `LeapfrogRAFSolver(alpha=0.05)` | Leapfrog with a Robert–Asselin filter of coefficient α, typically 0.01 to 0.1. |
| `SplitExplicitRKSolver(n_substeps=50)` | Split-explicit stepping: M fast substeps per slow step, for a barotropic and a baroclinic mode. |
| `SplitExplicitTerms(slow, fast, slow_forcing=None, couple=None)` | The vector fields a split-explicit march takes as `rhs`. |

## Use it

```python
import diffrax as dfx
import jax
import jax.numpy as jnp

from integrix.march import (
    AB2Solver,
    AbstractStepper,
    LeapfrogRAFSolver,
    RK4Classic,
    SplitExplicitRKSolver,
    SplitExplicitTerms,
    run,
)


# the vector field, written once: decay at a rate passed through args
def rhs(t, y, rate):
    return -rate * y


y0 = jnp.ones(3)

# a diffrax solver and a history stepper go through the same call
final = run(RK4Classic(), rhs, 0.0, y0, 0.1, 10, args=1.0)
assert jnp.allclose(final, jnp.exp(-1.0), rtol=1e-4)
ab2 = run(AB2Solver(), rhs, 0.0, y0, 0.01, 100, args=1.0)
assert jnp.allclose(ab2, jnp.exp(-1.0), rtol=1e-3)
leapfrog = run(LeapfrogRAFSolver(alpha=0.05), rhs, 0.0, y0, 0.001, 1000, args=1.0)
assert jnp.allclose(leapfrog, jnp.exp(-1.0), rtol=1e-2)

# every state, not only the last
every = run(AB2Solver(), rhs, 0.0, y0, 0.1, 5, args=1.0, saveat=dfx.SaveAt(steps=True))
assert every.shape == (5, 3)

# jit once around the whole march, and differentiate through it
decay = jax.jit(lambda rate: run(AB2Solver(), rhs, 0.0, y0, 0.01, 100, args=rate).sum())
assert jnp.allclose(jax.grad(decay)(1.0), -3.0 * jnp.exp(-1.0), rtol=1e-2)

# split-explicit: the state is a (slow, fast) pair
terms = SplitExplicitTerms(
    slow=lambda t, y3, mean_2d, args: -mean_2d,
    fast=lambda t, y2, y3, args: 0.0 * y2,
)
slow, fast = run(
    SplitExplicitRKSolver(n_substeps=5),
    terms,
    0.0,
    (jnp.array(1.0), jnp.array(1.0)),
    0.1,
    10,
)
assert jnp.allclose(slow, 0.0, atol=1e-5) and jnp.allclose(fast, 1.0)


# your own scheme: init takes the first step, step every later one
class Euler(AbstractStepper):
    @property
    def order(self):
        return 1

    def init(self, rhs, t0, y0, dt, args):
        return None, y0 + dt * rhs(t0, y0, args)

    def step(self, rhs, t, y, dt, state, args):
        return y + dt * rhs(t, y, args), state


assert jnp.allclose(run(Euler(), rhs, 0.0, y0, 0.1, 10, args=1.0), 0.9**10)
```

## Rules

- `rhs(t, y, args)` is the diffrax vector field, so a model writes its tendencies once for both kinds of stepper. A diffrax solver may take a diffrax term instead, such as a `diffrax.MultiTerm` for IMEX.
- `n_steps` is a Python integer, at least 1. The state `y0` is any pytree.
- `saveat` is `None` or `SaveAt(steps=True)`; anything else raises `ValidationError`.
- A diffrax solver goes to `diffrax.diffeqsolve` on the step times t₀ + n h, so it takes exactly N steps. An `AbstractStepper` runs in `jax.lax.scan` with its history in the carry.
- `init` returns `(state, y₁)` and `step` returns `(y_next, state)`; neither mutates the stepper.
- `SplitExplicitTerms`: `slow(t, y_3d, mean_2d, args)`, `fast(t, y_2d, y_3d, args)`, and the optional `slow_forcing(t, y_3d, args)` and `couple(y_3d, mean_2d, args)`.
- `LeapfrogRAFSolver.alpha` is an array leaf, so it can be traced or differentiated; the filter makes the scheme first order.
- For the memory of a gradient through a long march, see `adjoints.md`.
