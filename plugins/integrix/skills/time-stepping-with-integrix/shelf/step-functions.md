# Pure step functions

Role: ① primitive (`role-ladder`, `references/primitive.md`).
Unit: marches.

Open this when you own the loop: a step inside your own `jax.lax.scan`, with forcing, nudging or output between steps, for an autonomous right-hand side f(y).

## Names

All of them live in `integrix.march`. The state is any pytree, and `rhs_fn(y)` takes no time, except in `erk_step`.

| Name | Returns |
|---|---|
| `euler_step(state, rhs_fn, dt)` | yₙ₊₁ |
| `heun_step(state, rhs_fn, dt)` | yₙ₊₁, order 2 |
| `rk3_ssp_step(state, rhs_fn, dt)` | yₙ₊₁, order 3, Shu–Osher form |
| `rk4_step(state, rhs_fn, dt)` | yₙ₊₁, order 4 |
| `erk_step(solver, rhs, y, t, dt)` | yₙ₊₁ of any explicit Runge–Kutta solver's tableau, with `rhs(t, y)` and a `t` and `dt` that may hold one value per element of the leading axes |
| `ab2_step(state, rhs_fn, dt, rhs_nm1)` | `(yₙ₊₁, fₙ, fₙ₋₁)`: the new state and the history for the next step |
| `ab3_step(state, rhs_fn, dt, rhs_nm1, rhs_nm2)` | `(yₙ₊₁, fₙ, fₙ₋₁)` |
| `leapfrog_raf_step(state, state_nm1, rhs_fn, dt, alpha=0.05)` | `(yₙ₊₁, filtered yₙ)` |
| `split_explicit_step(state_3d, state_2d, rhs_3d, rhs_2d, couple_2d_to_3d, dt_slow, n_substeps, slow_forcing=None, t0=0.0)` | `(new_state_3d, new_state_2d)` |

## Use it

```python
import jax
import jax.numpy as jnp

from integrix.march import (
    RK3SSP,
    ab2_step,
    ab3_step,
    erk_step,
    euler_step,
    heun_step,
    leapfrog_raf_step,
    rk3_ssp_step,
    rk4_step,
    split_explicit_step,
)

f = lambda y: -y
y0 = jnp.array(1.0)

assert jnp.allclose(euler_step(y0, f, 0.1), 0.9)
assert jnp.allclose(heun_step(y0, f, 0.1), 0.905)
assert jnp.allclose(rk3_ssp_step(y0, f, 0.1), 1.0 - 0.1 + 0.1**2 / 2 - 0.1**3 / 6)

# any pytree is a state
state = rk4_step({"a": y0}, lambda y: {"a": -y["a"]}, 0.1)
assert jnp.allclose(state["a"], jnp.exp(-0.1), rtol=1e-5)


# your own loop: RK4 with a forcing read from the scanned input
def body(y, forcing):
    y = rk4_step(y, lambda y: -y + forcing, 0.1)
    return y, y


_, trajectory = jax.lax.scan(body, y0, jnp.zeros(10))
assert jnp.allclose(trajectory[-1], jnp.exp(-1.0), rtol=1e-4)

# any explicit tableau, with time, and one step length per element
ys = jnp.ones(3)
steps = jnp.array([0.0, 0.1, 0.2])
moved = erk_step(RK3SSP(), lambda t, y: -y, ys, 0.0, steps)
assert jnp.allclose(
    moved, jnp.array([rk3_ssp_step(1.0, f, h) for h in (0.0, 0.1, 0.2)])
)

# multistep: carry the history yourself, starting from f at the first state
y1, f0, _ = ab2_step(y0, f, 0.1, f(y0))
assert jnp.allclose(y1, 0.9) and jnp.allclose(f0, -1.0)
y1, f0, f_prev = ab3_step(y0, f, 0.1, f(y0), f(y0))
assert jnp.allclose(y1, 0.9)

# leapfrog returns the new state and the filtered middle level
y_next, filtered = leapfrog_raf_step(1.0, 1.0, lambda y: 0.0 * y, 0.1)
assert jnp.allclose(y_next, 1.0) and jnp.allclose(filtered, 1.0)

# one barotropic/baroclinic step: 5 fast substeps inside one slow step
slow, fast = split_explicit_step(
    jnp.array(1.0),
    jnp.array(1.0),
    lambda y3, mean_2d: -mean_2d,
    lambda t, y2, y3: 0.0 * y2,
    lambda y3, mean_2d: y3,
    0.1,
    5,
)
assert jnp.allclose(slow, 0.9) and jnp.allclose(fast, 1.0)
```

## Rules

- The right-hand side is autonomous here, `rhs_fn(y)`, except in `erk_step`, which takes `rhs(t, y)`. For a time-dependent `rhs(t, y, args)` over many steps, use the solvers or steppers with `integrix.march.run`.
- `erk_step` unrolls the tableau, so `jax.jvp` and `jax.grad` both work through it inside a `jax.lax.scan`, where diffrax's `solver.step` supports one mode only. A solver that is not a `diffrax.AbstractERK` raises `ValidationError`.
- Prefer `run` when the loop is only the steps: it carries the history and takes an adjoint option. Reach for a step function when something else happens between steps.
- `ab2_step` and `ab3_step` return the right-hand sides the next step needs; you supply them for the first step.
- `split_explicit_step` subcycles the 2-D state with M RK4 substeps, takes one forward-Euler step of the 3-D state with the mean of the 2-D state over the cycle, then applies `couple_2d_to_3d`. `rhs_2d(t, state_2d, state_3d)` sees the absolute stage time.
- `n_substeps` below 1 raises `ValidationError`.
