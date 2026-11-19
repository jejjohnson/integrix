# Single-step schemes as diffrax solvers

Role: ② strategy (`role-ladder`, `references/strategy.md`).
Unit: marches.

Open this when you choose an explicit scheme, or need a scheme diffrax does not ship that still works with `diffrax.diffeqsolve`, its step-size controllers and its adjoints.

## Names

All of them live in `integrix.march`, and each subclasses a diffrax solver.

| Name | Order | What it is |
|---|---|---|
| `ForwardEulerDfx()` | 1 | Forward Euler, yₙ₊₁ = yₙ + h F(tₙ, yₙ). |
| `RK2Heun()` | 2 | Heun's method, with an embedded forward-Euler error estimate. |
| `SSP_RK2()` | 2 | Second-order strong-stability-preserving Runge–Kutta, the same scheme as `RK2Heun`. |
| `RK3SSP()` | 3 | Third-order strong-stability-preserving Runge–Kutta (Shu–Osher). |
| `RK4Classic()` | 4 | The classical fourth-order Runge–Kutta method, fixed steps only. |
| `SSP_RK104()` | 4 | Ketcheson's ten-stage, fourth-order SSP Runge–Kutta, fixed steps only. |
| `AB3Solver()` | 3 | Third-order Adams–Bashforth with an SSP-RK3 start; its history lives in the solver state. |

## Use it

```python
import diffrax as dfx
import jax.numpy as jnp

from integrix.march import (
    AB3Solver,
    ForwardEulerDfx,
    RK2Heun,
    RK3SSP,
    RK4Classic,
    SSP_RK2,
    SSP_RK104,
    run,
)

term = dfx.ODETerm(lambda t, y, args: -y)
y0 = jnp.array(1.0)


def solve(solver, dt):
    return dfx.diffeqsolve(term, solver, t0=0.0, t1=1.0, dt0=dt, y0=y0).ys[0]


# first order: exactly ten Euler steps
assert jnp.allclose(solve(ForwardEulerDfx(), 0.1), 0.9**10, rtol=1e-5)

# the higher orders reach e⁻¹ with the same step
exact = jnp.exp(-1.0)
for solver in (RK2Heun(), SSP_RK2(), RK3SSP(), AB3Solver()):
    assert jnp.allclose(solve(solver, 0.01), exact, rtol=1e-3)
for solver in (RK4Classic(), SSP_RK104()):
    assert jnp.allclose(solve(solver, 0.1), exact, rtol=1e-4)

assert [s.order(None) for s in (RK2Heun(), RK3SSP(), RK4Classic(), SSP_RK104())] == [
    2,
    3,
    4,
    4,
]

# the same solver through integrix's march: a number of steps, not an end time
assert jnp.allclose(
    run(RK3SSP(), lambda t, y, args: -y, 0.0, y0, 0.01, 100), exact, rtol=1e-3
)

# an adaptive step from Heun's embedded error estimate
adaptive = dfx.diffeqsolve(
    term,
    RK2Heun(),
    t0=0.0,
    t1=1.0,
    dt0=0.1,
    y0=y0,
    stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-7),
)
assert jnp.allclose(adaptive.ys[0], exact, rtol=1e-3)
```

## Rules

- Use `diffrax.diffeqsolve` when you want an end time, a step-size controller, dense output or events, and `integrix.march.run` when you want exactly N constant steps.
- `RK4Classic` and `SSP_RK104` have no embedded error estimate: fixed steps only.
- Choose an SSP scheme (`SSP_RK2`, `RK3SSP`, `SSP_RK104`) for advection with a limiter or a WENO reconstruction, where a new extremum must not appear.
- The schemes that carry history between steps outside the solver state (AB2, leapfrog, split-explicit) are not diffrax solvers: see `run-and-steppers.md`.
- integrix pins diffrax's minor version, because these classes follow its solver protocol.
