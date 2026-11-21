# Semi-Lagrangian advection

Role: ② strategy (`role-ladder`, `references/strategy.md`).
Unit: marches.

Open this when a tracer must be advected with a step above the CFL limit, or when you have a gridded field, a velocity and an interpolation, and want q(t + Δt, x) = q(t, x_d). integrix integrates the trajectory back to the departure point x_d; the read-out at x_d is any functax plan you pass in, so the scheme has no grid of its own.

## Names

All of them live in `integrix.march`.

| Name | What it is |
|---|---|
| `departure_points(nodes, velocity, t, dt, *, iterations=2)` | x_d = x − Δt u(t + Δt/2, (x + x_d)/2) by fixed-point iteration; `iterations=0` is the Euler point x − Δt u(t, x). Second order in Δt for one iteration or more. |
| `semi_lagrangian_step(field, nodes, velocity, readout, t, dt, *, iterations=2)` | One step: `readout(x_d)(field)`, the old field read at the departure points. |
| `SemiLagrangianTerm(velocity)` | The characteristic equation Ẋ = u(t, X) as a diffrax term; its state is positions. |
| `SemiLagrangianSolver(nodes, readout, iterations=2)` | The step as a diffrax solver, for `diffeqsolve` with a `SemiLagrangianTerm` and the field as `y0`; fixed steps only. |

`velocity(t, x)` takes N points `x` of shape (N, D) and returns u at them, shaped the same. `readout(x_d)` returns the map from the old field to its values at x_d; a functax plan, `lambda x_d: BSpline(3).plan(grid, Points(x_d))`, reads a flat field of the grid's size.

## Use it

```python
import diffrax as dfx
import einx
import jax.numpy as jnp
from functax.geometry import Axis, Points
from functax.interp import BSpline

from integrix.march import (
    SemiLagrangianSolver,
    SemiLagrangianTerm,
    departure_points,
    semi_lagrangian_step,
)

# a periodic line of 64 nodes and a uniform flow at CFL 2.5
x = jnp.arange(64) / 64
axis = Axis(x, period=1.0)
nodes = einx.id("node -> node 1", x)
dt = 2.5 / 64


def velocity(t, p):
    return jnp.ones_like(p)


def readout(x_d):
    return BSpline(3).plan(axis, Points(x_d))


q = jnp.sin(2 * jnp.pi * x)
q_new = semi_lagrangian_step(q, nodes, velocity, readout, 0.0, dt)
assert jnp.allclose(q_new, jnp.sin(2 * jnp.pi * (x - dt)), atol=1e-5)

# the departure points alone
assert jnp.allclose(departure_points(nodes, velocity, 0.0, dt), nodes - dt)

# four steps under diffeqsolve
solver = SemiLagrangianSolver(nodes=nodes, readout=readout)
solution = dfx.diffeqsolve(SemiLagrangianTerm(velocity), solver, 0.0, 4 * dt, dt, q)
assert jnp.allclose(solution.ys[0], jnp.sin(2 * jnp.pi * (x - 4 * dt)), atol=1e-4)
```

## Rules

- The read-out sets the spatial accuracy. A linear read-out smears a feature at every step (a Gaussian of width 0.05 on 128 cells keeps 55 % of its peak after 200 steps at CFL 0.37); a cubic B-spline (`BSpline(3)`, SciPy's order-3 spline) keeps 99.7 %. Keys cubic convolution (`Cubic()`) sits between, at 97 %.
- A linear read-out never leaves the old field's range; cubic and B-spline read-outs can over- and undershoot. Use functax's `Pchip` where the tracer must stay monotone.
- Departure points outside the domain are handled by the read-out: a periodic `Axis`, or the plan's `Policies` (clamp, extrapolate, fill).
- Keep `iterations` at 1 or more for a second-order trajectory; 0 reproduces the first-order Euler departure point of gridax-jax's old step.
- `SemiLagrangianSolver` has no error estimate and no field tendency, so pass `dt0` to `diffeqsolve` and use constant steps.
