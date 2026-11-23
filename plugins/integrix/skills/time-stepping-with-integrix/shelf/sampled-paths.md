# Sampled forcing and controls

Role: ② strategy (`role-ladder`, `references/strategy.md`).
Unit: marches.

Open this when a forcing (reanalysis winds, river discharge, emission rates) or the control of a CDE is known only at sample times, and a solver must read it between them.

## Names

It lives in `integrix.march`.

| Name | What it is |
|---|---|
| `SampledPath.fit(scheme, ts, us, *, policies=None)` | A functax fit of the samples, time first, as a `diffrax.AbstractPath`: `evaluate(t)`, `evaluate(t0, t1)` for an increment, `derivative(t, left=True)`, and `ts` for the step control. |

## Use it

```python
import diffrax as dfx
import jax.numpy as jnp
from functax.interp import Linear, Pchip, Policies

from integrix.march import RK2Heun, RK3SSP, SampledPath, imex_stepsize_controller

# discharge at uneven times: PCHIP stays between neighbouring samples
ts = jnp.array([0.0, 0.5, 2.0, 3.0])
discharge = jnp.array([0.0, 3.0, 0.0, 0.0])
river = SampledPath.fit(Pchip(), ts, discharge)
assert float(jnp.min(jnp.stack([river.evaluate(t) for t in (0.2, 1.3, 2.6)]))) >= 0.0

# ẋ = −x + û(t), adaptive, with the steps landing on the samples
solution = dfx.diffeqsolve(
    dfx.ODETerm(lambda t, x, args: -x + river.evaluate(t)),
    RK2Heun(),
    0.0,
    3.0,
    0.1,
    jnp.array(0.0),
    stepsize_controller=imex_stepsize_controller(step_ts=river.ts),
)
assert bool(jnp.isfinite(solution.ys[0]))

# dx = dX(t): a CDE returns the increment of its control
xs = jnp.stack([ts, ts**2], axis=-1)  # (4, 2)
policies = Policies(outside="extrapolate")
control = SampledPath.fit(Linear(), ts, xs, policies=policies)
driven = dfx.diffeqsolve(
    dfx.ControlTerm(lambda t, x, args: jnp.eye(2), control),
    RK3SSP(),
    0.0,
    3.0,
    0.1,
    jnp.zeros(2),
)
assert jnp.allclose(driven.ys[0], control.evaluate(0.0, 3.0), atol=1e-5)
```

## Rules

- Let the steps land on `path.ts`: `imex_stepsize_controller(step_ts=path.ts)`, or a fixed step that divides the sample spacing. A step that straddles a sample loses order; RK4 on a linear reconstruction drops to second order.
- `Linear` and `Pchip` stay between neighbouring samples, so non-negative data stays non-negative. `CubicSpline` is C² and may overshoot.
- Past the samples the default holds the end values. `Policies(outside="extrapolate")` continues the end interval, as diffrax's own paths do.
- Samples are time first, `(N_t, *channel)`, and may be differentiated: the fit runs in JAX.
- On uneven times use `Linear` or a series scheme; `Cubic` and `BSpline` refuse them.
- Build the path outside `jit` when the times are fixed, so that functax can check their spacing.
