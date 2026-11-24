---
name: time-stepping-with-integrix
description: Step ODEs and discretised PDEs in time in JAX with integrix — forward Euler, Heun, SSP-RK2, SSP-RK3, classical RK4, SSP-RK(10,4) and AB3 as diffrax solvers, AB2, filtered leapfrog and split-explicit steppers that carry history, IMEX-SSP2 and a matrix-free IMEX solve with step control, the explicit CFL bound, pure step functions, one loop for all of them, and checkpointed or truncated adjoints through a long rollout. Use when a task writes a time loop, a Runge–Kutta or multistep scheme, an implicit-explicit split, a stable-step estimate, or a gradient through many steps, in a project that uses (or could use) integrix.
---

# Time stepping with integrix

A march steps a state whose tendency depends on the state: yₙ₊₁ = Φ(yₙ, tₙ, h).
integrix's single-step schemes are diffrax solvers, so `diffrax.diffeqsolve`, its controllers and its adjoints run them. The schemes that carry history between steps are `AbstractStepper`s, and `integrix.march.run` runs both kinds.

This skill covers the strategies of integrix's time stepping, with the primitives they are built from and the algorithm that runs them, in the vocabulary of the `role-ladder` skill.
Integrals that do not step in time are in `integrix:quadrature-and-expectations-with-integrix`.

## The contract

- `run(stepper, rhs, t0, y0, dt, n_steps)` takes N constant steps and returns y_N; `saveat=diffrax.SaveAt(steps=True)` returns y₁, …, y_N stacked.
- `rhs(t, y, args)` is the diffrax vector field; the state `y` is any pytree; `n_steps` is a Python integer.
- `adjoint=` sets how reverse mode runs back through the march; forward values never depend on it.
- A step function, such as `rk4_step(state, rhs_fn, dt)`, is one step for an autonomous `rhs_fn(y)`, for a loop you write yourself.
- The library never jits: jit once around the whole march.

## Which scheme

| Problem | Choose |
|---|---|
| A smooth ODE, fixed step | `RK4Classic()` |
| Advection with a limiter or WENO | `RK3SSP()`, or `SSP_RK104()` for a larger stable step per stage |
| One tendency evaluation per step | `AB2Solver()` or `AB3Solver()` |
| A leapfrog model with a time filter | `LeapfrogRAFSolver(alpha)` |
| Fast barotropic and slow baroclinic modes | `SplitExplicitRKSolver(n_substeps)` |
| A stiff term beside a non-stiff one, fixed step | `IMEX_SSP2()` |
| A stiff term, adaptive step, no Jacobian matrix | `imex_solver()` with `imex_stepsize_controller()` |
| An adaptive explicit step | `RK2Heun()` with a diffrax controller |
| A forcing or control known at sample times | `SampledPath.fit(scheme, ts, us)`, beside any of the above |

## Worked example

```python
import diffrax as dfx
import jax
import jax.numpy as jnp

from integrix.adjoint import Checkpointed
from integrix.march import RK3SSP, run, stable_step

# upwind advection of a tracer on a periodic 1-D grid
n, length, speed = 64, 1.0, 1.0
dx = length / n
x = jnp.arange(n) * dx
tracer0 = jnp.exp(-0.5 * ((x - 0.5) / 0.05) ** 2)


def tendency(t, tracer, speed):
    return -speed * (tracer - jnp.roll(tracer, 1)) / dx


# a step inside the explicit bound
dt = stable_step((dx,), speed=(speed,), cfl=0.5)
assert jnp.allclose(dt, 0.5 * dx)
n_steps = 64


@jax.jit
def forecast(tracer, speed):
    return run(RK3SSP(), tendency, 0.0, tracer, dt, n_steps, args=speed)


final = forecast(tracer0, speed)
assert jnp.allclose(final.sum(), tracer0.sum(), rtol=1e-4)  # the tracer is conserved
assert float(final.max()) <= float(tracer0.max()) + 1e-5  # and gains no new maximum

# every state, for output
states = run(
    RK3SSP(),
    tendency,
    0.0,
    tracer0,
    dt,
    n_steps,
    args=speed,
    saveat=dfx.SaveAt(steps=True),
)
assert states.shape == (n_steps, n)


# the gradient of a misfit with respect to the initial state, with checkpoints
def misfit(tracer):
    end = run(
        RK3SSP(),
        tendency,
        0.0,
        tracer,
        dt,
        n_steps,
        args=speed,
        adjoint=Checkpointed(every=8),
    )
    return 0.5 * jnp.sum((end - final) ** 2)


gradient = jax.grad(misfit)(jnp.zeros(n))
assert gradient.shape == (n,) and bool(jnp.isfinite(gradient).all())
```

## Don't write it

| Don't write | Use |
|---|---|
| A Python `for` loop over time steps | `run(stepper, rhs, t0, y0, dt, n_steps)` |
| Runge–Kutta stages written out in a `jax.lax.scan` body | a solver from `integrix.march`, or its `*_step` function |
| The previous tendencies carried by hand for Adams–Bashforth | `AB2Solver()` or `AB3Solver()` |
| `min(dx / u, dx**2 / (2 * K))` | `stable_step(spacing, speed=..., diffusivity=...)` |
| A dense Jacobian for the implicit part | `imex_solver()`, or `imex_ssp2_step` with a structured solve |
| `jax.checkpoint` wrapped around blocks of a scan | `Checkpointed(every=k)` or `checkpointed_scan` |
| `jax.lax.stop_gradient` on a carry to shorten backpropagation | `Truncated(k=k)` or `truncated_scan` |

## The shelf

Each entry is one file with the names it covers, a worked example and the rules.
Open the one the task needs; do not read them all.

<!-- shelf:begin -->
| Role | Entry | Covers | Names |
|---|---|---|---|
| ④ algorithm | [The march and its history steppers](shelf/run-and-steppers.md) | N constant steps of any stepper, the base class of a scheme that carries history, and the three that ship: AB2, filtered leapfrog and split-explicit | 6 |
| ② strategy | [Single-step schemes as diffrax solvers](shelf/diffrax-solvers.md) | forward Euler, Heun, the SSP Runge–Kutta schemes, classical RK4 and AB3, each a diffrax solver for diffeqsolve or for the march | 7 |
| ② strategy | [IMEX schemes and the stable step](shelf/imex-and-step-control.md) | a stiff term taken implicitly beside an explicit one, a matrix-free implicit stage, its step-size controller and the explicit CFL bound | 5 |
| ② strategy | [Semi-Lagrangian advection](shelf/semi-lagrangian.md) | a tracer advected past the CFL limit by reading the old field at departure points, with a midpoint trajectory and any functax read-out | 4 |
| ② strategy | [Sampled forcing and controls](shelf/sampled-paths.md) | forcing or control data known at sample times, read between them through any functax 1-D scheme as a diffrax path | 1 |
| ① primitive | [Pure step functions](shelf/step-functions.md) | one step of each explicit scheme for an autonomous right-hand side, to put inside your own scan | 8 |
| ② strategy | [Adjoints through a march](shelf/adjoints.md) | the exact gradient in less memory by checkpointing, a gradient cut to the last k steps, their scan forms, and the bridge to a diffrax adjoint | 5 |
<!-- shelf:end -->

## Before you finish

- Is the tendency written once as `rhs(t, y, args)`, with parameters in `args`?
- Is the step inside `stable_step`, or controlled by a step-size controller?
- Is the whole march jitted once, not each step?
- For a gradient through many steps, is `adjoint=` set, and paired with the right kind of stepper?

<!-- skill-tree:begin · generated by geostack scripts/skill_tree.py; do not edit by hand -->
## Builds on

Each card below says what a library under this skill guarantees.
A card is enough to call it.
Load a skill only when its card's last line applies.

### functax · base dependency · 1 of 8 units

#### Interpolation · ① ②, setup · skill `functax:interpolation-with-functax`

- **Provides.** Interpolation as a scheme frozen into a plan: f̂(X*) = H 𝒮(y), with the adjoint 𝒮ᵀ Hᵀ written by hand.
- **Nouns.** Schemes `functax.interp.Nearest`, `functax.interp.Linear`, `functax.interp.Cubic`, `functax.interp.BSpline`, `functax.interp.Tensor`, `functax.interp.Conservative`, `functax.interp.Binning`, `functax.interp.RBF`, `functax.interp.Kriging`; the result is a `functax.interp.Plan`.
- **Verbs.** `plan = scheme.plan(source, target)`, then `plan(field)` and `plan.adjoint(residual)`; `plan.operator()` is a lineax operator, `plan.derivative(axis)` reads a derivative, `scheme.fit(source, data)` gives a field read at targets chosen inside the trace.
- **Adverbs.** `policies=functax.interp.Policies(outside=..., missing=..., boundary=..., extent=...)`; `plan.valid` marks the targets the plan could read.
- **Keep.** Build a plan once, outside `jit`, and call it on many fields; a field is flattened to `(..., N)` in the source's row-major order.
- **Never.** Write a `searchsorted`, a gather with hand-made weights or a scatter-add for the adjoint; never differentiate through a plan's construction when `plan.adjoint` is what you need.
- **One rung down.** Open `functax:interpolation-with-functax` to choose a scheme, set the policies, regrid conservatively, krige or write a new scheme.

### gaussx · base dependency · 0 of 7 units

It uses fewer than 3 names of each unit, so no card. The front door is `gaussx:structured-gaussian-linalg`.

Used by (pointers, not skills to load): finitevolX, pipekit, plumax, sesmax, vardax.

<!-- skill-tree:end -->
