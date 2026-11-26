# IMEX schemes and the stable step

Role: ② strategy (`role-ladder`, `references/strategy.md`).
Unit: marches.

Open this when one term is stiff (vertical diffusion, a fast linear wave, a relaxation) and you want to take it implicitly beside an explicit one, or when you need a first bound on an explicit step.

## Names

All of them live in `integrix.march`.

| Name | What it is |
|---|---|
| `IMEX_SSP2(root_finder=...)` | IMEX-SSP2(2,2,2) as a diffrax solver for a `diffrax.MultiTerm` of an explicit and an implicit term. |
| `imex_ssp2_step(state, rhs_explicit, rhs_implicit, implicit_solve, dt)` | One IMEX-SSP2 step with your own implicit solve, `implicit_solve(r, gamma_dt)` returning Y with Y − γh f_I(Y) = r. |
| `imex_solver(*, rtol=1e-4, atol=1e-6, gmres_restart=20)` | A `diffrax.KenCarp3` whose implicit stage is matrix-free (Newton with GMRES). |
| `imex_stepsize_controller(*, rtol=1e-4, atol=1e-6, step_ts=None)` | An adaptive `diffrax.PIDController` for an IMEX solve, stepping exactly to the times `step_ts` when they are given. |
| `count_steps(span, step)` | K = ⌈T/Δt⌉ on the host, with a quotient within 10⁻⁹ K of a whole number taken as it, so 2.1 s in 0.15 s is 14 steps; 0 for T = 0. |
| `step_lengths(span, step)` | The K step lengths in float64: Δt, and T − (K − 1) Δt for the last. |
| `stable_step(spacing, *, speed=(), diffusivity=(), cfl=1.0)` | The largest stable forward-Euler step for advection and diffusion together: CFL / (∑ᵢ \|uᵢ\| / Δxᵢ + 2 ∑ᵢ Kᵢ / Δxᵢ²). |

## Use it

```python
import diffrax as dfx
import jax.numpy as jnp

from integrix.march import (
    IMEX_SSP2,
    count_steps,
    imex_solver,
    imex_ssp2_step,
    imex_stepsize_controller,
    stable_step,
    step_lengths,
)

# y′ = −y − 50 y: the slow part explicit, the stiff part implicit
terms = dfx.MultiTerm(
    dfx.ODETerm(lambda t, y, args: -y),
    dfx.ODETerm(lambda t, y, args: -50.0 * y),
)

# an adaptive, matrix-free solve
solution = dfx.diffeqsolve(
    terms,
    imex_solver(rtol=1e-5, atol=1e-7),
    0.0,
    1.0,
    0.01,
    jnp.ones(4),
    stepsize_controller=imex_stepsize_controller(rtol=1e-5, atol=1e-7),
)
assert jnp.allclose(solution.ys[0], jnp.exp(-51.0), atol=1e-4)

# fixed steps with IMEX-SSP2
solver = IMEX_SSP2(root_finder=dfx.VeryChord(rtol=1e-8, atol=1e-8))
fixed = dfx.diffeqsolve(terms, solver, 0.0, 0.1, 0.001, jnp.array(1.0))
assert jnp.allclose(fixed.ys[0], jnp.exp(-5.1), rtol=1e-2)

# one step with your own implicit solve: here the stiff part is linear, −10 y
y = imex_ssp2_step(
    jnp.array(1.0),
    lambda y: -y,
    lambda y: -10.0 * y,
    lambda r, gamma_dt: r / (1.0 + 10.0 * gamma_dt),
    0.01,
)
assert jnp.allclose(y, jnp.exp(-0.11), rtol=1e-3)

# the explicit bound: advection and diffusion add
assert jnp.allclose(stable_step((0.5, 1.0), speed=(1.0, 0.0)), 0.5)
assert jnp.allclose(stable_step((1.0,), speed=(1.0,), diffusivity=(0.5,)), 0.5)
assert bool(jnp.isinf(stable_step((1.0, 1.0))))  # nothing moves or diffuses

# the steps of a span: no step of zero length, and a shorter last one
assert count_steps(2.1, 0.15) == 14
assert step_lengths(12.0, 5.0).tolist() == [5.0, 5.0, 2.0]
```

## Rules

- Order the `MultiTerm` as (explicit, implicit).
- `count_steps` and `step_lengths` take the span T, not two times: count times from an epoch near the run, because the difference of two Unix-epoch times has already rounded.
- Give `imex_ssp2_step` the solve the structure allows, such as a tridiagonal solve along columns for vertical diffusion, instead of a general Newton iteration.
- `stable_step` adds the advective and diffusive rates. Taking the smaller of two separate bounds would allow up to twice the step.
- `stable_step` is a forward-Euler estimate that catches a step orders of magnitude too large. It is not a stability certificate for a high-order reconstruction or a Runge–Kutta scheme.
- `stable_step` validates Python and NumPy inputs when called and JAX arrays with `equinox.error_if`, so the checks hold under `jax.jit`.
