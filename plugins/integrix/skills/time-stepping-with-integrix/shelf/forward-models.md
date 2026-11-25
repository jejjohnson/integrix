# A diffrax solve as a forward model

Role: ③ model (`role-ladder`, `references/model.md`).
Unit: marches.

Open this when an assimilation cycle, a filter or a 4D-Var needs a forward model M: x(t₀) ↦ x(t₁), and the dynamics are an ODE or an SDE.

## Names

It lives in `integrix.march`.

| Name | What it is |
|---|---|
| `DiffraxForwardModel(vector_field, solver=…, dt0=…, dt=…, adjoint=…)` | One `diffrax.diffeqsolve` as a model: `step(state, dt)` over [0, dt], `as_dynamics()` with an explicit span, `with_adjoint(spec)` for a copy with another adjoint. |

## Use it

```python
import diffrax as dfx
import jax
import jax.numpy as jnp
import lineax as lx

from integrix.march import DiffraxForwardModel, RK3SSP

x0 = jnp.array([1.0, 2.0])

# an ODE, ẋ = −x, with one of integrix's own schemes
model = DiffraxForwardModel(lambda t, x, args: -x, solver=RK3SSP(), dt0=0.01)
assert jnp.allclose(model.step(x0, 1.0), x0 * jnp.exp(-1.0), atol=1e-5)

# the (state, t0, t1) form that a filter takes
dynamics = model.as_dynamics()
assert jnp.allclose(dynamics(x0, 1.0, 3.0), x0 * jnp.exp(-2.0), atol=1e-5)

# the adjoint is a setting, and a gradient goes through the solve
direct = model.with_adjoint(dfx.DirectAdjoint())
gradient = jax.grad(lambda x: direct.step(x, 1.0).sum())(x0)
assert jnp.allclose(gradient, jnp.exp(-1.0), atol=1e-5)

# an SDE, dx = −x dt + 0.1 dW, with diagonal noise
noisy = DiffraxForwardModel(
    lambda t, x, args: -x,
    solver=dfx.Heun(),
    dt0=0.01,
    diffusion=lambda t, x, args: lx.DiagonalLinearOperator(0.1 * jnp.ones_like(x)),
    sde_key=jax.random.key(0),
)
assert noisy.step(x0, 1.0).shape == (2,)
```

## Rules

- `step` starts every window at t = 0, so a vector field that depends on t needs `as_dynamics`.
- The Brownian path is the same in every call: rebuild the model with another `sde_key` for an independent window.
- `brownian_shape=(N_w,)` is needed when the diffusion is an (N_x, N_w) matrix.
- A parameter that `diffrax.BacksolveAdjoint` differentiates by has to be a leaf of the vector field (an `eqx.Module` field), not a closure.
- It satisfies pipekit-cycle's `ForwardModel` and `SupportsAdjoint` structurally, and its adjoint specs resolve by class name; integrix imports no pipekit package.
