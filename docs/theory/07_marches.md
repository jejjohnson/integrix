---
numbering:
  enumerator: "7.%s"
---

# Marches

## An ocean model stepping forward

An ocean model advances its state (velocities, temperature, salinity, sea surface height) by a fixed time step, many thousands of times. A typical setup uses several schemes at once: a strong-stability-preserving Runge–Kutta method for the advection of tracers, an implicit treatment for vertical diffusion (which is stiff on a fine vertical grid), a leapfrog step with a Robert–Asselin filter in older codes, and fast barotropic substeps inside each slow baroclinic step. These look like a different subject from the global mean of chapter 5, and most libraries keep them apart.

They are not a different subject. Each step of each scheme integrates the model's tendency over one time step, and each does it with a quadrature rule in time. In this chapter we read the steppers in `integrix.march` that way: their nodes and weights, their exactness, and the local and global errors that come with them. Stability, which has no counterpart in plain quadrature, is the other half of a stepper and closes the chapter.

## An ODE is an integral equation

Let the state $\vect{x}(t) \in \mathbb{R}^{D_x}$ evolve as $\dot{\vect{x}} = f(\vect{x}, t)$, where $f: \mathbb{R}^{D_x} \times \mathbb{R} \to \mathbb{R}^{D_x}$ is the tendency (the right-hand side), and take steps of size $h$ between times $t_k = t_0 + k h$. Integrating both sides over one step gives

$$\vect{x}(t_{k+1}) = \vect{x}(t_k) + \int_{t_k}^{t_{k+1}} f(\vect{x}(t), t) \mathrm{d}t$$ (eq-ode-integral)

which is exact, so every one-step method is a way of approximating the integral on the right with samples of $f$. This is why $f$ keeps one letter across this section (see [Notation](notation.md)): the integrand of a march is the tendency. The difference from chapter 5 is that the integrand depends on the unknown $\vect{x}(t)$ inside the step, so the samples have to be predicted before they can be weighted.

## Runge–Kutta methods as quadrature

An explicit Runge–Kutta method with $N_\text{stage}$ stages predicts the state at stage times $t_k + c_i h$, samples the tendency there and sums the samples with weights $b_i$,

$$\vect{k}_i = f\Big(\vect{x}_k + h \sum_{j<i} a_{ij} \vect{k}_j, t_k + c_i h\Big), \quad \vect{x}_{k+1} = \vect{x}_k + h \sum_{i=1}^{N_\text{stage}} b_i \vect{k}_i$$ (eq-runge-kutta)

where $\vect{k}_i \in \mathbb{R}^{D_x}$ are the stage tendencies and $a_{ij}$, $b_i$, $c_i$ the Butcher coefficients. If $f$ does not depend on $\vect{x}$, the stages are just samples of $f$ in time and [](#eq-runge-kutta) becomes

$$\int_{t_k}^{t_k + h} f(t) \mathrm{d}t \approx h \sum_{i=1}^{N_\text{stage}} b_i f(t_k + c_i h)$$ (eq-rk-quadrature)

which is a rule on $[0, 1]$ with nodes $c_i$ and weights $b_i$, scaled by $h$. So every Runge–Kutta method carries a quadrature rule, and [](#tbl-rk-rules) lists the ones in `integrix.march`.

:::{table} The quadrature rule inside each explicit Runge–Kutta stepper of `integrix.march`: its nodes $c_i$ and weights $b_i$ on $[0, 1]$, and the order $q$ of the full method.
:label: tbl-rk-rules

| Stepper | Nodes $c_i$ | Weights $b_i$ | Rule | Order $q$ |
| --- | --- | --- | --- | --- |
| `ForwardEulerDfx` | $0$ | $1$ | left rectangle | 1 |
| `RK2Heun`, `SSP_RK2` | $0, 1$ | $\frac{1}{2}, \frac{1}{2}$ | trapezoid | 2 |
| `RK3SSP` | $0, 1, \frac{1}{2}$ | $\frac{1}{6}, \frac{1}{6}, \frac{2}{3}$ | Simpson | 3 |
| `RK4Classic` | $0, \frac{1}{2}, \frac{1}{2}, 1$ | $\frac{1}{6}, \frac{1}{3}, \frac{1}{3}, \frac{1}{6}$ | Simpson, midpoint sampled twice | 4 |
| `SSP_RK104` | ten stages (Ketcheson, 2008) | | | 4 |
:::

The order conditions of a Runge–Kutta method start with exactly these quadrature conditions, $\sum_i b_i c_i^m = 1/(m + 1)$ for $m = 0, \ldots, q - 1$ (the moment equations of [](#eq-moments) on $[0, 1]$), and add conditions on the $a_{ij}$ for the coupling through $\vect{x}$ (Hairer et al., 1993). So a method of order $q$ needs a rule of degree at least $q - 1$. We can measure each stepper's degree directly, by taking one step of $h = 1$ on $\dot{x} = t^m$ from $x(0) = 0$, where the exact answer is $1/(m + 1)$.

```python
import jax

jax.config.update("jax_enable_x64", True)

from collections.abc import Callable
from typing import Any

import diffrax as dfx
import einx
import jax.numpy as jnp
import lineax as lx
import numpy as np
from integrix import march
from jaxtyping import Array, Float

# Shapes: D_x = 1, a scalar state


# user
def one_step(
    solver: dfx.AbstractSolver,
    tendency: Callable[..., Any],
    state: Float[Array, ""],
    step: float,
) -> Float[Array, ""]:
    """One step of h from t = 0 with a diffrax solver."""
    # today (diffrax)
    # ẋ = f(x, t)
    term = dfx.ODETerm(tendency)
    # today (diffrax)
    # x(h) = x(0) + ∫₀ʰ f(x(t), t) dt, one step
    solution = dfx.diffeqsolve(
        term, solver, t0=0.0, t1=step, dt0=step, y0=state, max_steps=2
    )
    return solution.ys[0]


# user
def degree_of_exactness(solver: dfx.AbstractSolver) -> int:
    """The largest m for which one step integrates tᵐ over [0, 1] exactly."""
    # no degree is exact yet
    degree = -1
    for power in range(8):
        # ẋ = tᵐ
        monomial = lambda t, x, args: t**power
        # x(1) = 1/(m + 1)
        estimate = one_step(solver, monomial, jnp.asarray(0.0), 1.0)  # → ()
        # e = |x̂(1) − 1/(m + 1)|
        if abs(float(estimate) - 1.0 / (power + 1)) > 1e-12:
            break
        degree = power
    return degree


# the five steppers whose rules we compare
steppers = {
    # today (integrix)
    "ForwardEulerDfx": march.ForwardEulerDfx(),
    # today (integrix)
    "RK2Heun": march.RK2Heun(),
    # today (integrix)
    "RK3SSP": march.RK3SSP(),
    # today (integrix)
    "RK4Classic": march.RK4Classic(),
    # today (integrix)
    "SSP_RK104": march.SSP_RK104(),
}
for name, solver in steppers.items():
    print(f"{name:16s} degree {degree_of_exactness(solver)}")
```

```text
ForwardEulerDfx  degree 0
RK2Heun          degree 1
RK3SSP           degree 3
RK4Classic       degree 3
SSP_RK104        degree 3
```

Euler is the left rectangle and integrates only constants. Heun is the trapezoid rule, degree 1. SSP-RK3 and RK4 both sample the step at $0$, $\frac{1}{2}$ and $1$ with Simpson's weights, so both have degree 3, even though one is a third-order method and the other fourth-order: the extra order of RK4 comes from its $a_{ij}$, not its rule. Ketcheson's ten-stage method spends its extra stages on a larger strong-stability-preserving step instead of a higher degree. Going the other way, choosing the $c_i$ as Gauss–Legendre nodes gives the implicit Gauss collocation methods, of order $2 N_\text{stage}$; integrix does not ship them, but they are the global rule of chapter 3 used as a stepper.

## Adams–Bashforth: interpolate the past

A multistep method reuses tendencies it has already computed instead of predicting new stages. Adams–Bashforth interpolates the last few tendencies $f_k, f_{k-1}, \ldots$ with a polynomial in time and integrates that polynomial over the next step, which is the interpolatory construction of chapter 2 with nodes at the past times. In units of $h$, the nodes are $0, -1, -2, \ldots$, the interval is $[0, 1]$, and the weights solve the moment equations [](#eq-moments) with $b_m = \int_0^1 t^m \mathrm{d}t = 1/(m + 1)$.

```python
# Shapes: N = 2 or 3 past times, the nodes of the rule
for n_past, scale in ((2, 2), (3, 12)):
    # n = 0, …, N − 1
    ordinal = jnp.arange(float(n_past))                         # → (N,)
    # tₙ = −n, the past times (in units of h)
    past_times = -ordinal                                       # → (N,)
    # Vₙₘ = tₙᵐ
    vander = jnp.vander(past_times, increasing=True)            # → (N, N)
    # [V]ₘₙ = tₙᵐ
    vandermonde = einx.id("n m -> m n", vander)                 # → (N, N)
    # m + 1 = 1, …, N
    powers = jnp.arange(1.0, n_past + 1.0)                      # → (N,)
    # bₘ = ∫₀¹ tᵐ dt = 1/(m + 1)
    moments = 1.0 / powers                                      # → (N,)
    # today (lineax)
    # V = [V]ₘₙ as an operator
    operator = lx.MatrixLinearOperator(vandermonde)             # (N, N) operator
    # today (lineax)
    # V w = b
    weights = lx.linear_solve(operator, moments).value          # → (N,)
    print(f"AB{n_past}: {jnp.round(weights * scale, 10).tolist()} / {scale}")
```

```text
AB2: [3.0, -1.0] / 2
AB3: [23.0, -16.0, 5.0] / 12
```

These are the coefficients of `ab2_step` and `ab3_step`. The rule extrapolates outside its nodes, so it has negative weights, and chapter 2's warning about negative weights shows up here as a small stability region: AB3 needs a smaller step than RK3 for the same advection problem. The leapfrog step, $\vect{x}_{k+1} = \vect{x}_{k-1} + 2 h f(\vect{x}_k)$, is the midpoint rule over two steps, and the split-explicit step of `split_explicit_step` drives the slow state with the trapezoidal mean of the fast barotropic substeps, which is a rule over the slow step too.

## Local and global error

A stepper makes an error on every step, and these local errors add up over the run, like the interval errors of a scan in chapter 6. If one step from the exact state has a local truncation error $O(h^{q+1})$, then after $T / h$ steps to a fixed end time $T$ the global error is

$$\Vert \vect{x}_K - \vect{x}(T) \Vert = O(h^q), \quad K = T / h$$ (eq-global-error)

one power of $h$ lower, since there are $O(1/h)$ steps and the dynamics carry each local error forward without blowing it up (for a stable problem). The order $q$ of a method is its global order, and it is what integrix's order test measures: `tests/march/test_order.py` runs each stepper on the decay problem $\dot{x} = \gamma x$ with $\gamma = -1$ to $T = 1$ and fits the slope of $\log$ error against $\log h$. We run the same problem with the same step sizes and measure both errors.

```python
# Shapes: D_x = 1; K = T/h steps; N_h = 4 step sizes h


# user
def decay(
    t: float, state: Float[Array, ""], args: Any
) -> Float[Array, ""]:
    """ẋ = γ x with γ = −1."""
    # ẋ = γ x
    return -state


# user
def final_state(solver: dfx.AbstractSolver, step: float) -> Float[Array, ""]:
    """x(T) at T = 1 with constant steps h."""
    # K = T/h
    n_steps = round(1.0 / step)
    # today (diffrax)
    # ẋ = γ x
    term = dfx.ODETerm(decay)
    # today (diffrax)
    # x(T) = x(0) + ∫₀ᵀ γ x dt, in K steps of h
    solution = dfx.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=1.0,
        dt0=step,
        y0=jnp.asarray(1.0),
        max_steps=n_steps + 1,
    )
    return solution.ys[0]


cases = {
    # today (integrix)
    "RK2Heun": (march.RK2Heun(), np.array([0.04, 0.02, 0.01, 0.005])),
    # today (integrix)
    "RK3SSP": (march.RK3SSP(), np.array([0.04, 0.02, 0.01, 0.005])),
    # today (integrix)
    "RK4Classic": (march.RK4Classic(), np.array([0.1, 0.05, 0.025, 0.0125])),
}
for name, (solver, steps) in cases.items():
    # local: one step from x(0) = 1 against exp(γh)
    # ε_loc(h) = |x̂₁ − exp(−h)|
    local = [
        abs(float(one_step(solver, decay, jnp.asarray(1.0), h)) - np.exp(-h))
        for h in steps
    ]
    # global: K = T/h steps against exp(γT)
    # ε_glob(h) = |x̂_K − exp(−1)|
    global_ = [abs(float(final_state(solver, h)) - np.exp(-1.0)) for h in steps]
    # q_loc = slope of log ε_loc against log h
    local_slope = np.polyfit(np.log(steps), np.log(local), 1)[0]
    # q_glob = slope of log ε_glob against log h
    global_slope = np.polyfit(np.log(steps), np.log(global_), 1)[0]
    print(f"{name:11s} local slope {local_slope:.2f}, global slope {global_slope:.2f}")
```

```text
RK2Heun     local slope 3.00, global slope 2.01
RK3SSP      local slope 4.00, global slope 3.01
RK4Classic  local slope 4.99, global slope 4.03
```

Each method loses exactly one power of $h$ between one step and the whole run, which is the relation in [](#eq-global-error), and the global slopes are the orders in [](#tbl-rk-rules). It is the same global and local bookkeeping as in chapter 3, now in time: the error is made locally, one step at a time, and judged globally, at the end. Chapter 8 uses the local error of each step to choose the step size against a global tolerance.

## Stability: the other half

Quadrature has no counterpart to stability, because a rule's samples do not feed back into the integrand. A stepper's do. On the test problem $\dot{x} = \gamma x$, one step multiplies the state by a stability function $R(\zeta)$ of $\zeta = h \gamma$, which for an explicit Runge–Kutta method is a polynomial: $R(\zeta) = 1 + \zeta$ for Euler, and the Taylor series of $e^\zeta$ up to $\zeta^4$ for RK4. The step is stable when $|R(h \gamma)| \le 1$ for every eigenvalue $\gamma$ of the linearised tendency, so stability limits $h$ even when accuracy would allow a larger step.

The global and local thread shows up once more. For advection on a grid, each cell has its own limit, $h \le \Delta s / |u|$ for a speed $u$ and spacing $\Delta s$ (the CFL condition), and diffusion with diffusivity $K$ adds $h \le \Delta s^2 / 2K$. These are local conditions, cell by cell, but the model takes one global step, so the most restrictive cell sets the step for the whole domain. [`stable_step`](xref:api#integrix.march.stable_step) computes that bound. An ocean column with 10 km horizontal cells and 1 m s⁻¹ currents allows a step of about three hours, but adding vertical diffusion of $10^{-2}$ m² s⁻¹ on a 1 m vertical grid cuts it to under a minute.

```python
# Shapes: 2 axes (horizontal, vertical), one cell size, speed and diffusivity each
# today (integrix)
# h_max = 1 / (∑ᵢ |uᵢ| / Δsᵢ + 2 ∑ᵢ Kᵢ / Δsᵢ²)
advection_only = march.stable_step((10_000.0,), speed=(1.0,))   # → ()
# today (integrix)
# h_max = 1 / (∑ᵢ |uᵢ| / Δsᵢ + 2 ∑ᵢ Kᵢ / Δsᵢ²)
with_diffusion = march.stable_step(                             # → ()
    (10_000.0, 1.0), speed=(1.0, 0.0), diffusivity=(0.0, 1e-2)
)
print(f"advection only:            h_max = {float(advection_only):.0f} s")
print(f"with vertical diffusion:   h_max = {float(with_diffusion):.1f} s")
```

```text
advection only:            h_max = 10000 s
with vertical diffusion:   h_max = 49.8 s
```

This is why the vertical diffusion goes into the implicit half of an IMEX scheme. `IMEX_SSP2` treats the explicit term with an SSP Runge–Kutta tableau and the stiff term with an A-stable implicit one ($|R(\zeta)| \le 1$ for every $\zeta$ with a negative real part), so the step is set by the advection alone. Three more stability ideas complete the set in `integrix.march`. SSP methods keep a property of forward Euler, such as no new extrema in a tracer, under a step-size bound. The leapfrog step has a second, computational mode that flips sign every step, which the Robert–Asselin filter damps at the cost of first-order accuracy. The split-explicit step subcycles the fast mode with small steps so that the slow mode can take large ones.

## Local stencils and global transforms in space

Finally, the tendency $f$ itself sits on one side of chapter 3 or the other. In a finite-volume model it is a local stencil: each cell's tendency reads a few neighbours, and its Jacobian is sparse. In a spectral model it is a global transform: each tendency goes through spherical harmonics or Fourier modes, and every point depends on every other. integrix does not care which, since the steppers only call $f$, so the same `RK3SSP` can step a finite-volume tendency or a spectral one.
