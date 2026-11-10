---
numbering:
  enumerator: "9.%s"
---

# Adjoints

## The gradient of a forecast misfit

4D-Var finds the initial state of a forecast that best fits the observations over a window. The cost it minimises is a sum of misfits over the window, and every misfit depends on the initial state through the model's time steps, so its gradient has to travel back through every step of the march. The same question comes up in a plume inversion, where we want the sensitivity of an inferred emission rate to the wind that carried the plume, and in a retrieval, where we want the sensitivity of a column to the methane at each level.

These are all gradients of integrals, and they all come down to one fact: the derivative of a weighted sum with respect to its samples is the weights. In this chapter we follow that fact through the three operations. A rule's adjoint is its weights, a scan's adjoint is a reversed scan, and a march's adjoint is a backward march, whose memory is the practical problem that [`Checkpointed`](xref:api#integrix.adjoint.Checkpointed) and [`Truncated`](xref:api#integrix.adjoint.Truncated) solve.

## A rule's adjoint is its weights

The rule [](#eq-rule) is linear in the samples, so

$$\frac{\partial \hat{I}_N}{\partial f_n} = w_n$$ (eq-rule-adjoint)

and the vector-Jacobian product of a rule with a cotangent $I^\dagger$ is $I^\dagger \vect{w}$. This is why the pressure weighting function of chapter 2 is also the column's sensitivity: the derivative of $\mathrm{XCH_4}$ with respect to the methane at level $n$ is $h_n$. JAX gets it from integrix without a custom rule.

```python
import jax

jax.config.update("jax_enable_x64", True)

import diffrax as dfx
import jax.numpy as jnp
from integrix import adapt, adjoint, march, rule

# Shapes: N_P = 6 pressure levels

pressure = jnp.linspace(0.0, 1013.0, 6)  # → (N_P,)
widths = jnp.diff(pressure)
zero = jnp.zeros(1)
# trapezoid weights, wₙ = (pₙ₊₁ − pₙ₋₁)/2
weights = 0.5 * (jnp.concatenate([widths, zero]) + jnp.concatenate([zero, widths]))
methane = 1600.0 + 300.0 * pressure / 1013.0  # → (N_P,)


def column_average(profile):
    """XCH₄ = (1/p_s) ∑ₙ wₙ c(pₙ)."""
    # user
    # today (integrix)
    return rule.integrate(weights, profile).value / pressure[-1]


# ∂XCH₄/∂cₙ = wₙ / p_s = hₙ
sensitivity = jax.grad(column_average)(methane)  # → (N_P,)
print("dXCH4/dc =", " ".join(f"{value:.3f}" for value in sensitivity))
print("h        =", " ".join(f"{value:.3f}" for value in weights / pressure[-1]))
```

```text
dXCH4/dc = 0.100 0.200 0.200 0.200 0.200 0.100
h        = 0.100 0.200 0.200 0.200 0.200 0.100
```

The limits of an integral have an adjoint too. The Leibniz rule gives $\frac{\mathrm{d}}{\mathrm{d}b} \int_a^b f(s) \mathrm{d}s = f(b)$ and $\frac{\mathrm{d}}{\mathrm{d}a} \int_a^b f(s) \mathrm{d}s = -f(a)$, and an adaptive rule differentiates its final rule on its final partition (chapter 8), which includes the limits.

```python
# today (integrix)
# d/db ∫₀ᵇ eˢ ds = eᵇ
upper_derivative = jax.grad(lambda b: adapt.GaussKronrod()(jnp.exp, 0.0, b).value)(1.5)
print(f"d/db = {float(upper_derivative):.12f}, exp(b) = {float(jnp.exp(1.5)):.12f}")
```

```text
d/db = 4.481689070338, exp(b) = 4.481689070338
```

## A scan's adjoint is a reversed scan

The running integral [](#eq-scan) is a prefix sum of increments, $F_k = \sum_{j \le k} \Delta_j$, so $\partial F_k / \partial \Delta_j$ is 1 when $j \le k$ and 0 otherwise. A cotangent $F^\dagger_k$ on each entry therefore gives the increments the cotangent

$$\Delta^\dagger_j = \sum_{k \ge j} F^\dagger_k$$ (eq-scan-adjoint)

which is a prefix sum taken from the far end: a reversed scan. The adjoint of a forward scan is a reversed scan, and the adjoint of a reversed scan (an optical depth from the top) is a forward one. This is local and global once more. Each increment touches two neighbouring samples, and the cotangent of each increment collects every later entry of the scan.

## A march's adjoint is a backward march

A march applies one step map $\vect{x}_{k+1} = \Phi(\vect{x}_k)$ per step. Its gradient composes the Jacobian of each step, $\mat{J}_{\Phi,k} = \partial \Phi / \partial \vect{x}$ at $\vect{x}_k$, from the last step back to the first. For a cost $\mathcal{J}$ that sums a misfit $\mathcal{J}_k(\vect{x}_k)$ at every step, the adjoint state $\vect{x}^\dagger_k \in \mathbb{R}^{D_x}$ runs backwards as

$$\vect{x}^\dagger_k = \mat{J}_{\Phi,k}^\top \vect{x}^\dagger_{k+1} + \nabla_{\vect{x}_k} \mathcal{J}_k, \quad \nabla_{\vect{x}_0} \mathcal{J} = \vect{x}^\dagger_0$$ (eq-march-adjoint)

with $\vect{x}^\dagger_K = \nabla_{\vect{x}_K} \mathcal{J}_K$ at the last step $K$. Each step's Jacobian is local in time, since it only knows its own step, and their product is the global sensitivity of the cost to the initial state. That is the gradient 4D-Var needs.

The catch is memory. Each $\mat{J}_{\Phi,k}$ is evaluated at $\vect{x}_k$, so plain reverse mode stores all $K$ states on the way forward. For an ocean model with $10^7$ unknowns and a few thousand steps that does not fit. Checkpointing (Griewank and Walther, 2008) stores every $B$-th state only (`Checkpointed(every=B)`) and recomputes the $B$ states of one block at a time on the way back, so it holds $K/B$ stored states plus $B$ recomputed ones,

$$\text{memory} \propto \frac{K}{B} + B, \quad \text{smallest at } B = \sqrt{K}$$ (eq-checkpoint-memory)

at the cost of one extra forward pass. We count the states that reverse mode stores with JAX's own accounting (`saved_residuals`, an internal JAX helper, and the same count as integrix's test of [`checkpointed_scan`](xref:api#integrix.adjoint.checkpointed_scan)) for $K = 400$ steps of an 8-dimensional state, and add the $k$ states that one block recomputes.

```python
import math

from jax._src.ad_checkpoint import saved_residuals

# Shapes: D_x = 8; K = 400 steps
n_steps = 400
initial = jnp.ones(8)  # → (D_x,)


def step(state, _):
    """xₖ₊₁ = 1.01 sin(xₖ)."""
    # user
    return jnp.sin(state) * 1.01, None


def stored_states(scan_fn):
    """The number of D_x-sized states reverse mode keeps from the forward pass."""

    # user
    def final_sum(state):
        return jnp.sum(scan_fn(step, state, None)[0])

    residuals = saved_residuals(final_sum, initial)
    return sum(math.prod(aval.shape) for aval, _ in residuals) // initial.size


plain = stored_states(lambda f, c, x: jax.lax.scan(f, c, x, length=n_steps))
print(f"plain reverse mode: {plain} stored")
for every in (5, 10, 20, 40, 80):
    # today (integrix)
    stored = stored_states(
        lambda f, c, x: adjoint.checkpointed_scan(f, c, x, every=every, length=n_steps)
    )
    print(
        f"B = {every:2d}: {stored:3d} stored + {every:2d} recomputed = {stored + every}"
    )
```

```text
plain reverse mode: 400 stored
B =  5:  80 stored +  5 recomputed = 85
B = 10:  40 stored + 10 recomputed = 50
B = 20:  20 stored + 20 recomputed = 40
B = 40:  10 stored + 40 recomputed = 50
B = 80:   5 stored + 80 recomputed = 85
```

The stored count is exactly $K/B$, and the peak is smallest at $B = 20 = \sqrt{400}$, where it is 40 states instead of 400.

## 4D-Var on an inertial oscillation

Now the whole thing on a toy 4D-Var problem. A surface current with velocity $(u, v)$ in m s⁻¹ turns under the Coriolis parameter $f_\text{cor} = 10^{-4}$ s⁻¹ and decays with a damping rate $r$, so $\dot{u} = f_\text{cor} v - r u$ and $\dot{v} = -f_\text{cor} u - r v$ (with time in hours, $f_\text{cor} = 0.36$ h⁻¹ and $r = 0.01$ h⁻¹). We observe $u$ every half hour for 400 steps, from a true initial current $(0.2, 0)$, and take the gradient of the misfit $\mathcal{J}(\vect{x}_0) = \frac{1}{2} \sum_k (u_k - y_k)^2$ at a first guess $\vect{x}_0 = (0.1, 0.05)$, with an Adams–Bashforth stepper and each adjoint in turn.

```python
# Shapes: D_x = 2 (u, v); K = 400 half-hour steps
coriolis, damping = 0.36, 0.01


def tendency(t, state, args):
    """u̇ = f_cor v − r u, v̇ = −f_cor u − r v, per hour."""
    # user
    u, v = state[0], state[1]
    return jnp.stack([coriolis * v - damping * u, -coriolis * u - damping * v])


every_step = dfx.SaveAt(steps=True)
# today (integrix)
truth = march.run(
    march.AB2Solver(), tendency, 0.0, jnp.array([0.2, 0.0]), 0.5, 400, saveat=every_step
)
observations = truth[:, 0]  # (K, D_x) → (K,)


def misfit(initial_state, adjoint_option):
    """𝒥(x₀) = ½ ∑ₖ (uₖ − yₖ)²."""
    # user
    # today (integrix)
    trajectory = march.run(
        march.AB2Solver(),
        tendency,
        0.0,
        initial_state,
        0.5,
        400,
        saveat=every_step,
        adjoint=adjoint_option,
    )
    return 0.5 * jnp.sum((trajectory[:, 0] - observations) ** 2)


first_guess = jnp.array([0.1, 0.05])
options = {
    "plain": None,
    "Checkpointed(every=20)": adjoint.Checkpointed(every=20),
    "Truncated(k=20)": adjoint.Truncated(k=20),
}
for name, option in options.items():
    gradient = jax.grad(misfit)(first_guess, option)
    print(f"{name:23s} grad = ({float(gradient[0]):+.6f}, {float(gradient[1]):+.6f})")
```

```text
plain                   grad = (-4.863018, +2.347982)
Checkpointed(every=20)  grad = (-4.863018, +2.347982)
Truncated(k=20)         grad = (-0.087042, -0.015927)
```

Checkpointing changes the memory and nothing else: its gradient is the plain one to every digit printed. Truncation is a different object. [`Truncated`](xref:api#integrix.adjoint.Truncated) cuts the gradient through the state before the last $k_\text{trunc}$ steps (its `k`), so the early misfits no longer reach the initial state through the dynamics, and the gradient here is a small fraction of the true one. That is not a bug. Truncation shortens the gradient path rather than the memory (reverse mode still stores every step), and it is used on purpose when training through long or chaotic rollouts, where the full gradient explodes and a short horizon acts as a regulariser.
