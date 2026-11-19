# Quickstart

This page picks up where the [README](https://github.com/jejjohnson/integrix#readme) stops. We work the README's two examples through in full, to see why their numbers come out as they do. Then we show the calling convention every procedure shares and a first example of each module the README only lists: an adaptive rule to a tolerance, a gradient through a long march with a checkpointed adjoint, and an expectation with its standard error. The notation is the README's and the [mathematical reference](../theory/notation.md)'s, and every number below is printed by the block above it.

## One integral, three operations

Every integral integrix computes is

$$I[f] = \int_\Omega f(\vect{s}) \mathrm{d}\mu(\vect{s})$$ (eq-quickstart-integral)

where $\vect{s} \in \Omega \subset \mathbb{R}^{D_s}$ is a point of the domain (a height, a wavelength, a time, a pixel or a draw from a distribution), $\mu$ is a measure on $\Omega$ and $f: \Omega \to \mathbb{R}^{D_f}$ is the integrand, batched over $D_f$ entries. A rule replaces it with a weighted sum over $N$ nodes $\vect{s}_n$, with weights $w_n$ and samples $f_n = f(\vect{s}_n)$,

$$\hat{I}_N[f] = \sum_{n=1}^{N} w_n f(\vect{s}_n)$$ (eq-quickstart-rule)

and the operations differ in what they keep of that sum and where its samples come from. With $\vect{x}_k \in \mathbb{R}^{D_x}$ the state of a model at time $t_k$, $E_N[f] = I[f] - \hat{I}_N[f]$ the error of a rule and $p$ a probability density on $\Omega$:

:::{table} The three operations on one integral, and the three modules that apply across them.
:label: tbl-quickstart-operations

| Operation | Equation | Module | For |
|---|---|---|---|
| Sum | $\hat{I}_N[f] = \sum_n w_n f_n$ | `integrix.rule` | a band radiance, a column amount, the mass of a plume |
| Scan | $F_k = F_{k-1} + \frac{1}{2} (s_k - s_{k-1}) (f_{k-1} + f_k)$ | `integrix.scan` | the integral up to every node: optical depth, cumulative mass |
| March | $\vect{x}_{k+1} = \vect{x}_k + \int_{t_k}^{t_{k+1}} f(\vect{x}(t), t) \mathrm{d}t$ | `integrix.march` | an integrand that depends on the solution, as in time stepping |
| Adapt | $\vert E_N[f] \vert \le \max(\epsilon_\text{abs}, \epsilon_\text{rel} \vert \hat{I}_N[f] \vert)$ | `integrix.adapt` | a rule or a scan refined to a tolerance |
| Differentiate | reverse mode, checkpointed or truncated | `integrix.adjoint` | gradients through a long scan or march |
| Expect | $\mathbb{E}_p[f] = \int_\Omega f(\vect{s}) p(\vect{s}) \mathrm{d}\vect{s}$ | `integrix.expect` | an integral against a probability, by sampling |
:::

## Who owns the weights

functax owns rules: a rule is data, its nodes and weights. integrix owns what is done with them (summing, scanning, adapting, marching and differentiating), and Gaussian expectations with a closed rule (Gauss–Hermite, unscented, sigma points) stay in gaussx. So the fixed rules below come from `functax.quad`, handed to integrix as they are.

## A column: a rule, a scan and a gradient

The README's first example is an absorbing gas whose extinction falls off with height $z$ as $\beta(z; H) = \beta_0 e^{-z/H}$, with $\beta_0 = 0.1$ km⁻¹ at the surface and the scale height $H = 8$ km, sampled at $N_z = 121$ levels $z_n$ every $h = 0.5$ km up to $z_\text{top} = 60$ km. The rule gives the optical depth of the whole column, the reversed scan the optical depth from the top down to every level, and `jax.grad` through the rule the sensitivity to the scale height:

$$\tau(z_n) = \int_{z_n}^{z_\text{top}} \beta(z; H) \mathrm{d}z$$ (eq-quickstart-column)

so the optical depth of the whole column is $\tau_0(H) = \tau(0)$, and the transmittance of the path from the top of the column to $z_n$ is $\mathcal{T}(z_n) = e^{-\tau(z_n)}$.

```python
import jax
import jax.numpy as jnp
from functax import quad

from integrix import rule, scan


# The checks against the closed forms below need double precision.
jax.config.update("jax_enable_x64", True)

# Shapes: N_z = 121 levels, every 0.5 km from the surface to z_top = 60 km

# z, the heights of the levels in km
height = jnp.linspace(0.0, 60.0, 121)  # → (N_z,)
# today (functax)
# the trapezoid rule on the levels: nodes zₙ and weights wₙ
levels = quad.trapezoid(height)
# β₀ = 0.1 km⁻¹ and H = 8 km
surface_extinction = 0.1
scale_height = 8.0


# user
# β(z; H) = β₀ exp(−z / H)
def extinction(scale_height):
    return surface_extinction * jnp.exp(-height / scale_height)  # → (N_z,)


# user
# τ₀(H) = ∫₀^z_top β(z; H) dz ≈ ∑ₙ wₙ β(zₙ; H)
def column_optical_depth(scale_height):
    # today (integrix)
    return rule.integrate(levels, extinction(scale_height)).value  # → ()


column = column_optical_depth(scale_height)  # → ()
# today (integrix)
# τ(zₙ) = ∫_zₙ^z_top β(z; H) dz, summed from the top down
optical_depth = scan.cumulative(  # → (N_z,)
    levels, extinction(scale_height), reverse=True
)
# 𝒯(zₙ) = exp(−τ(zₙ))
transmittance = jnp.exp(-optical_depth)  # → (N_z,)
# today (jax)
# ∂τ₀/∂H, through the rule
sensitivity = jax.grad(column_optical_depth)(scale_height)  # → ()

print(f"{column:.5f} {transmittance[0]:.4f} {sensitivity:.5f}")
```

```text
0.79982 0.4494 0.09950
```

The closed forms are $\tau_0 = \beta_0 H (1 - e^{-z_\text{top}/H}) = 0.79956$ and $\partial \tau_0 / \partial H = \beta_0 (1 - (1 + z_\text{top}/H) e^{-z_\text{top}/H}) = 0.09953$ per km, so the rule is $3.3 \times 10^{-4}$ high and its derivative $3.3 \times 10^{-4}$ low. Both come from the trapezoid rule's error. For a smooth integrand the Euler–Maclaurin formula gives $\hat{I}_N - I \approx \frac{h^2}{12} (f'(b) - f'(a))$ ([chapter 3](../theory/03_global_and_local.md)), and here $\beta'(0) = -\beta_0 / H$ while $\beta'(z_\text{top})$ is nearly 0, so the rule overestimates by $h^2 \beta_0 / (12 H)$:

```python
# ε = 1 − τ₀ / τ̂₀, the relative error of the rule
measured = 1.0 - (
    surface_extinction * scale_height * (1.0 - jnp.exp(-60.0 / scale_height))
) / column                                                          # → ()
# ε ≈ h² β₀ / (12 H τ₀), the Euler–Maclaurin estimate with h = 0.5 km
predicted = 0.5**2 * surface_extinction / (12.0 * scale_height * column)  # → ()
print(f"{measured:.3e} {predicted:.3e}")
```

```text
3.254e-04 3.256e-04
```

The error falls as $h^2$, so halving the spacing quarters it.

## The calling convention

[`integrate`](xref:api#integrix.rule.integrate) is `integrate(rule, values_or_fn, *, axis=None, args=None, where=None, nan_policy="propagate")`, and the other procedures of `integrix.rule` and `integrix.scan` follow it:

- **`rule`** is a functax `Rule` (anything with `nodes` and `weights`), a tuple of 1-D rules (a tensor rule, one per axis) or a plain weights array (a spectral response, the cell areas of an image).
- **`values_or_fn`** is the samples, or a callable evaluated at the rule's nodes (with `args`, as `f(nodes, args)`), which returns the node axes first.
- **`axis`** names the axes to integrate over, one per weights axis; `None` takes the last ones, so a batch of columns is integrated level by level with no argument at all.
- **`where`** is a boolean mask that broadcasts against the samples: a sample where it is false contributes 0, and the weights are not rebuilt for the shorter domain.
- **`nan_policy="omit"`** leaves NaN samples out as `np.nansum` does, but traceably and with a zero gradient on them; the default `"propagate"` returns NaN.

The result is an [`Integral`](xref:api#integrix.rule.Integral) with `value`, `error`, `n_evaluations` and `converged`; a fixed rule has no error estimate, so its `error` is `None`.

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from functax import quad
from integrix import rule

# Shapes: N_c = 3 columns; N_z = 5 levels
# zₙ, the heights of the levels in km
height = jnp.array([0.0, 1.0, 2.0, 4.0, 8.0])                       # → (N_z,)
# today (functax)
# the trapezoid rule on the levels: wₙ = (0.5, 1, 1.5, 3, 2) km
levels = quad.trapezoid(height)
# q, a tracer on the levels of three columns; the third has a missing value
tracer = jnp.array(                                                 # → (N_c, N_z)
    [
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [5.0, 4.0, 3.0, 2.0, 1.0],
        [1.0, jnp.nan, 1.0, 1.0, 1.0],
    ]
)
# the levels each column has: the second stops below 8 km
valid = jnp.array(                                                  # → (N_c, N_z)
    [
        [True, True, True, True, True],
        [True, True, True, True, False],
        [True, True, True, True, True],
    ]
)
# today (integrix)
# Î = ∑ₙ wₙ qₙ over the last axis, batched over the columns
total = rule.integrate(levels, tracer)                              # → (N_c,)
# today (integrix)
# Î = ∑ over kept n of wₙ qₙ
kept = rule.integrate(levels, tracer, where=valid, nan_policy="omit")  # → (N_c,)
# today (integrix)
# Î = ∑ₙ wₙ f(zₙ), with f evaluated at the rule's nodes
decay = rule.integrate(levels, lambda z: jnp.exp(-z / 8.0))         # → ()
print(total.value, kept.value, f"{decay.value:.4f}", total.error)
```

```text
[ 8. 19. nan] [ 8. 17.  7.] 5.1060 None
```

The second column loses its top level and with it the top level's weight, 2 km × 1, so 19 becomes 17 rather than the integral of a column that ends at 4 km. The third keeps its four finite samples. The callable form gives 5.1060 against the exact $8 (1 - e^{-1}) = 5.0570$, since five levels are far too few for this rule.

## Precision

integrix never casts silently: a rule sums in the dtype of its samples, pairwise, and the caller chooses the precision by choosing the dtype of the data. We turn on JAX's 64-bit mode here because the examples compare against closed forms to many digits. A model whose state is float32 keeps float32 integrals, and [chapter 11](../theory/11_precision.md) says what that costs: the error grows with the condition number of the sum, so cancel large terms of both signs before integrating rather than after.

## An ocean column: a march and a rule

The README's second example warms a 200 m ocean column, with a 20 °C mixed layer over a thermocline, for ten days. The temperature $T(z, t)$ in °C at depth $z$ follows $\partial T / \partial t = \kappa \partial^2 T / \partial z^2$ with $\kappa = 10^{-4}$ m² s⁻¹, the surface takes in $Q = 100$ W m⁻² and the bottom nothing, and the heat content of the column is

$$E(t) = \rho c_p \int_0^{200} T(z, t) \mathrm{d}z$$ (eq-quickstart-heat)

in J m⁻², with the volumetric heat capacity of sea water $\rho c_p = 1025 \times 3990$ J m⁻³ K⁻¹.

```python
import jax
import jax.numpy as jnp
from functax import quad

from integrix import march, rule


# The heat budget below closes to round-off in double precision.
jax.config.update("jax_enable_x64", True)

# Shapes: N_z = 41 levels, every 5 m from the surface to 200 m; N_t = 1,440 steps

# z, the depths of the levels in m, and their spacing Δz
depth = jnp.linspace(0.0, 200.0, 41)  # → (N_z,)
spacing = depth[1] - depth[0]  # → ()
# today (functax)
# the trapezoid rule on the levels
levels = quad.trapezoid(depth)
# κ in m² s⁻¹, Q in W m⁻² and ρ c_p in J m⁻³ K⁻¹
diffusivity = 1e-4
heating = 100.0
heat_capacity = 1025.0 * 3990.0
# T(z, 0) in °C: a 20 °C mixed layer over a thermocline down to 10 °C
temperature = 10.0 + 10.0 / (1.0 + jnp.exp((depth - 50.0) / 10.0))  # → (N_z,)


# user
# ∂T/∂t = κ ∂²T/∂z², plus Q / (ρ c_p) into the top half-cell
def tendency(t, temperature, args):
    # mirrored ends: no flux through the surface or the bottom but Q
    padded = jnp.pad(temperature, 1, mode="reflect")  # (N_z,) → (N_z + 2,)
    # ∂²T/∂z² ≈ (Tₙ₊₁ − 2Tₙ + Tₙ₋₁) / Δz²
    curvature = (padded[2:] - 2.0 * temperature + padded[:-2]) / spacing**2
    # 1 / (Δz / 2) on the top level, 0 below
    surface = jnp.zeros_like(temperature).at[0].set(2.0 / spacing)  # → (N_z,)
    return diffusivity * curvature + heating / heat_capacity * surface


# user
# E = ρ c_p ∫₀^200 T(z) dz, in J m⁻²
def heat_content(temperature):
    # today (integrix)
    return heat_capacity * rule.integrate(levels, temperature).value  # → ()


# today (integrix)
# Tₖ₊₁ = Φ(Tₖ, tₖ, h), with SSP-RK3 and h = 600 s for 10 days
final = march.run(march.RK3SSP(), tendency, 0.0, temperature, 600.0, 1440)  # → (N_z,)
# ΔE = E(t) − E(0), in MJ m⁻²
gained = (heat_content(final) - heat_content(temperature)) / 1e6  # → ()
# T(0, t) − T(0, 0), the warming at the surface
warming = final[0] - temperature[0]  # → ()

print(f"{gained:.4f} {warming:.2f}")
```

```text
86.4000 2.31
```

The column gained $Q t = 100 \times 864{,}000$ J m⁻² = 86.4 MJ m⁻² to round-off. That is no accident: the rule and the stencil describe the same discrete column. With the trapezoid weights $w_0 = w_{N_z - 1} = \Delta z / 2$ and $w_n = \Delta z$ between, the mirrored second differences telescope, $\sum_n w_n (T_{n+1} - 2 T_n + T_{n-1}) / \Delta z^2 = 0$, and the surface term contributes $w_0 \cdot 2 Q / (\rho c_p \Delta z) = Q / (\rho c_p)$, so $\mathrm{d}E / \mathrm{d}t = Q$ holds exactly for the discrete state. A Runge–Kutta step is a weighted sum of tendencies, so it keeps that linear invariant to round-off whatever its order. The step size changes only how the heat spreads with depth, not how much there is.

## To a tolerance: adaptive rules

With a fixed rule we choose the nodes ourselves. The rules of `integrix.adapt` refine until the error estimate meets $\max(\epsilon_\text{abs}, \epsilon_\text{rel} \vert \hat{I} \vert)$, and return the estimate, the number of evaluations and a `converged` flag instead of raising, so they run under `jit`. [`TanhSinh`](xref:api#integrix.adapt.TanhSinh) handles an endpoint singularity, here $\int_0^1 x^{-1/2} \mathrm{d}x = 2$, and [`GaussKronrod`](xref:api#integrix.adapt.GaussKronrod) a sharp peak, here a Lorentzian of width $\omega = 0.01$ at 0.3:

```python
from integrix import adapt

# today (integrix)
# ∫₀¹ x^(−1/2) dx = 2, with an integrable singularity at 0
singular = adapt.TanhSinh(rtol=1e-10)(lambda x: 1.0 / jnp.sqrt(x), 0.0, 1.0)
# today (integrix)
# ∫₀¹ dx / (ω² + (x − 0.3)²), a peak of width ω = 0.01 at 0.3
peaked = adapt.GaussKronrod(rtol=1e-10)(
    lambda x: 1.0 / (1e-4 + (x - 0.3) ** 2), 0.0, 1.0
)
# (arctan(0.7 / ω) + arctan(0.3 / ω)) / ω
exact = (jnp.arctan(70.0) + jnp.arctan(30.0)) / 0.01                # → ()
for result, reference in ((singular, 2.0), (peaked, exact)):
    relative = abs(result.value / reference - 1.0)
    print(
        f"{relative:.1e} {float(result.error):.1e}",
        int(result.n_evaluations),
        bool(result.converged),
    )
```

```text
0.0e+00 2.7e-15 513 True
4.4e-16 1.6e-08 495 True
```

Each line holds the true relative error, the rule's own error estimate, the number of evaluations and the flag. Both rules meet $\epsilon_\text{rel} = 10^{-10}$ in about 500 evaluations, and Gauss–Kronrod's estimate (the gap between the Kronrod and Gauss values on each subinterval) is conservative by eight orders of magnitude. [Chapter 8](../theory/08_adaptivity.md) has the theory and says which rule suits which integrand.

## Through a long march: adjoints

Reverse mode through a march stores every state by default. [`Checkpointed`](xref:api#integrix.adjoint.Checkpointed) stores every $k$-th state and recomputes the rest during the backward sweep, which trades memory for a second forward pass and gives the same gradient. For $\mathrm{d}y / \mathrm{d}t = -k y$ with $y(0) = 1$, the gradient of $y(T) = e^{-kT}$ with respect to the rate is $-T e^{-kT}$:

```python
from integrix import adjoint, march


# user
# dy/dt = −k y
def decay_rate(t, y, rate):
    return -rate * y


# user
# y(T) for T = 10, in 1,000 steps of h = 0.01
def final_value(rate, how):
    # today (integrix)
    return march.run(                                               # → ()
        march.RK4Classic(), decay_rate, 0.0, jnp.array(1.0), 0.01, 1000,
        args=rate, adjoint=how,
    )


# today (jax)
# ∂y(T)/∂k, every state stored
plain = jax.grad(final_value)(0.5, None)                            # → ()
# today (jax, integrix)
# ∂y(T)/∂k, every 100th state stored and the rest recomputed
checkpointed = jax.grad(final_value)(0.5, adjoint.Checkpointed(every=100))
# ∂y(T)/∂k = −T e^(−kT)
exact = -10.0 * jnp.exp(-5.0)                                       # → ()
print(f"{plain:.8f} {checkpointed:.8f} {exact:.8f}")
```

```text
-0.06737947 -0.06737947 -0.06737947
```

The plain and the checkpointed gradients agree, and both match the closed form. The checkpointed one stored 10 states instead of 1,000, which is what we want on a long rollout of a large state. [`Truncated`](xref:api#integrix.adjoint.Truncated), for the `AbstractStepper`s, cuts the gradient before the last $k$ steps instead, and `checkpointed_scan` and `truncated_scan` do the same for any `lax.scan` loop ([chapter 9](../theory/09_adjoints.md)).

## Against a probability: expectations

An expectation is an integral against a probability, $\mathbb{E}_p[f] = \int f(\vect{s}) p(\vect{s}) \mathrm{d}\vect{s}$, and in more than a few dimensions it is sampled rather than summed on a grid. Every estimator of `integrix.expect` returns an [`Expectation`](xref:api#integrix.expect.Expectation) with its standard error. For $f(\vect{s}) = \cos s_1 \cos s_2$ and $\vect{s} \sim \mathcal{N}(\vect{0}, \mat{I})$, the exact value is $e^{-1} = 0.36788$:

```python
from integrix import expect

# Shapes: D = 2 dimensions; S = 4,096 draws
mc_key, qmc_key = jax.random.split(jax.random.key(0))


# user
# f(s) = cos s₁ cos s₂, with 𝔼[f] = e^(−1) for s ~ 𝒩(0, I)
def product_of_cosines(point):
    return jnp.prod(jnp.cos(point))                                 # (D,) → ()


# user
# s ~ 𝒩(0, I), S draws
def standard_normal(key, n):
    return jax.random.normal(key, (n, 2))                           # → (S, D)


# today (integrix)
# Î = (1/S) ∑ₛ f(sₛ), with σ̂ / √S
plain = expect.monte_carlo(product_of_cosines, standard_normal, 4096, key=mc_key)
# today (integrix)
# Î over 16 shifted Sobol sets of 256 points, mapped to 𝒩(0, I)
quasi = expect.qmc(
    product_of_cosines, 2, 256, transform=jax.scipy.special.ndtri, key=qmc_key
)
for estimate in (plain, quasi):
    print(
        f"{estimate.value:.5f} ± {estimate.standard_error:.5f}",
        estimate.n_samples,
        f"{jnp.exp(-1.0):.5f}",
    )
```

```text
0.37056 ± 0.00681 4096 0.36788
0.36871 ± 0.00165 4096 0.36788
```

Both estimates are within one standard error of $e^{-1}$, and for the same 4,096 evaluations the randomised Sobol points of `qmc` give a standard error four times smaller than plain Monte Carlo, since their error falls faster than $S^{-1/2}$ for a smooth integrand. `importance` adds the effective sample size and the Pareto $\hat{k}$ of its weights, which flag a proposal whose tails are too light ([chapter 10](../theory/10_expectations.md)).

## Errors

Every error integrix raises derives from [`IntegrixError`](xref:api#integrix.IntegrixError). Bad input raises [`ValidationError`](xref:api#integrix.ValidationError), which is also a `ValueError`, so an existing `except ValueError` still catches it.

```python
import integrix as ix

ix.__version__  # a version string such as '0.0.2'
```
