<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
  <img alt="integrix" src="docs/assets/logo-light.svg" width="340">
</picture>

**Integration for JAX: quadrature sums, running integrals, time steppers, adaptive quadrature, adjoints and expectations, on functax, diffrax, lineax and optimistix.**

[![Tests](https://github.com/jejjohnson/integrix/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/integrix/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/integrix/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/integrix/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/integrix/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/integrix/actions/workflows/typecheck.yml)
[![Docs](https://github.com/jejjohnson/integrix/actions/workflows/pages.yml/badge.svg)](https://jejjohnson.github.io/integrix/)
[![codecov](https://codecov.io/gh/jejjohnson/integrix/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/integrix)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)

📖 [Documentation](https://jejjohnson.github.io/integrix/) ·
🚀 [Quickstart](https://jejjohnson.github.io/integrix/quickstart/) ·
📐 [Mathematical reference](https://jejjohnson.github.io/integrix/integral/) ·
📚 [API reference](https://jejjohnson.github.io/integrix/reference/) ·
🧭 [Design](https://github.com/jejjohnson/geostack/blob/main/docs/design/libraries/integrix.md)

</div>

---

A band radiance, the mass of a plume over an image, the optical depth down to every level of a column and the state of an ocean model after ten days of stepping are all integrals. Each library of the stack used to write its own (a trapezoid sum here, a cumulative sum of increments there, a Runge–Kutta loop somewhere else), so integrix treats them as one object,

```math
I[f] = \int_\Omega f(\mathbf{s}) \mathrm{d}\mu(\mathbf{s})
```

where $`\mathbf{s} \in \Omega \subset \mathbb{R}^{D_s}`$ is a point of the domain (a height, a wavelength, a time, a pixel or a draw from a distribution), $`\mu`$ is a measure on $`\Omega`$ (a length, an area, a spectral response or a probability) and $`f: \Omega \to \mathbb{R}^{D_f}`$ is the integrand, batched over $`D_f`$ entries, so $`I[f] \in \mathbb{R}^{D_f}`$. integrix computes it in three ways: a rule sums weights against samples, a scan keeps the integral up to every node, and a march steps an integrand that depends on the solution, as the right-hand side of an ODE does.

```python
# Shapes: N_z levels
# (levels, extinction, scale_height, tendency and temperature as in the examples below)

# β(zₙ; H) at every level
profile = extinction(scale_height)  # → (N_z,)
# today (integrix)
# τ₀ = ∫ β dz ≈ ∑ₙ wₙ β(zₙ)
column = rule.integrate(levels, profile).value  # → ()
# today (integrix)
# τ(zₙ) = ∫ from zₙ to z_top of β dz, at every level
optical_depth = scan.cumulative(levels, profile, reverse=True)  # → (N_z,)
# today (integrix)
# Tₖ₊₁ = Φ(Tₖ, tₖ, h)
final = march.run(march.RK3SSP(), tendency, 0.0, temperature, 600.0, 1440)  # → (N_z,)
```

None of it is a new framework. A rule is [`functax`](https://github.com/jejjohnson/functax)'s: integrix reads its `nodes` and `weights` and never builds one functax already has, so a trapezoid, Simpson, Gauss–Legendre or Clenshaw–Curtis rule, a rule along a polyline or the cell areas of a grid is summed and scanned as it is. The single-step steppers are [`diffrax`](https://github.com/patrick-kidger/diffrax) solvers, so `diffrax.diffeqsolve`, its step-size controllers and its adjoints keep working with them. And the implicit stage of an IMEX step is a Newton solve from [`optimistix`](https://github.com/patrick-kidger/optimistix) with a GMRES inner solve from [`lineax`](https://github.com/patrick-kidger/lineax), so the Jacobian of a large field is never formed.

<p align="center"><img src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/integrix_overview/integrix_overview.png" alt="integrix in one picture: the integral I = ∫ f dμ at the top; three operations on it, a sum (integrix.rule), a scan (integrix.scan) and a march (integrix.march), each with its equation and public names; adaptivity, adjoints and expectations as cross-cutting modules below them; functax, diffrax, and lineax with optimistix underneath" width="100%"></p>

## 🌟 Highlights

- 🧮 **Sums against any rule**: `rule.integrate` contracts a functax rule, a tuple of 1-D rules (a tensor rule) or a plain weights array over one axis or several, batched over the rest, with a mask (`where=`) and `nan_policy="omit"`; `log_integrate` sums in log space and `weighted_mean` averages against a weight.
- 📈 **Running integrals**: `scan.cumulative` returns the trapezoid integral up to every node, forward or from the far end, as a prefix sum of depth $`O(\log N)`$ with `jax.lax.associative_scan`.
- ⏱️ **Time steppers**: forward Euler, Heun, SSP Runge–Kutta of orders 2, 3 and 4 (Ketcheson's ten-stage scheme), classic RK4, Adams–Bashforth 2 and 3, leapfrog with a Robert–Asselin filter, split-explicit and IMEX-SSP2, all run by `march.run`, with `stable_step` for the largest stable step.
- 🎯 **Adaptive quadrature**: `GaussKronrod`, `TanhSinh` (for endpoint singularities and infinite limits) and `Romberg` refine to a tolerance and return an error estimate and a `converged` flag instead of raising, so they run under `jit`; `cumulative_adaptive` gives the integral to every node.
- 🔁 **Adjoints for long loops**: `Checkpointed` stores every k-th state and recomputes the rest, and `Truncated` cuts the gradient before the last k steps, passed to `march.run` or as `checkpointed_scan` and `truncated_scan` around any `lax.scan` loop.
- 🎲 **Expectations with a standard error**: `monte_carlo`, randomised `qmc` (Sobol or Halton) and `importance` return an `Expectation` with its standard error, and importance sampling also reports the effective sample size and the Pareto $`\hat{k}`$ of its weights.
- ✅ **Tested against references**: SciPy, quadax and closed forms in double precision, with an order-of-accuracy slope for every stepper and a standard-error bound for every estimator.

## 📐 One integral, three operations

A rule replaces the integral with a weighted sum over $`N`$ nodes $`\mathbf{s}_n \in \Omega`$, with weights $`w_n`$ and samples $`f_n = f(\mathbf{s}_n)`$,

```math
\hat{I}_N[f] = \sum_{n=1}^{N} w_n f(\mathbf{s}_n)
```

and the three operations differ in what they keep of that sum and where its samples come from. With $`\mathbf{x}_k \in \mathbb{R}^{D_x}`$ the state of a model at time $`t_k`$, $`E_N[f] = I[f] - \hat{I}_N[f]`$ the error of a rule and $`p`$ a probability density on $`\Omega`$:

| Operation | Equation | Module | For |
|---|---|---|---|
| Sum | $`\hat{I}_N[f] = \sum_n w_n f_n`$ | `integrix.rule` | a band radiance, a column amount, the mass of a plume |
| Scan | $`F_k = F_{k-1} + \frac{1}{2} (s_k - s_{k-1}) (f_{k-1} + f_k)`$ | `integrix.scan` | the integral up to every node: optical depth, cumulative mass |
| March | $`\mathbf{x}_{k+1} = \mathbf{x}_k + \int_{t_k}^{t_{k+1}} f(\mathbf{x}(t), t) \mathrm{d}t`$ | `integrix.march` | an integrand that depends on the solution, as in time stepping |
| Adapt | $`\vert E_N[f] \vert \le \max(\epsilon_\text{abs}, \epsilon_\text{rel} \vert \hat{I}_N[f] \vert)`$ | `integrix.adapt` | a rule or a scan refined to a tolerance |
| Differentiate | reverse mode, checkpointed or truncated | `integrix.adjoint` | gradients through a long scan or march |
| Expect | $`\mathbb{E}_p[f] = \int_\Omega f(\mathbf{s}) p(\mathbf{s}) \mathrm{d}\mathbf{s}`$ | `integrix.expect` | an integral against a probability, by sampling |

The first three rows are the operations, and the last three apply across them: adaptivity refines a rule or a scan, an adjoint runs back through a scan or a march, and an expectation is an integral against a probability.

The march row is the integral form of an ODE whose right-hand side $`f`$ is the integrand. A Runge–Kutta step with $`N_\text{stage}`$ stages and step $`h`$ is a rule on $`[t_k, t_k + h]`$, with the nodes $`t_k + c_i h`$ and the weights $`h b_i`$ of its Butcher tableau,

```math
\int_{t_k}^{t_k + h} f(t) \mathrm{d}t \approx h \sum_{i=1}^{N_\text{stage}} b_i f(t_k + c_i h)
```

except that each stage evaluates $`f`$ at a state the earlier stages built, so a march is stepped one node at a time where a rule is summed in one pass ([chapter 7](https://jejjohnson.github.io/integrix/marches/) of the mathematical reference).

## 📥 Install

integrix is not on PyPI, so install it from GitHub. It brings in JAX, Equinox, einx, jaxtyping, diffrax, lineax and optimistix from PyPI, and functax and gaussx from GitHub, which its own metadata pins by tag, so pip and uv fetch them with no extra configuration. It has no extras and needs Python 3.12 or newer.

```bash
pip install "integrix @ git+https://github.com/jejjohnson/integrix.git"
```

The stack pins integrix by tag in the same way, as `integrix @ git+https://github.com/jejjohnson/integrix@vX.Y.Z`.

## 🛰️ Quickstart: optical depth through a column

An absorbing gas whose extinction falls off with height $`z`$ as $`\beta(z; H) = \beta_0 e^{-z/H}`$, with $`\beta_0 = 0.1`$ km⁻¹ at the surface and the scale height $`H = 8`$ km, sampled at $`N_z = 121`$ levels $`z_n`$ every 0.5 km up to $`z_\text{top} = 60`$ km. A rule gives the optical depth of the whole column, $`\tau_0(H) = \int_0^{z_\text{top}} \beta(z; H) \mathrm{d}z`$. A reversed scan gives the optical depth $`\tau(z_n)`$ from the top of the column down to every level, and with it the transmittance $`\mathcal{T}(z_n) = e^{-\tau(z_n)}`$ of the path from the top to that level. And `jax.grad` through the rule gives the sensitivity $`\partial \tau_0 / \partial H`$.

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

This prints `0.79982 0.4494 0.09950`: the optical depth of the column, the transmittance of the whole column (45 % of the light at the top reaches the surface) and the growth of the optical depth per km of scale height. The closed forms are $`\tau_0 = \beta_0 H (1 - e^{-z_\text{top}/H}) = 0.79956`$ and $`\partial \tau_0 / \partial H = \beta_0 (1 - (1 + z_\text{top}/H) e^{-z_\text{top}/H}) = 0.09953`$ per km, so both values are off by $`3.3 \times 10^{-4}`$ in relative terms, which is the error of the trapezoid rule at a 0.5 km spacing. Nothing here depends on the profile being exponential: swap `extinction` for a model's profile, or `quad.trapezoid` for any other functax rule, and the sum, the scan and the gradient stay the same.

The [quickstart](https://jejjohnson.github.io/integrix/quickstart/) goes further: the calling convention (`axis=`, `where=`, `nan_policy=`), an adaptive rule to a tolerance, a gradient through a long march with a checkpointed adjoint, and an expectation with its standard error.

## 🌊 Warming an ocean column

A march is the operation whose integrand depends on the solution. Take a 200 m ocean column with a 20 °C mixed layer over a thermocline, mixed by diffusion with $`\kappa = 10^{-4}`$ m² s⁻¹ and heated through the surface by $`Q = 100`$ W m⁻², with no flux through the bottom. The temperature $`T(z, t)`$ in °C at depth $`z`$ follows $`\partial T / \partial t = \kappa \partial^2 T / \partial z^2`$, and the heat content of the column is $`E(t) = \rho c_p \int_0^{200} T(z, t) \mathrm{d}z`$ in J m⁻², with the volumetric heat capacity of sea water $`\rho c_p = 1025 \times 3990`$ J m⁻³ K⁻¹. The march steps $`T`$ for ten days with `march.RK3SSP`, a diffrax solver, and the rule measures $`E`$ before and after.

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

This prints `86.4000 2.31`: the column gained 86.4 MJ m⁻², which is $`Q t = 100 \times 864{,}000`$ J m⁻² to round-off, and the surface warmed by 2.31 °C. The budget closes exactly because the trapezoid weights and the mirrored ends describe the same discrete column, so the rule measures the heat the stencil conserves. Swapping `march.RK3SSP()` for another stepper of `integrix.march`, or for one of diffrax's own solvers, changes nothing else in the block.

## 🔬 What else is in the box

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/integrix_roles/integrix_roles-dark.svg">
    <img alt="integrix on the GeoStack role ladder: run and the checkpointed and truncated scans as algorithms; split-explicit terms beside diffrax's terms as the model; adaptive rules, diffrax solvers, steppers, IMEX and adjoint options as strategies; sums, running integrals, expectations, importance weights and step functions as primitives; the Integral and Expectation results beside functax's rules as structures; with diffrax, optimistix and lineax as the floor the strategies extend" src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/integrix_roles/integrix_roles-light.svg" width="100%">
  </picture>
</p>

Each public name sits on one rung of the GeoStack role ladder, by the job it does; the table lists them by module.

| Module | Symbols | Notes |
|---|---|---|
| `integrix.rule` | `Integral`, `integrate`, `log_integrate`, `weighted_mean` | Weights against samples over one axis or several, batched over the rest; an `Integral` carries `value`, `error`, `n_evaluations` and `converged` |
| `integrix.scan` | `cumulative` | The running trapezoid integral along one axis, forward or reversed |
| `integrix.march` | `run`, `AbstractStepper`, `AB2Solver`, `LeapfrogRAFSolver`, `SplitExplicitRKSolver`, `SplitExplicitTerms`, `ForwardEulerDfx`, `RK2Heun`, `SSP_RK2`, `RK3SSP`, `RK4Classic`, `SSP_RK104`, `AB3Solver`, `IMEX_SSP2`, `imex_solver`, `imex_stepsize_controller`, `stable_step`, `euler_step`, `heun_step`, `rk3_ssp_step`, `rk4_step`, `ab2_step`, `ab3_step`, `leapfrog_raf_step`, `split_explicit_step`, `imex_ssp2_step` | The single-step schemes and AB3 subclass diffrax's solvers; AB2, leapfrog and split-explicit are `AbstractStepper`s that `run` loops with `lax.scan`; each `*_step` function is one step on arrays |
| `integrix.adapt` | `GaussKronrod`, `TanhSinh`, `Romberg`, `cumulative_adaptive` | To a tolerance, with an error estimate and a `converged` flag |
| `integrix.adjoint` | `Checkpointed`, `Truncated`, `checkpointed_scan`, `truncated_scan`, `to_diffrax_adjoint` | Passed to `march.run` as `adjoint=`, or around any `lax.scan`; `to_diffrax_adjoint` maps a pipekit-cycle adjoint spec onto a diffrax adjoint |
| `integrix.expect` | `Expectation`, `monte_carlo`, `qmc`, `importance`, `effective_sample_size`, `pareto_smoothed_log_weights`, `log_gauss_hermite`, `log_simpson` | Sampled expectations with a standard error, the diagnostics of importance weights, and two rules that stay in log space: a lognormal expectation by Gauss–Hermite, and Simpson on a grid in log q |
| `integrix` | `IntegrixError`, `ValidationError`, `__version__` | Every error derives from `IntegrixError`; `ValidationError` is also a `ValueError` |

## 📖 Documentation

- **[Quickstart](https://jejjohnson.github.io/integrix/quickstart/)**: the calling convention, double precision, and a first example of each module.
- **[Mathematical reference](https://jejjohnson.github.io/integrix/integral/)**: thirteen chapters from the integral to the mass of a methane plume, through rules, scans, marches, adaptivity, adjoints, expectations and precision, with a notation page.
- **[Spatial integrals](https://jejjohnson.github.io/integrix/spatial-integrals/)**: the mass of a plume over an image, the flux across a line and the column over model levels, all with `integrate`, `where=` and `nan_policy=`.
- **[Global and local convergence](https://jejjohnson.github.io/integrix/theory-convergence/)**: an executed notebook comparing global and local rules on a smooth field and a kinked one.
- **[API reference](https://jejjohnson.github.io/integrix/reference/)**: every public symbol, generated from the docstrings.
- **[Design page](https://github.com/jejjohnson/geostack/blob/main/docs/design/libraries/integrix.md)**: the scope, the boundaries and the decisions, in the GeoStack docs.

## 🧩 Ecosystem

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/integrix_ecosystem/integrix_ecosystem-dark.svg">
    <img alt="The packages that import integrix: finitevolX in GeoNumerics, which uses its steppers; vardax in GeoML, which uses its adjoints; plumax and sesmax-core in GeoModels, which use its rules, expectations and steppers; plumax and sesmax-core build on finitevolX, and on vardax through an extra; integrix builds on functax, diffrax, lineax, optimistix, JAX, Equinox, einx and jaxtyping" src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/integrix_ecosystem/integrix_ecosystem-light.svg" width="100%">
  </picture>
</p>

integrix is Core tier 1 of the [GeoStack](https://github.com/jejjohnson/geostack). It owns every procedure that sums, scans, adapts or marches with a rule, and leaves the rest to the libraries below:

| Concern | Owner |
|---|---|
| Rules: nodes and weights on intervals, cells, curves and the sphere (integrix sums and scans them) | [`functax`](https://github.com/jejjohnson/functax) |
| The solver protocol, `diffeqsolve`, step-size controllers and adjoints (the steppers of `integrix.march` extend them) | [`diffrax`](https://github.com/patrick-kidger/diffrax) |
| The Newton solve of an implicit IMEX stage | [`optimistix`](https://github.com/patrick-kidger/optimistix) |
| The matrix-free GMRES inside that Newton solve | [`lineax`](https://github.com/patrick-kidger/lineax) |
| Gaussian expectations with a closed rule (Gauss–Hermite, unscented, sigma points) | [`gaussx`](https://github.com/jejjohnson/gaussx) |

Dependencies point one way: finitevolX, vardax, plumax and sesmax-core import integrix today, never the reverse. radtrax, geotoolz and xtremax switch over in Wave 5 ([#13](https://github.com/jejjohnson/integrix/issues/13)), where the switches of plumax and vardax have already closed.

## 🌳 Repository layout

```
src/integrix/
├── __init__.py        ← the version and the exceptions
├── rule.py            ← public modules, each re-exporting from _src/
├── scan.py
├── march.py
├── adapt.py
├── adjoint.py
├── expect.py
├── exceptions.py
└── _src/              ← rule/, scan/, march/, adapt/, adjoint/, expect/
docs/                  ← MyST guides, the mathematical reference and notebooks, plus the Zensical API reference
plugins/integrix/      ← the Claude Code plugin: skills and a reuse reviewer for downstream projects
tests/                 ← one folder per module, against SciPy, quadax and closed forms
```

## ⚙️ Development

```bash
git clone https://github.com/jejjohnson/integrix.git
cd integrix
uv run poe install     # every dependency group + the pre-commit hooks
uv run poe check       # lint, format check, typecheck, tests with the 90 % coverage gate
uv run poe docs        # the MyST prose and the Zensical API reference, assembled into public/
```

`uv run poe` lists every task. [`AGENTS.md`](AGENTS.md) holds the rules for contributors and coding agents: the module map, the contracts, the tests that enforce them and the house style. [`docs/contributing.md`](docs/contributing.md) has the label taxonomy and the epic model.

## 🎓 References

- Blelloch (1990). *Prefix Sums and Their Applications*. Technical report CMU-CS-90-190, Carnegie Mellon University.
- Davis and Rabinowitz (1984). *Methods of Numerical Integration*, 2nd edition. Academic Press.
- Durran (2010). *Numerical Methods for Fluid Dynamics: With Applications to Geophysics*, 2nd edition. Springer.
- Griewank and Walther (2008). *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*, 2nd edition. SIAM.
- Hairer, Nørsett and Wanner (1993). *Solving Ordinary Differential Equations I: Nonstiff Problems*, 2nd edition. Springer.
- Higham (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd edition. SIAM.
- Ketcheson (2008). Highly efficient strong stability-preserving Runge–Kutta methods with low-storage implementations. *SIAM Journal on Scientific Computing* 30(4), 2113–2136.
- Owen (2013). *Monte Carlo Theory, Methods and Examples*.
- Piessens, de Doncker-Kapenga, Überhuber and Kahaner (1983). *QUADPACK: A Subroutine Package for Automatic Integration*. Springer.
- Takahasi and Mori (1974). Double exponential formulas for numerical integration. *Publications of the Research Institute for Mathematical Sciences* 9, 721–741.
- Trefethen (2008). Is Gauss quadrature better than Clenshaw–Curtis? *SIAM Review* 50(1), 67–87.
- Trefethen and Weideman (2014). The exponentially convergent trapezoidal rule. *SIAM Review* 56(3), 385–458.
- Varon et al. (2018). Quantifying methane point sources from fine-scale satellite observations of atmospheric methane plumes. *Atmospheric Measurement Techniques* 11, 5673–5686.
- Vehtari, Simpson, Gelman, Yao and Gabry. [Pareto smoothed importance sampling](https://arxiv.org/abs/1507.02646). arXiv:1507.02646.

## 📄 License

MIT. See [LICENSE](LICENSE).
