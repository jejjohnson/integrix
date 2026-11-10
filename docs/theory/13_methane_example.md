---
numbering:
  enumerator: "13.%s"
---

# End to end: a methane plume

## One scene, every module

This chapter follows one synthetic satellite scene of a methane plume from the spectrum to an emission rate with an uncertainty, and every step is an integral that an earlier chapter set up. The source emits $Q = 0.3$ kg s⁻¹ (1080 kg h⁻¹), the wind carries the gas downwind, an imaging spectrometer sees the extra absorption in a band near 2300 nm, and we have to get back to $Q$.

| Step | Integral | integrix | Chapter |
| --- | --- | --- | --- |
| the column of the background atmosphere | optical depth from the top | `scan.cumulative(reverse=True)` | 6 |
| the band radiance the instrument sees | a weighted mean over wavelength | `rule.weighted_mean` | 5 |
| the retrieval's sensitivity | the derivative of that mean | `jax.grad` through it | 9 |
| the plume itself | a march of the transport equation | `march.run` with `RK3SSP` | 7 |
| the mass in the plume | an area integral over masked pixels | `rule.integrate` | 5 |
| the flux across the plume | line integrals across transects | `rule.integrate` | 5 |
| the sensitivity to the wind | the adjoint of the march | `Checkpointed` | 9 |
| the uncertainty of $Q$ | an expectation over wind and noise | `expect.monte_carlo` | 10 |

Everything here is a toy: a three-line spectrum, an exponential atmosphere, a 40 by 24 grid of 50 m pixels and a constant wind, so that the whole page runs in under half a minute. The numbers are not those of any real instrument, but every step is the one a real retrieval takes. The area and line integrals use plain weights arrays and masks; chapter 12 has the theory behind them.

## From spectrum to column

The background column is the vertical integral of methane, and its optical depth at wavelength $\lambda$ is a cross-section $\sigma(\lambda)$ times that column. We get the column from the top down with a reversed scan of the number density $n(z) \propto e^{-z / 8 \text{ km}}$, scale the cross-section so that the background has an optical depth of 0.4 at the strongest line, and let the sunlight cross the column twice (down and back up). A plume adds a fraction $\delta$ of the background column, so

$$L(\lambda; \delta) = e^{-2 \tau_0(\lambda) (1 + \delta)}, \quad L_b(\delta) = \frac{\int L(\lambda; \delta) R_b(\lambda) \mathrm{d}\lambda}{\int R_b(\lambda) \mathrm{d}\lambda}$$ (eq-plume-radiance)

with $\tau_0(\lambda)$ the background optical depth, $L$ the radiance relative to the sunlight and $L_b$ the band radiance under a Gaussian response $R_b$ 5 nm wide. The retrieval inverts $L_b$ for $\delta$ by linearising the log radiance about the background, $\hat{\delta} = \log(L_b / L_b(0)) / s$ with the sensitivity $s = \mathrm{d} \log L_b / \mathrm{d}\delta$ at $\delta = 0$, and JAX computes $s$ through the weighted mean (chapter 9).

```python
import jax

jax.config.update("jax_enable_x64", True)

from typing import Any

import einx
import jax.numpy as jnp
import numpy as np
from functax import quad
from integrix import adjoint, expect, march, rule, scan
from jaxtyping import Array, Float, PRNGKeyArray

# Shapes: N_λ = 401 wavelengths; N_z = 61 levels; 3 lines



# --- spectrum ---
# λₖ = 2280 + 0.1 k nm
wavelength = jnp.linspace(2280.0, 2320.0, 401)                    # → (N_λ,)
# today (functax)
# (λ, w) = the trapezoid rule on λ
spectral_rule = quad.trapezoid(wavelength)
# methane line centres λₗ in nm
line_centres = jnp.array([2295.0, 2300.0, 2306.0])                # → (3,)
# relative line strengths Sₗ
line_strengths = jnp.array([0.6, 1.0, 0.8])                       # → (3,)
# σ(λ) ∝ ∑ₗ Sₗ (γ/π) / ((λ − λₗ)² + γ²), γ = 0.4 nm
# oₗₖ = λₗ − λₖ
offsets = einx.subtract(                                          # → (3, N_λ)
    "line, wl -> line wl", line_centres, wavelength
)
# pₗₖ = (γ/π) / (oₗₖ² + γ²)
profiles = (0.4 / jnp.pi) / (offsets**2 + 0.4**2)                 # → (3, N_λ)
# σₖ = ∑ₗ Sₗ pₗₖ
cross_section = einx.dot(                                         # → (N_λ,)
    "line wl, line -> wl", profiles, line_strengths
)
# σ = σ / max σ
cross_section = cross_section / jnp.max(cross_section)            # → (N_λ,)
# uₖ = (λₖ − 2300) / 5
response_arg = (wavelength - 2300.0) / 5.0                        # → (N_λ,)
# R_b = exp(−uₖ² / 2)
response = jnp.exp(-0.5 * response_arg**2)                        # → (N_λ,)

# --- column ---
# zₖ = k km
height = jnp.linspace(0.0, 60.0, 61)                              # → (N_z,)
# n(z) = exp(−z / 8 km), relative
density = jnp.exp(-height / 8.0)                                  # → (N_z,)
# (z, w) = the trapezoid rule on z
# today (functax)
rule_z = quad.trapezoid(height)
# today (integrix)
# N(z) = ∫_z^top n(z′) dz′, from the top down
column_above = scan.cumulative(rule_z, density, reverse=True)     # → (N_z,)
# σ₀ = 0.4 / N(0): the background has τ₀ = 0.4 at the strongest line
cross_section_scale = 0.4 / column_above[0]                       # → ()
# σ₀ σ(λ)
unit_depth = cross_section_scale * cross_section                  # → (N_λ,)
# τ₀(λ) = σ₀ σ(λ) N(0)
background_depth = unit_depth * column_above[0]                   # → (N_λ,)


# user
def band_radiance(enhancement: float) -> Float[Array, ""]:
    """L_b(δ) = ∫ exp(−2 τ₀ (1 + δ)) R_b dλ / ∫ R_b dλ."""
    # 1 + δ
    scale = 1.0 + enhancement                                     # → ()
    # L(λ; δ) = exp(−2 τ₀ (1 + δ))
    radiance = jnp.exp(-2.0 * background_depth * scale)           # → (N_λ,)
    # today (integrix)
    # L_b = ∫ L R_b dλ / ∫ R_b dλ
    band = rule.weighted_mean(                                    # → ()
        spectral_rule, radiance, weight=response
    )
    return band


# L_b(0)
background_band = band_radiance(0.0)                              # → ()
# s = d log L_b / dδ at δ = 0
sensitivity = jax.grad(lambda d: jnp.log(band_radiance(d)))(0.0)  # → ()
print(
    f"column above 30 km: {float(column_above[30] / column_above[0]):.3f} of the total"
)
print(
    f"background band radiance {float(background_band):.4f}, "
    f"s = {float(sensitivity):.4f}"
)
for true_enhancement in (0.05, 0.2):
    # L_b(δ)
    band = band_radiance(true_enhancement)                        # → ()
    # r = L_b(δ) / L_b(0)
    band_ratio = band / background_band                           # → ()
    # δ̂ = log(r) / s
    retrieved = jnp.log(band_ratio) / sensitivity                 # → ()
    print(f"delta = {true_enhancement:.2f}: retrieved {float(retrieved):.4f}")
```

```text
column above 30 km: 0.023 of the total
background band radiance 0.8875, s = -0.1068
delta = 0.05: retrieved 0.0497
delta = 0.20: retrieved 0.1960
```

The linear retrieval is within 1 % of the enhancement at 5 % and loses about 2 % of it at 20 %, because the line saturates (the band radiance is convex in $\delta$). For the plume below the enhancement stays near 10 %, where the linear retrieval is close enough.

## The plume

The plume is the steady state of the advection–diffusion of the methane column $\Omega(x, y)$ in kg m⁻², in a wind $U = 4$ m s⁻¹ along $x$, with diffusivity $K = 20$ m² s⁻¹ and the source in one pixel. We step it with `RK3SSP` for 150 steps of 5 s, after checking the step against [`stable_step`](xref:api#integrix.march.stable_step) (chapter 7), on a grid whose tendency is a local stencil (upwind advection and centred diffusion, with zero outside the domain).

```python
# Shapes: N_y = 24 rows; N_x = 40 columns of 50 m pixels
# grid size and pixel width in m
n_rows, n_cols, spacing = 24, 40, 50.0
# wind in m/s, diffusivity in m²/s, emission in kg/s
wind, diffusivity, emission = 4.0, 20.0, 0.3
# S = Q / ΔA in the one source pixel, kg m⁻² s⁻¹
source = jnp.zeros((n_rows, n_cols)).at[n_rows // 2, 4].set(      # → (N_y, N_x)
    emission / spacing**2
)


# user
def tendency(
    t: float, column: Float[Array, "row col"], wind_speed: float
) -> Float[Array, "row col"]:
    """∂Ω/∂t = −U ∂Ω/∂x + K ∇²Ω + source, on a local five-point stencil."""
    # Ω = 0 outside the domain
    padded = jnp.pad(column, 1)                                   # → (N_y + 2, N_x + 2)
    # Ωʷ_ij = Ω_i,j−1
    west = padded[1:-1, :-2]                                      # → (N_y, N_x)
    # Ωᵉ_ij = Ω_i,j+1
    east = padded[1:-1, 2:]                                       # → (N_y, N_x)
    # Ωˢ_ij = Ω_i−1,j
    south = padded[:-2, 1:-1]                                     # → (N_y, N_x)
    # Ωⁿ_ij = Ω_i+1,j
    north = padded[2:, 1:-1]                                      # → (N_y, N_x)
    # −U ∂Ω/∂x ≈ −U (Ω − Ωʷ) / Δx
    advection = -wind_speed * (column - west) / spacing           # → (N_y, N_x)
    # Σ neighbours = Ωʷ + Ωᵉ + Ωˢ + Ωⁿ
    neighbours = west + east + south + north                      # → (N_y, N_x)
    # 4 Ω
    centre = 4.0 * column                                         # → (N_y, N_x)
    # Σ neighbours − 4 Ω
    laplacian_sum = neighbours - centre                           # → (N_y, N_x)
    # K ∇²Ω ≈ K (Σ neighbours − 4 Ω) / Δx²
    diffusion = diffusivity * laplacian_sum / spacing**2          # → (N_y, N_x)
    # ∂Ω/∂t = advection + diffusion + S
    rate = advection + diffusion + source                         # → (N_y, N_x)
    return rate


# user
def plume(wind_speed: float, adjoint_option: Any = None) -> Float[Array, "row col"]:
    """Ω after 150 steps of 5 s from an empty domain."""
    # Ω(0) = 0
    empty = jnp.zeros((n_rows, n_cols))                           # → (N_y, N_x)
    # today (integrix)
    # Ω(T) = Ω(0) + ∫₀ᵀ (∂Ω/∂t) dt, 150 RK3SSP steps of 5 s
    final = march.run(                                            # → (N_y, N_x)
        march.RK3SSP(),
        tendency,
        0.0,
        empty,
        5.0,
        150,
        args=wind_speed,
        adjoint=adjoint_option,
    )
    return final


# today (integrix)
# Δt_max from the advection and diffusion limits
limit = march.stable_step(                                        # → ()
    (spacing, spacing), speed=(wind, 0.0), diffusivity=(diffusivity, diffusivity)
)
# Ω(T) = march of ∂Ω/∂t
column = plume(wind)                                              # → (N_y, N_x)
# ΔA = Δx²
pixel_area = jnp.full((n_rows, n_cols), spacing**2)               # → (N_y, N_x)
# today (integrix)
# M = ∑ Ωᵢⱼ Δx²
plume_mass = rule.integrate(                                      # → ()
    pixel_area, column, axis=(0, 1)
).value
# τ = (N_x − 4) Δx / U: the time to cross from the source to the east edge
residence = (n_cols - 4) * spacing / wind
print(f"stable step {float(limit):.1f} s (we use 5 s)")
print(
    f"mass in the domain {float(plume_mass):.1f} kg, "
    f"Q x residence time {emission * float(residence):.1f} kg"
)
```

```text
stable step 8.9 s (we use 5 s)
mass in the domain 135.0 kg, Q x residence time 135.0 kg
```

The mass in the domain matches the emission rate times the time the gas takes to cross from the source to the downwind edge, which says the march has reached its steady state.

## The scene and the retrieval

The column enhancement relative to the background is $\delta = \Omega / \Omega_\text{bg}$, where the background column mass $\Omega_\text{bg} = \mathrm{XCH_4} \frac{p_\text{s}}{g} \frac{M_\text{CH4}}{M_\text{air}}$ follows from 1.9 ppm of methane under 1013 hPa. Each pixel's band radiance comes from [](#eq-plume-radiance), with 0.05 % noise, and the retrieval inverts it with the sensitivity above.

```python
# Shapes: N_y = 24 rows; N_x = 40 columns of 50 m pixels
# Ω_bg = XCH₄ (p_s / g) (M_CH4 / M_air), kg m⁻²
background_column = 1.9e-6 * 101325.0 / 9.81 * 16.04 / 28.96      # → ()
# relative noise on each band radiance
noise_level = 5e-4
# one key for the noise field, one for the Monte Carlo draws
noise_key, draws_key = jax.random.split(jax.random.key(0))
# δ = Ω / Ω_bg
enhancement = column / background_column                          # → (N_y, N_x)
# L_b,ij = L_b(δ_ij)
true_bands = jax.vmap(jax.vmap(band_radiance))(enhancement)       # → (N_y, N_x)


# user
def retrieve(bands: Float[Array, "row col"]) -> Float[Array, "row col"]:
    """Ω̂ = Ω_bg log(L_b / L_b(0)) / s, per pixel."""
    # r = L_b / L_b(0)
    ratio = bands / background_band                               # → (N_y, N_x)
    # log r
    log_ratio = jnp.log(ratio)                                    # → (N_y, N_x)
    # Ω̂ = Ω_bg log r / s
    estimate = background_column * log_ratio / sensitivity        # → (N_y, N_x)
    return estimate


# ε ~ 𝒩(0, 1)
noise = jax.random.normal(noise_key, (n_rows, n_cols))            # → (N_y, N_x)
# L̃_b = L_b (1 + η ε)
noisy_bands = true_bands * (1.0 + noise_level * noise)            # → (N_y, N_x)
# Ω̂ = Ω_bg log(L̃_b / L_b(0)) / s
retrieved = retrieve(noisy_bands)                                 # → (N_y, N_x)
# σ_Ω = Ω_bg η / |s|
column_noise = background_column * noise_level / abs(float(sensitivity))
print(
    f"background column {background_column:.4f} kg/m2, "
    f"retrieval noise {column_noise:.1e} kg/m2"
)
print(
    f"peak enhancement: true {float(column.max()):.2e}, "
    f"retrieved {float(retrieved.max()):.2e} kg/m2"
)
```

```text
background column 0.0109 kg/m2, retrieval noise 5.1e-05 kg/m2
peak enhancement: true 1.24e-03, retrieved 1.20e-03 kg/m2
```

## Two ways to an emission rate

The first estimate is the integrated mass enhancement. We keep the pixels where the retrieval is above three times its noise, the plume mask $P$, and integrate over it, $\mathrm{IME} = \int_P \hat{\Omega} \mathrm{d}A$. In a steady plume the mass in $P$ is the emission rate times the time the gas spends in $P$, which is the plume's length along the wind $L_P$ divided by the wind speed, so

$$\hat{Q}_\text{IME} = \frac{\mathrm{IME} U}{L_P}$$ (eq-ime)

The second estimate is a mass balance across the plume. In a steady state every transect across the wind carries the whole emission, $Q = U \int \Omega(x, y) \mathrm{d}y$, which is local conservation (chapter 3) applied to a line instead of a cell. Each transect is noisy on its own, so we average the transects from 0.5 to 1.5 km downwind.

```python
# Shapes: N_y = 24 rows; N_x = 40 columns; N_P plume columns; K = 21 transects
# P = {Ω̂ᵢⱼ > 3 σ_Ω}
in_plume = retrieved > 3.0 * column_noise                         # → (N_y, N_x)
# today (integrix)
# IME = ∑ over P of Ω̂ᵢⱼ Δx²
ime = rule.integrate(                                             # → ()
    pixel_area, retrieved, axis=(0, 1), where=in_plume
).value
# pₓ = any over y of Pᵢₓ
plume_in_column = einx.any("y x -> x", in_plume)                  # → (N_x,)
# the columns that hold plume pixels
plume_columns = jnp.nonzero(plume_in_column)[0]                   # → (N_P,)
# x_last = the downwind-most plume column
last_column = float(plume_columns.max())
# L_P = (x_last − x_source + ½) Δx
plume_length = (last_column - 4 + 0.5) * spacing
# Q̂_IME = IME U / L_P, in kg/h
ime_rate = float(ime) * wind / plume_length * 3600.0
print(
    f"IME {float(ime):.1f} kg over {int(einx.sum('y x ->', in_plume))} pixels, "
    f"L_P = {plume_length:.0f} m: Q = {ime_rate:.0f} kg/h"
)

# Δy = Δx
row_widths = jnp.full(n_rows, spacing)                            # → (N_y,)
# columns 10 to 30, 0.5 to 1.5 km downwind
transects = range(10, 31)
for name, field in (("true column", column), ("retrieved", retrieved)):
    # today (integrix)
    # Q = U ∫ Ω dy on each transect
    fluxes = [wind * rule.integrate(row_widths, field[:, c]).value for c in transects]
    print(
        f"transects, {name:11s}: Q = {float(np.mean(fluxes)) * 3600.0:.0f} kg/h "
        f"(single transects {float(np.min(fluxes)) * 3600.0:.0f} "
        f"to {float(np.max(fluxes)) * 3600.0:.0f})"
    )
```

```text
IME 105.6 kg over 149 pixels, L_P = 1775 m: Q = 857 kg/h
transects, true column: Q = 1080 kg/h (single transects 1080 to 1080)
transects, retrieved  : Q = 1075 kg/h (single transects 733 to 1374)
```

On the true column every transect returns the emission rate to within a fraction of a kg h⁻¹, because the finite-volume march conserves mass cell by cell and each transect sees all of it. On the retrieved column single transects scatter widely with the noise, and their average is much closer. The IME estimate comes out low: the mask misses the faint edges of the plume, where diffusion has spread the gas below the noise, so it holds less than the full residence-time mass. That is a known weakness of the method, and it is why real IME retrievals calibrate an effective wind speed against simulated plumes (Varon et al., 2018).

## Sensitivity to the wind

The wind enters twice, in the transport and in [](#eq-ime). Its effect on the transport is a gradient through 150 steps of the march, which is the adjoint of chapter 9. We take it with a checkpointed adjoint, storing every tenth state, and check it against plain reverse mode and a finite difference.

```python
# Shapes: N_y = 24 rows; N_x = 40 columns of 50 m pixels


# user
def domain_mass(wind_speed: float, adjoint_option: Any) -> Float[Array, ""]:
    """M(U) = ∑ Ωᵢⱼ(U) Δx² after the march."""
    # Ω(T; U) = march of ∂Ω/∂t
    final = plume(wind_speed, adjoint_option)                     # → (N_y, N_x)
    # today (integrix)
    # M = ∑ Ωᵢⱼ Δx²
    mass = rule.integrate(pixel_area, final, axis=(0, 1)).value   # → ()
    return mass


# today (integrix)
# store every tenth state of the march
option = adjoint.Checkpointed(every=10)
# ∂M/∂U by the checkpointed adjoint
checkpointed = jax.grad(domain_mass)(wind, option)                # → ()
# ∂M/∂U by plain reverse mode
plain = jax.grad(domain_mass)(wind, None)                         # → ()
# M(U + h)
mass_up = domain_mass(wind + 1e-4, None)                          # → ()
# M(U − h)
mass_down = domain_mass(wind - 1e-4, None)                        # → ()
# ∂M/∂U ≈ (M(U + h) − M(U − h)) / 2h
finite_difference = (mass_up - mass_down) / 2e-4                  # → ()
print(
    f"dM/dU: checkpointed {float(checkpointed):.4f}, plain {float(plain):.4f}, "
    f"finite difference {float(finite_difference):.4f} kg per m/s"
)
```

```text
dM/dU: checkpointed -33.6742, plain -33.6742, finite difference -33.6742 kg per m/s
```

A faster wind flushes the gas out of the scene sooner, so the mass in the domain falls by about 34 kg for each m s⁻¹, close to $-M / U$, which is what a residence time inversely proportional to the wind would give.

## The uncertainty of the emission rate

The wind is rarely known to better than 10 to 20 %, and the noise is different in every scene. We treat both as random, the wind as $U \sim \mathcal{N}(4, 0.4^2)$ m s⁻¹ and the noise as fresh draws, and propagate them through the transect estimate with [`monte_carlo`](xref:api#integrix.expect.monte_carlo). The emission rate's mean and spread are two expectations, $\mathbb{E}[\hat{Q}]$ and $\mathbb{E}[\hat{Q}^2]$, taken from the same draws.

```python
# Shapes: N_E = 256 draws of (wind, noise); N_y = 24 rows; N_x = 40 columns
# K = 21 transects


# user
def scene_draws(
    key: PRNGKeyArray, n: int
) -> tuple[Float[Array, "draw"], Float[Array, "draw row col"]]:
    """U⁽ᵉ⁾ ~ 𝒩(4, 0.4²) and a fresh noise field per draw."""
    # one key for the winds, one for the noise fields
    wind_key, field_key = jax.random.split(key)
    # ξ⁽ᵉ⁾ ~ 𝒩(0, 1)
    wind_noise = jax.random.normal(wind_key, (n,))                # → (N_E,)
    # U⁽ᵉ⁾ = 4 + 0.4 ξ⁽ᵉ⁾
    winds = 4.0 + 0.4 * wind_noise                                # → (N_E,)
    # ε⁽ᵉ⁾ ~ 𝒩(0, 1)
    fields = jax.random.normal(field_key, (n, n_rows, n_cols))    # → (N_E, N_y, N_x)
    return winds, fields


# user
def transect_rate(
    draw: tuple[Float[Array, ""], Float[Array, "row col"]],
) -> Float[Array, "2"]:
    """Q̂ = U mean over transects of ∫ Ω̂ dy, and Q̂², in kg/h."""
    # one wind and one noise field per draw
    wind_draw, noise_draw = draw
    # L̃_b = L_b (1 + η ε)
    noisy_bands = true_bands * (1.0 + noise_level * noise_draw)   # → (N_y, N_x)
    # Ω̂ = Ω_bg log(L̃_b / L_b(0)) / s
    estimate = retrieve(noisy_bands)                              # → (N_y, N_x)
    # Ω̂ on the transect columns 10 to 30
    window = estimate[:, 10:31]                                   # → (N_y, K)
    # today (integrix)
    # ∫ Ω̂ dy on each transect
    line_integrals = rule.integrate(                              # → (K,)
        row_widths, window, axis=0
    ).value
    # mean over the transects
    mean_integral = jnp.mean(line_integrals)                      # → ()
    # Q̂ = U mean(∫ Ω̂ dy), in kg/h
    rate = wind_draw * mean_integral * 3600.0                     # → ()
    # Q̂²
    rate_squared = rate**2                                        # → ()
    # (Q̂, Q̂²)
    rate_moments = jnp.stack([rate, rate_squared])                # → (2,)
    return rate_moments


# today (integrix)
# E[Q̂], E[Q̂²] ≈ (1/N_E) ∑ₑ (Q̂⁽ᵉ⁾, Q̂⁽ᵉ⁾²)
moments = expect.monte_carlo(                                     # → value (2,)
    transect_rate, scene_draws, 256, key=draws_key
)
# E[Q̂]
mean_rate = float(moments.value[0])
# σ_Q = √(E[Q̂²] − E[Q̂]²)
spread = float(np.sqrt(moments.value[1] - moments.value[0] ** 2))
print(
    f"Q = {mean_rate:.0f} +/- {spread:.0f} kg/h "
    f"(Monte Carlo error of the mean {float(moments.standard_error[0]):.0f} kg/h; "
    f"true {emission * 3600.0:.0f})"
)
```

```text
Q = 1074 +/- 104 kg/h (Monte Carlo error of the mean 7 kg/h; true 1080)
```

The spread is about 10 %, the size of the wind error, since $Q$ is proportional to $U$; the noise adds less, because the 21 transects average much of it out. The Monte Carlo error of the mean is small next to the spread, so 256 draws are enough to describe it.
