---
numbering:
  enumerator: "1.%s"
---

# The integral

## One day of Earth observation

Take one day of a methane-observing satellite and follow the numbers. An imaging spectrometer records radiance in a few hundred channels, and each channel is an integral of the spectrum against the instrument's spectral response. The retrieval turns the radiance into a column of methane, which is an integral of the mole fraction over pressure. The level-3 product averages the columns onto a map, and the global mean of that map is an integral over the sphere weighted by the cosine of latitude. The ocean model that the same group runs reports the heat content of the upper ocean, an integral of temperature over depth. Finally, the emission rate of a plume comes with an uncertainty, and that uncertainty is an expectation, which is an integral against a posterior density.

So one day of data has at least five integrals in it, over five different domains: wavelength, pressure, the sphere, depth and a parameter space. They look very different, but they are the same object, and integrix treats them with the same few operations. In this chapter we write that object down once, so that the rest of the section can be about how we compute it.

## The definition

Let $\Omega$ be a domain (an interval of wavelength, a column of pressure, the sphere, a parameter space) with coordinates $\vect{s} \in \Omega \subset \mathbb{R}^{D_s}$, and let $\mu$ be a measure on it. The integral of a function $f: \Omega \to \mathbb{R}$ is

$$I[f] = \int_\Omega f(\vect{s}) \mathrm{d}\mu(\vect{s})$$ (eq-integral)

where $I[f] \in \mathbb{R}$ is a number. When the measure has a density $\rho_\mu(\vect{s}) \ge 0$ with respect to the ordinary volume element (length, area, volume), [](#eq-integral) is $\int_\Omega f(\vect{s}) \rho_\mu(\vect{s}) \mathrm{d}\vect{s}$. The density is where most of the geoscience hides: the $\cos \phi$ of an area on the sphere, the spectral response of a channel, the posterior density of an emission rate.

A function $f$ can carry extra axes that are not integrated (channels, time steps, ensemble members), so $f: \Omega \to \mathbb{R}^{D_f}$ and $I[f] \in \mathbb{R}^{D_f}$ are integrated entry by entry. integrix calls these the batch axes.

## Five integrals, one form

The five integrals of the day all fit [](#eq-integral). For each one we write the full form, then read off its domain $\Omega$, its measure $\mathrm{d}\mu$ and the integrand $f$ that is left inside, and name the integrix function that computes it. [](#tbl-day-symbols) collects the physical symbols.

:::{table} The physical symbols of this page, with their shapes and units.
:label: tbl-day-symbols

| Symbol | Shape, units | Meaning |
| --- | --- | --- |
| $\lambda$ | $\mathbb{R}_{>0}$, nm | wavelength, in the spectral domain $\mathcal{S}$ |
| $L(\lambda)$, $L_c$ | $\mathbb{R}_{\ge 0}$, radiance units | the spectrum and the band radiance of channel $c$ |
| $R_c(\lambda)$ | $\mathbb{R}_{\ge 0}$ | the spectral response of channel $c$ |
| $p$, $p_\text{s}$ | $\mathbb{R}_{\ge 0}$, hPa | pressure and surface pressure |
| $c(p)$ | $\mathbb{R}_{>0}$, ppb | the methane dry-air mole fraction |
| $\phi$, $\vartheta$ | radians | latitude and longitude |
| $T$, $\Theta$ | K, °C | surface temperature and ocean potential temperature |
| $z$, $H$ | m | height (negative below the sea surface) and the depth of the layer |
| $\rho_0$, $c_p$ | kg m⁻³, J kg⁻¹ K⁻¹ | reference density and heat capacity of seawater |
| $Q$, $\vect{y}$ | kg h⁻¹, $\mathbb{R}^{D_y}$ | the emission rate and the observations it is inferred from |
| $p_\text{post}(Q \mid \vect{y})$ | $\mathbb{R}_{\ge 0}$, h kg⁻¹ | the posterior density of the emission rate, labelled because $p$ is pressure on this page |
:::

**Band radiance.** An imaging spectrometer never sees the spectrum itself, since each channel sees the spectrum averaged under its own spectral response. We have the full form

$$L_c = \frac{\int_{\mathcal{S}} L(\lambda) R_c(\lambda) \mathrm{d}\lambda}{\int_{\mathcal{S}} R_c(\lambda) \mathrm{d}\lambda}$$ (eq-band-radiance)

where the domain is the spectral window $\mathcal{S}$, the integrand is the spectrum $L(\lambda)$, and the measure is the normalised response $R_c(\lambda) \mathrm{d}\lambda / \int_{\mathcal{S}} R_c(\lambda) \mathrm{d}\lambda$. It is a ratio of two integrals with the same measure, so integrix computes it with [`weighted_mean`](xref:api#integrix.rule.weighted_mean), which uses the same weights for both sums. If the two sums used different weights, the ratio would drift for a reason that has nothing to do with the spectrum.

**Column average.** A retrieval reports methane as a column-averaged dry-air mole fraction, which is what a model column has to be compared with. Assuming dry air and a constant gravity, a pressure interval is proportional to the mass of air in it, and we have the full form

$$\mathrm{XCH_4} = \frac{1}{p_\text{s}} \int_0^{p_\text{s}} c(p) \mathrm{d}p$$ (eq-xch4)

where the domain is the column $[0, p_\text{s}]$ from the top of the atmosphere to the surface, the integrand is the mole fraction $c(p)$, and the measure is $\mathrm{d}p / p_\text{s}$, the fraction of the column's air in each pressure interval. It is a plain [`integrate`](xref:api#integrix.rule.integrate) over the levels. A real retrieval adds water vapour and an averaging kernel on top, which change the weights but not the form.

**Global mean.** A level-3 map is averaged to one number per day, and every cell has to count by its area, not by its index. We have the full form

$$\bar{T} = \frac{1}{4 \pi} \int_0^{2 \pi} \int_{-\pi/2}^{\pi/2} T(\phi, \vartheta) \cos \phi \mathrm{d}\phi \mathrm{d}\vartheta$$ (eq-global-mean)

where the domain is the unit sphere $\mathbb{S}^2$ in latitude $\phi$ and longitude $\vartheta$, the integrand is the surface temperature $T(\phi, \vartheta)$, and the measure is the area element $\cos \phi \mathrm{d}\phi \mathrm{d}\vartheta$ divided by the sphere's area $4 \pi$. The measure is a product of one factor per axis, so integrix computes it with [`integrate`](xref:api#integrix.rule.integrate) and a tensor rule, one rule in latitude and one in longitude.

**Ocean heat content.** The heat stored in the upper ocean is the quantity behind most statements about ocean warming, and it comes from a temperature profile. We have the full form

$$\mathrm{OHC} = \rho_0 c_p \int_{-H}^{0} \Theta(z) \mathrm{d}z$$ (eq-ohc)

where the domain is the layer $[-H, 0]$ from depth $H$ to the sea surface, the integrand is the potential temperature $\Theta(z)$, and the measure is $\rho_0 c_p \mathrm{d}z$, the heat capacity of each slab of water. The total is an [`integrate`](xref:api#integrix.rule.integrate) over depth, and the heat content above every depth, a profile instead of a number, is a [`cumulative`](xref:api#integrix.scan.cumulative) scan.

**Expected emission rate.** An inversion does not return one emission rate for a plume but a posterior over rates, and the number we report is its mean, with an uncertainty from the same posterior. We have the full form

$$\mathbb{E}[Q \mid \vect{y}] = \int_0^\infty Q p_\text{post}(Q \mid \vect{y}) \mathrm{d}Q$$ (eq-expected-rate)

where the domain is the positive rates $\mathbb{R}_{>0}$, the integrand is the rate $Q$ itself, and the measure is the posterior $p_\text{post}(Q \mid \vect{y}) \mathrm{d}Q$ given the observations $\vect{y}$. We can usually sample the posterior but not write it down, so integrix computes this with [`monte_carlo`](xref:api#integrix.expect.monte_carlo) on posterior samples, or with [`qmc`](xref:api#integrix.expect.qmc) when the rate is a transform of uniform variables.

## Three properties we lean on

The integral is a linear functional, and we lean on three of its properties over and over again.

1. **Linearity.** $I[a f + b g] = a I[f] + b I[g]$ for scalars $a, b$. Every rule in this section is linear in the samples too, which is what makes its gradient a vector of weights (chapter 9).
2. **Positivity.** If $f \ge 0$ then $I[f] \ge 0$. A rule with positive weights keeps this, and a rule with negative weights can return a negative column of methane from a positive profile (chapter 2).
3. **Additivity over the domain.** If $\Omega = \Omega_1 \cup \Omega_2$ with no overlap, then $I_\Omega[f] = I_{\Omega_1}[f] + I_{\Omega_2}[f]$. This is the property behind composite rules, scans and every finite-volume budget, and it is the start of the global and local thread of chapter 3.

$$\int_\Omega f(\vect{s}) \mathrm{d}\mu(\vect{s}) = \sum_{r=1}^{N_\text{tile}} \int_{\Omega^{(r)}} f(\vect{s}) \mathrm{d}\mu(\vect{s})$$ (eq-additivity)

Here $\Omega^{(r)}$, $r = 1, \ldots, N_\text{tile}$, are tiles that cover $\Omega$ without overlap: grid cells, pressure layers or time steps.

## In code

The column and the band radiance of the day, on toy data. A linear methane profile on 21 pressure levels has the exact column average 1750 ppb, and the trapezoid rule (chapter 2) gets it exactly, because the rule is exact for straight lines.

```python
import jax.numpy as jnp
from integrix import rule
from jaxtyping import Array, Float

# Shapes: N_P = 21 pressure levels; N_S = 401 wavelengths; N nodes of a rule


# user
def trapezoid_weights(nodes: Float[Array, "node"]) -> Float[Array, "node"]:
    """wₙ = (sₙ₊₁ − sₙ₋₁)/2 inside, half a spacing at each end."""
    # hₙ = sₙ₊₁ − sₙ
    widths = jnp.diff(nodes)                                 # (N,) → (N - 1,)
    # padding for the two ends, where one neighbour is missing
    zero = jnp.zeros(1, dtype=nodes.dtype)                   # → (1,)
    # rₙ = hₙ, with r_N = 0
    right = jnp.concatenate([widths, zero])                  # (N - 1,) → (N,)
    # lₙ = hₙ₋₁, with l₁ = 0
    left = jnp.concatenate([zero, widths])                   # (N - 1,) → (N,)
    # wₙ = (rₙ + lₙ)/2
    return 0.5 * (right + left)


# pₙ from the top of the atmosphere to the surface, hPa
pressure = jnp.linspace(0.0, 1013.0, 21)                     # → (N_P,)
# 300 pₙ
slope_term = 300.0 * pressure                                # (N_P,) → (N_P,)
# c(pₙ) = 1600 + 300 pₙ/p_s, ppb
methane = 1600.0 + slope_term / 1013.0                       # (N_P,) → (N_P,)
# wₙ = (pₙ₊₁ − pₙ₋₁)/2
pressure_weights = trapezoid_weights(pressure)               # (N_P,) → (N_P,)
# today (integrix)
# I = ∑ₙ wₙ c(pₙ)
column_integral = rule.integrate(pressure_weights, methane)  # → ()
# XCH₄ = I / p_s
column_average = column_integral.value / pressure[-1]        # → ()
print(f"XCH4 = {float(column_average):.1f} ppb")
```

```text
XCH4 = 1750.0 ppb
```

The band radiance is a weighted mean: a toy spectrum with one absorption line at 2300 nm, seen by a channel with a Gaussian response 5 nm wide.

```python
# Shapes: N_S = 401 wavelengths

# λₙ, nm
wavelength = jnp.linspace(2280.0, 2320.0, 401)    # → (N_S,)
# δₙ = λₙ − 2300
offset = wavelength - 2300.0                      # (N_S,) → (N_S,)
# gₙ = exp(−δₙ²/2)
line_shape = jnp.exp(-0.5 * offset**2)            # (N_S,) → (N_S,)
# L(λₙ) = 1 − 0.4 gₙ
spectrum = 1.0 - 0.4 * line_shape                 # (N_S,) → (N_S,)
# uₙ = δₙ/5
scaled = offset / 5.0                             # (N_S,) → (N_S,)
# R_c(λₙ) = exp(−uₙ²/2)
response = jnp.exp(-0.5 * scaled**2)              # (N_S,) → (N_S,)
# wₙ = (λₙ₊₁ − λₙ₋₁)/2
spectral_weights = trapezoid_weights(wavelength)  # (N_S,) → (N_S,)
# today (integrix)
# L_c = ∑ₙ wₙ Rₙ Lₙ / ∑ₙ wₙ Rₙ
band_radiance = rule.weighted_mean(               # → ()
    spectral_weights, spectrum, weight=response
)
print(f"L_c = {float(band_radiance):.3f}")
```

```text
L_c = 0.922
```

The line takes about 8 % of the light in the channel, since its width (1 nm) is a fifth of the response's. Everything in the rest of the section is about the line `rule.integrate(weights, samples)`: where the weights come from, how good they are, and what happens when we cannot write them down in advance.
