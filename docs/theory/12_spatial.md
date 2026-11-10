---
numbering:
  enumerator: "12.%s"
---

# Spatial integrals

## Three integrals over space

Three integrals come up again and again when a model or a satellite looks at a region rather than a point. The heat content of an ocean basin is a volume integral of temperature over water columns of different depths, so the domain is cut by the bathymetry. The mass of a methane plume is an area integral over the pixels of a plume mask, on an image with clouds in it. The flux of that plume is a line integral across the wind. All three are rules from chapter 5, with weights that come from the geometry of the grid, but space adds two things time never needed: the domain is rarely a rectangle, and the data have holes.

This chapter is the theory behind the [Spatial integrals](../guide/spatial_integrals.md) guide. We write the three integrals as weighted sums, look at masks and NaNs as a choice of domain, and then come back to global and local one last time: a finite-volume model conserves mass cell by cell, and a global budget is what those cells add up to. On this page $\Omega(x, y)$ is a methane column mass in kg m⁻², as in the IME literature, so the domains get their own letters ($\mathcal{B}$ for a basin, $P$ for a plume mask, $L$ for a line).

| Symbol | Shape, units | Meaning |
| --- | --- | --- |
| $\Theta(x, y, z)$ | °C | ocean potential temperature |
| $H(x, y)$ | m | the depth of the sea floor, positive downwards |
| $\rho_0$, $c_p$ | kg m⁻³, J kg⁻¹ K⁻¹ | reference density and heat capacity of seawater |
| $\Delta A_{ij}$, $\Delta z_{kij}$ | m², m | the area of a column and the wet thickness of a layer in it |
| $m_n$ | $\{0, 1\}$ | a mask, 1 where a sample is kept |
| $\Omega(x, y)$ | kg m⁻² | the methane column mass of a plume |
| $\vect{F}$, $\vect{n}$ | kg m⁻¹ s⁻¹, unitless | a horizontal mass flux and a unit outward normal |
| $R$, $\phi$, $\vartheta$ | m, radians | the Earth's radius, latitude and longitude |

## Volume integrals on a model grid

The heat content of a basin $\mathcal{B}$ is

$$\mathrm{OHC} = \rho_0 c_p \int_{\mathcal{B}} \int_{-H(x, y)}^{0} \Theta(x, y, z) \mathrm{d}z \mathrm{d}A$$ (eq-basin-ohc)

and on a grid with columns $(i, j)$ and layers $k$ it becomes a weighted sum with one weight per cell, $w_{kij} = \Delta z_{kij} \Delta A_{ij}$, where $\Delta z_{kij}$ is the part of layer $k$ that lies above the sea floor (a partial bottom cell keeps only its wet thickness). The weights keep the grid's shape $(N_z, N_y, N_x)$, which is Decision 10 of the spatial design (#47), and [`integrate`](xref:api#integrix.rule.integrate) contracts all three axes at once.

Land is the part of the grid outside $\mathcal{B}$. Its weights are already zero, but a model file usually fills land with NaN, and $0 \times \mathrm{NaN}$ is NaN, so folding the mask into the weights is not enough. The mask goes in as `where=`, which leaves those samples out before the multiply.

We take a bowl-shaped basin 400 km by 300 km and up to 4000 m deep, on 10 km columns and twenty 200 m layers, with a temperature that decays from 20 °C at the surface to 2 °C at depth, $\Theta(z) = 2 + 18 e^{z / 500 \text{ m}}$. Each column then has a closed-form integral, $\int_{-H}^{0} \Theta \mathrm{d}z = 2 H + 9000 (1 - e^{-H / 500})$, which is the reference.

```python
import jax

jax.config.update("jax_enable_x64", True)

import einx
import jax.numpy as jnp
import numpy as np
from integrix import rule

# Shapes: N_z = 20 layers; N_y = 30 rows; N_x = 40 columns of 10 km
n_x, n_y, n_z, spacing = 40, 30, 20, 10_000.0
x_frac = (jnp.arange(n_x) + 0.5) / n_x  # → (N_x,)
y_frac = (jnp.arange(n_y) + 0.5) / n_y  # → (N_y,)
# H(x, y): a bowl up to 4000 m deep, land where it is not positive
bowl_x = ((einx.id("x -> y x", x_frac, y=n_y) - 0.5) / 0.45) ** 2
bowl_y = ((einx.id("y -> y x", y_frac, x=n_x) - 0.5) / 0.45) ** 2
sea_floor = jnp.maximum(4000.0 * (1.0 - bowl_x - bowl_y), 0.0)  # → (N_y, N_x)
layer_edges = jnp.linspace(0.0, -4000.0, n_z + 1)  # → (N_z + 1,)
layer_tops, layer_bottoms = layer_edges[:-1], layer_edges[1:]
layer_centres = 0.5 * (layer_tops + layer_bottoms)  # → (N_z,)
# Δz_kij: the wet part of layer k above the sea floor
above_floor = einx.subtract("z, y x -> z y x", layer_tops, -sea_floor)
wet = jnp.clip(above_floor, 0.0, None)
wet = einx.minimum("z y x, z -> z y x", wet, layer_tops - layer_bottoms)
ocean = wet > 0.0  # → (N_z, N_y, N_x)
# w_kij = Δz_kij ΔA_ij
weights = wet * spacing**2  # → (N_z, N_y, N_x), m³
# Θ(z) = 2 + 18 exp(z / 500), NaN on land as in a model file
profile = 2.0 + 18.0 * jnp.exp(layer_centres / 500.0)  # → (N_z,)
temperature = jnp.where(ocean, einx.id("z -> z y x", profile, y=n_y, x=n_x), jnp.nan)
heat_capacity = 1025.0 * 3990.0  # ρ₀ c_p, J m⁻³ K⁻¹
# OHC = ρ₀ c_p ∑ₖᵢⱼ Θ_kij Δz_kij ΔA_ij, over the basin only
columns = 2.0 * sea_floor + 9000.0 * (1.0 - jnp.exp(-sea_floor / 500.0))
exact = heat_capacity * float(jnp.sum(columns)) * spacing**2
# today (integrix)
folded = rule.integrate(weights, temperature, axis=(0, 1, 2)).value
# today (integrix)
masked = rule.integrate(weights, temperature, axis=(0, 1, 2), where=ocean).value
print(f"mask folded into the weights: {float(folded)}")
print(
    f"where=ocean: OHC = {heat_capacity * float(masked):.4e} J, "
    f"relative error {heat_capacity * float(masked) / exact - 1.0:+.1e}"
)
```

```text
mask folded into the weights: nan
where=ocean: OHC = 3.6863e+21 J, relative error -5.5e-03
```

The masked integral is within about half a percent of the closed form. That error is the layers', not the mask's: $\Theta$ is sampled once per 200 m layer, at the layer's centre, while it changes by a factor of $e^{0.4}$ across the top layer, and partial bottom cells use the full layer's centre too.

## Masks and NaNs are a choice of domain

A mask does not approximate anything. With $m_n = 1$ on the samples we keep and 0 elsewhere,

$$\hat{I}_{P}[f] = \sum_{n=1}^{N} m_n w_n f_n$$ (eq-masked-rule)

is the same rule applied to the smaller domain $P$, the pixels of a plume or the wet cells of a basin. `nan_policy="omit"` chooses the domain where the data exist, so $m_n = 0$ wherever $f_n$ is NaN, and `"propagate"` says that a missing sample inside the domain makes the integral undefined. Neither choice is right in general. A plume under a cloud loses the cloud's share of its mass when the cloud is omitted, and a model column with a missing level is better reported as missing than as a column with a hole in it, which is why integrix keeps `"propagate"` as the default.

The gradient is where the implementation matters. The derivative of [](#eq-masked-rule) is $\partial \hat{I}_P / \partial f_n = m_n w_n$, so a dropped sample should get exactly 0. Multiplying by $m_n$ after the fact does not give that when $f_n$ is NaN, since the chain rule then multiplies $0$ by NaN. integrix replaces every dropped sample by 0 before it is weighted, so neither the value nor the gradient ever sees the NaN (the guarded `where` of the code review guide).

## Line integrals and fluxes

The flux of a plume across a line $L$ is

$$\Phi = \int_L \Omega(\vect{s}) \vect{u}(\vect{s}) \cdot \vect{n}(\vect{s}) \mathrm{d}\ell$$ (eq-line-flux)

with $\vect{u}$ the wind and $\vect{n}$ the line's unit normal. It has two discrete forms, and the [guide](../guide/spatial_integrals.md) runs both. The sampled form is a rule on the curve: nodes along $L$, arc-length weights $\Delta s_k$ and normals $\vect{n}_k$, with the field interpolated to the nodes. The rule is data and belongs to functax (functax#81), and integrix only contracts it (Decision 13). The strip form replaces the line by a strip of pixels of width $w$ and divides an area integral by $w$, $\Phi \approx \frac{1}{w} \sum_{i \in S} \Omega_i (\vect{u}_i \cdot \vect{n}) \Delta A_i$, which needs only a mask. The strip averages the flux over its width, which costs nothing in a steady plume, since the flux is the same across every line (the next section says why).

## Local conservation and global budgets

A finite-volume model stores cell averages and updates each one by the fluxes through its faces, so the mass of one cell $M_{ij} = \Omega_{ij} \Delta A_{ij}$ changes as

$$\frac{\mathrm{d} M_{ij}}{\mathrm{d}t} = -\sum_{\text{faces } f \text{ of } (i,j)} \vect{F}_f \cdot \vect{n}_f \Delta \ell_f + S_{ij}$$ (eq-cell-budget)

where $\vect{F}_f$ is the flux through face $f$, $\Delta \ell_f$ its length and $S_{ij}$ the source in the cell. Every interior face is shared by two cells with opposite normals, so when we sum [](#eq-cell-budget) over the grid the interior fluxes cancel in pairs, and only the boundary is left,

$$\frac{\mathrm{d} M}{\mathrm{d}t} = -\oint_{\partial \mathcal{D}} \vect{F} \cdot \vect{n} \mathrm{d}\ell + \sum_{ij} S_{ij}$$ (eq-global-budget)

which is the divergence theorem, exact in the discrete form. This is the global and local thread of the whole section in its plainest form. The budget is made locally, cell by cell, and the global budget is nothing more than their sum. It is also why every transect across a steady plume carries the whole emission: the cells between two transects gain nothing in a steady state, so what enters through one must leave through the other.

We check [](#eq-global-budget) on the transport scheme of chapter 13 (upwind advection at $U = 4$ m s⁻¹, diffusion with $K = 20$ m² s⁻¹, a source of 0.3 kg s⁻¹ in one cell, and zero outside the domain), for a random column field that is far from steady. The left side is an area integral of the tendency, and the right side is the source minus four line integrals, one along each edge of the domain.

```python
# Shapes: N_y = 24 rows; N_x = 40 columns of 50 m cells
n_rows, n_cols, cell = 24, 40, 50.0
wind, diffusivity, emission = 4.0, 20.0, 0.3
source = jnp.zeros((n_rows, n_cols)).at[12, 4].set(emission / cell**2)
column = jnp.asarray(np.random.default_rng(1).uniform(0.0, 1e-3, (n_rows, n_cols)))
padded = jnp.pad(column, 1)  # zero outside the domain
west, east = padded[1:-1, :-2], padded[1:-1, 2:]
south, north = padded[:-2, 1:-1], padded[2:, 1:-1]
# ∂Ω/∂t = −U ∂Ω/∂x + K ∇²Ω + S, cell by cell
tendency = (
    -wind * (column - west) / cell
    + diffusivity * (west + east + south + north - 4.0 * column) / cell**2
    + source
)  # → (N_y, N_x)
# today (integrix)
# dM/dt = ∑ᵢⱼ (∂Ω/∂t)ᵢⱼ ΔA
mass_change = rule.integrate(jnp.full((n_rows, n_cols), cell**2), tendency, axis=(0, 1))
# outward flux through each edge: advection leaves through the east edge only,
# diffusion through all four (K Ω / Δx against the zero outside)
edge_rows = jnp.full(n_rows, cell)  # Δℓ along the east and west edges
edge_cols = jnp.full(n_cols, cell)  # Δℓ along the north and south edges
outflow = 0.0
for edge_values, lengths in (
    (wind * column[:, -1] + diffusivity * column[:, -1] / cell, edge_rows),
    (diffusivity * column[:, 0] / cell, edge_rows),
    (diffusivity * column[-1, :] / cell, edge_cols),
    (diffusivity * column[0, :] / cell, edge_cols),
):
    # today (integrix)
    # ∫ along the edge of F · n dℓ
    outflow += float(rule.integrate(lengths, edge_values).value)
print(f"area integral of the tendency: {float(mass_change.value):+.12f} kg/s")
print(f"source minus boundary flux:    {emission - outflow:+.12f} kg/s")
```

```text
area integral of the tendency: -2.942014828576 kg/s
source minus boundary flux:    -2.942014828576 kg/s
```

The two sides agree to round-off, although the field is random and the domain is losing mass fast: no interior flux survives the sum.

## Areas on the sphere

On the sphere, the area of a latitude–longitude cell is exact in closed form, $\Delta A = R^2 \Delta \vartheta (\sin \phi_{j+1/2} - \sin \phi_{j-1/2})$, and these areas add up to $4 \pi R^2$ exactly, the global and local thread again: local weights that close a global total. The common shortcut $R^2 \cos \phi_j \Delta \phi \Delta \vartheta$ at the cell centre is a midpoint rule for the same integral and does not.

```python
# Shapes: N_φ latitude bands; N_ϑ = 2 N_φ longitudes
radius = 6.371e6
sphere = 4.0 * np.pi * radius**2
for degrees in (5.0, 1.0):
    n_lat, n_lon = int(180 / degrees), int(360 / degrees)
    edges = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, n_lat + 1)  # → (N_φ + 1,)
    centres = 0.5 * (edges[1:] + edges[:-1])  # → (N_φ,)
    step = 2.0 * jnp.pi / n_lon
    # ΔA = R² Δϑ (sin φⱼ₊½ − sin φⱼ₋½)
    exact_areas = radius**2 * step * jnp.diff(jnp.sin(edges))
    # ΔA ≈ R² cos φⱼ Δφ Δϑ
    midpoint_areas = radius**2 * step * jnp.cos(centres) * (edges[1] - edges[0])
    exact_total = n_lon * float(jnp.sum(exact_areas))
    midpoint_total = n_lon * float(jnp.sum(midpoint_areas))
    print(
        f"{degrees:.0f} deg: exact areas {exact_total / sphere - 1.0:+.1e}, "
        f"cos(phi) areas {midpoint_total / sphere - 1.0:+.1e}"
    )
```

```text
5 deg: exact areas +2.2e-16, cos(phi) areas +3.2e-04
1 deg: exact areas +2.2e-16, cos(phi) areas +1.3e-05
```

The exact areas close the sphere to round-off at every resolution. The midpoint areas overcount by a relative $O(\Delta \phi^2)$, about $3 \times 10^{-4}$ on a 5° grid and $10^{-5}$ on a 1° grid, which is small next to most physics but large next to the round-off of a budget that is meant to close (chapter 11).
