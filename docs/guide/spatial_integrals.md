# Spatial integrals

Most of the integrals a plume or a column needs are over space: the mass of a plume over an image, the flux of methane across a line downwind of the source, the column of a gas over the vertical levels of a model. integrix has no separate spatial API for them. They are all [`integrate`](xref:api#integrix.rule.integrate) with the right weights, plus two arguments that space makes necessary: `where=`, because we rarely want the whole image, and `nan_policy=`, because satellite images have holes. This page works through the four common cases on one synthetic plume. The theory behind them is in the [mathematical reference](../theory/01_integral.md).

## A synthetic plume

We use a Gaussian plume downwind of a point source, with emission rate $Q = 0.5$ kg s⁻¹ and wind speed $U = 4$ m s⁻¹ along $x$. Its column enhancement, in kg m⁻², is

$$\Omega(x, y) = \frac{Q}{U \sqrt{2 \pi} \sigma(x)} e^{-y^2 / 2 \sigma(x)^2}, \quad \sigma(x) = 20 + 0.1 x$$ (eq-gaussian-plume)

with $x$, $y$ and the plume width $\sigma(x)$ in metres. Every transect across the wind carries the whole emission, $U \int \Omega(x, y) \mathrm{d}y = Q$, which gives the line integrals below an exact answer to aim for. We sample it on an image of 20 m pixels, 100 columns along the wind and 60 rows across it, and put clouds (NaN) over about 10 % of the pixels.

```python
import jax

jax.config.update("jax_enable_x64", True)

import einx
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from integrix import rule
from jaxtyping import Array, Float

# Shapes: N_y = 60 rows; N_x = 100 columns of 20 m pixels; K transect nodes
# emission rate in kg/s and wind speed in m/s
emission, wind = 0.5, 4.0
# pixel width in m
pixel = 20.0
# xᵢ = 10 + Δx i: pixel centres along the wind, in m
x = 10.0 + pixel * jnp.arange(100)                                  # → (N_x,)
# yⱼ = −590 + Δx j: pixel centres across the wind, in m
y = -590.0 + pixel * jnp.arange(60)                                 # → (N_y,)


# user
def plume(
    x: Float[Array, "..."], y: Float[Array, "..."]
) -> Float[Array, "..."]:
    """Ω(x, y) = Q / (U √(2π) σ(x)) exp(−y² / 2σ(x)²), in kg m⁻²."""
    # σ(x) = 20 + 0.1 x
    width = 20.0 + 0.1 * x                                          # → (...)
    # A = Q / (U √(2π) σ)
    amplitude = emission / (wind * jnp.sqrt(2.0 * jnp.pi) * width)  # → (...)
    # g = exp(−y² / 2σ²)
    falloff = jnp.exp(-(y**2) / (2.0 * width**2))                   # → (...)
    # Ω = A g
    column = amplitude * falloff                                    # → (...)
    return column


# x̃ᵢⱼ = xⱼ
x_grid = einx.id("x -> y x", x, y=60)                               # → (N_y, N_x)
# ỹᵢⱼ = yᵢ
y_grid = einx.id("y -> y x", y, x=100)                              # → (N_y, N_x)
# Ωᵢⱼ = Ω(x̃ᵢⱼ, ỹᵢⱼ) on the image, rows y and columns x
enhancement = plume(x_grid, y_grid)                                 # → (N_y, N_x)
# a seeded generator so the clouds are reproducible
rng = np.random.default_rng(0)
# uᵢⱼ ~ 𝒰(0, 1)
draws = rng.uniform(size=(60, 100))                                 # → (N_y, N_x)
# cᵢⱼ = 1 where uᵢⱼ < 0.1: about 10 % of the pixels
clouds = jnp.asarray(draws < 0.1)                                   # → (N_y, N_x)
# Ω̃ = NaN under a cloud, Ω elsewhere
cloudy = jnp.where(clouds, jnp.nan, enhancement)                    # → (N_y, N_x)
# ΔA = Δx², in m²
pixel_area = jnp.full((60, 100), pixel**2)                          # → (N_y, N_x)
```

## The integrated mass enhancement

The integrated mass enhancement (IME) is the mass in the plume, $\mathrm{IME} = \int_P \Omega \mathrm{d}A \approx \sum_{i \in P} \Omega_i \Delta A_i$ over the pixels $P$ of a plume mask. The weights are the pixel areas, kept in the image's shape, and the mask and the clouds go in as arguments. With `nan_policy="omit"` the cloudy pixels are left out, which is `np.nansum` over the mask, but traceable and differentiable.

```python
# Shapes: N_y = 60 rows; N_x = 100 columns; N_P pixels in the plume mask
# P = {Ω > 10⁻⁵ kg m⁻²} in the clear-sky field
in_plume = enhancement > 1e-5                                 # → (N_y, N_x)
# today (integrix)
# IME = ∑ over P of Ωᵢ ΔAᵢ, cloudy pixels left out
ime = rule.integrate(                                         # → ()
    pixel_area, cloudy, axis=(0, 1), where=in_plume, nan_policy="omit"
).value
# Ωᵢ on the plume pixels, as a NumPy array
masked_field = np.asarray(cloudy)[in_plume]                   # → (N_P,)
# ΔAᵢ on the plume pixels, as a NumPy array
masked_area = np.asarray(pixel_area)[in_plume]                # → (N_P,)
# IME = ∑ over P of Ωᵢ ΔAᵢ, skipping NaN
check = np.nansum(masked_field * masked_area)                 # → ()
# today (integrix)
# IME = ∑ over P of Ωᵢ ΔAᵢ, clear sky
clear = rule.integrate(                                       # → ()
    pixel_area, enhancement, axis=(0, 1), where=in_plume
).value
print(
    f"IME {float(ime):.2f} kg (np.nansum {check:.2f}), clear sky {float(clear):.2f} kg"
)
# today (integrix)
# IME = ∑ over P of Ωᵢ ΔAᵢ, NaN propagated
propagated = rule.integrate(                                  # → ()
    pixel_area, cloudy, axis=(0, 1), where=in_plume
).value
print(f"with the default nan_policy: {float(propagated)}")
```

```text
IME 225.99 kg (np.nansum 225.99), clear sky 248.39 kg
with the default nan_policy: nan
```

Leaving the clouds out is not the same as filling them: the IME loses the mass under the clouds, about 9 % here, and a retrieval has to decide whether to accept that or to fill the gaps first. The default `nan_policy="propagate"` refuses to guess and returns NaN.

The gradient of the IME with respect to the field is the pixel area on the kept pixels and exactly 0 on the masked and cloudy ones, with no NaN anywhere, so the IME can sit inside a differentiable retrieval.

```python
# Shapes: N_y = 60 rows; N_x = 100 columns


# user
def plume_mass(field: Float[Array, "row col"]) -> Float[Array, ""]:
    """IME(Ω), cloudy pixels left out."""
    # today (integrix)
    # IME = ∑ over P of Ωᵢ ΔAᵢ, cloudy pixels left out
    mass = rule.integrate(                                    # → ()
        pixel_area, field, axis=(0, 1), where=in_plume, nan_policy="omit"
    ).value
    return mass


# ∂IME/∂Ωᵢ
gradient = jax.grad(plume_mass)(cloudy)                       # → (N_y, N_x)
# kept = P ∧ ¬cloud
kept = in_plume & ~clouds                                     # → (N_y, N_x)
print(
    f"finite: {bool(jnp.all(jnp.isfinite(gradient)))}, "
    f"kept pixels {float(gradient[kept].min()):.0f} m2, "
    f"others {float(jnp.abs(gradient[~kept]).max()):.0f}"
)
```

```text
finite: True, kept pixels 400 m2, others 0
```

## The cross-sectional flux, strip form

A mass balance across the plume gives the emission rate directly: every line across the wind carries $\Phi = U \int \Omega \mathrm{d}y = Q$. The strip form, which plumax uses today, approximates the line by a strip of pixels $S$ of width $w$ along the wind, $\Phi \approx \frac{U}{w} \sum_{i \in S} \Omega_i \Delta A_i$, so it is the IME of a strip and needs nothing beyond `where=`.

```python
# Shapes: N_y = 60 rows; N_x = 100 columns
# d = |x − 1010 m|, the distance of each column from the strip centre
distance = jnp.abs(x - 1010.0)                                # → (N_x,)
# sⱼ = 1 where dⱼ < 31 m: a strip of three columns, w = 60 m, 1 km downwind
strip_columns = distance < 31.0                               # → (N_x,)
# Sᵢⱼ = sⱼ
in_strip = einx.id("x -> y x", strip_columns, y=60)           # → (N_y, N_x)
# w = 3 Δx
strip_width = 3 * pixel
# each field with the NaN policy that goes with it
cases = (
    ("clear sky", enhancement, "propagate"),
    ("cloudy, omitted", cloudy, "omit"),
)
for name, field, policy in cases:
    # today (integrix)
    # IME_S = ∑ over S of Ωᵢ ΔAᵢ
    strip_mass = rule.integrate(                              # → ()
        pixel_area, field, axis=(0, 1), where=in_strip, nan_policy=policy
    ).value
    # Φ ≈ (U / w) IME_S
    flux = wind * float(strip_mass) / strip_width
    print(f"strip, {name:15s}: Q = {flux:.4f} kg/s (true {emission})")
```

```text
strip, clear sky      : Q = 0.5000 kg/s (true 0.5)
strip, cloudy, omitted: Q = 0.4560 kg/s (true 0.5)
```

On a clear sky the strip recovers the emission to the four digits printed. Under clouds it loses the cloudy pixels' share of the flux (about 9 % again), and a single strip has no way to know.

## The cross-sectional flux, sampled form

The sampled form integrates along the line itself: nodes $\vect{s}_k$ on a transect, arc-length weights $\Delta s_k$ and unit normals $\vect{n}_k$, so that

$$\Phi = \int_L \Omega (\vect{u} \cdot \vect{n}) \mathrm{d}\ell \approx \sum_{k=1}^{K} \Delta s_k \Omega(\vect{s}_k) (\vect{u}_k \cdot \vect{n}_k)$$ (eq-sampled-flux)

with $\vect{u}$ the wind. The rule on the curve (nodes, weights and normals) belongs to functax (functax#81), the field at the nodes comes from functax's interpolation, and integrix only contracts: it adds no line primitive. Until functax#81 lands we build the transect rule by hand, and we sample the plume analytically at the nodes instead of interpolating the image.

```python
# Shapes: K transect nodes (5, 9, 17 or 33); 2 coordinates (x, y)


# user
class Transect(eqx.Module):
    """A rule on a straight segment: a stand-in for functax#81's curve rules."""

    # points along the segment
    nodes: Float[Array, "node 2"]
    # arc-length weights Δs
    weights: Float[Array, "node"]
    # unit normals
    normals: Float[Array, "node 2"]


# user
def straight_transect(
    start: Float[Array, "2"], end: Float[Array, "2"], n_nodes: int
) -> Transect:
    """Trapezoid weights in arc length and the normal to the left of the direction."""
    # tₖ = k / (K − 1)
    fraction = jnp.linspace(0.0, 1.0, n_nodes)                        # → (K,)
    # d = end − start
    direction = end - start                                           # → (2,)
    # oₖ = tₖ d
    offsets = einx.multiply("k, d -> k d", fraction, direction)       # → (K, 2)
    # sₖ = start + tₖ d
    nodes = start + offsets                                           # → (K, 2)
    # dᵢ²
    squares = direction**2                                            # → (2,)
    # d₁² + d₂²
    total = jnp.sum(squares)                                          # → ()
    # ℓ = ‖d‖
    length = jnp.sqrt(total)                                          # → ()
    # Δs = ℓ / (K − 1)
    spacing = length / (n_nodes - 1)                                  # → ()
    # the first and last node
    ends = jnp.array([0, -1])                                         # → (2,)
    # Δsₖ = Δs, with Δs / 2 at the two ends
    weights = jnp.full(n_nodes, spacing).at[ends].set(spacing / 2)    # → (K,)
    # d⊥ = (−d₂, d₁)
    perpendicular = jnp.array([-direction[1], direction[0]])          # → (2,)
    # n = d⊥ / ℓ
    normal = perpendicular / length                                   # → (2,)
    # nₖ = n
    normals = einx.id("d -> k d", normal, k=n_nodes)                  # → (K, 2)
    return Transect(nodes, weights, normals)


for n_nodes in (5, 9, 17, 33):
    # from (1 km, 1 km) to (1 km, −1 km): the normal points downwind, +x
    start, end = jnp.array([1000.0, 1000.0]), jnp.array([1000.0, -1000.0])
    # (s, Δs, n) on the segment
    line = straight_transect(start, end, n_nodes)
    # xₖ and yₖ of the nodes
    node_x = line.nodes[:, 0]                                         # → (K,)
    node_y = line.nodes[:, 1]                                         # → (K,)
    # Ωₖ = Ω(xₖ, yₖ)
    at_nodes = plume(node_x, node_y)                                  # → (K,)
    # u · n with u = (U, 0)
    normal_wind = wind * line.normals[:, 0]                           # → (K,)
    # fₖ = Ωₖ (uₖ · nₖ)
    integrand = at_nodes * normal_wind                                # → (K,)
    # today (integrix)
    # Φ = ∑ₖ Δsₖ fₖ
    flux = rule.integrate(line, integrand).value                      # → ()
    print(f"K = {n_nodes:2d}: Q = {float(flux):.10f} kg/s")
```

```text
K =  5: Q = 0.8314120967 kg/s
K =  9: Q = 0.5105892862 kg/s
K = 17: Q = 0.5000000126 kg/s
K = 33: Q = 0.5000000000 kg/s
```

Five nodes 500 m apart cannot see a plume 240 m wide, but from there the sampled form converges very fast: the error falls from 2 % at 9 nodes to $10^{-8}$ at 17 and below the printed digits at 33. The plume decays to zero long before the ends of the transect, so the trapezoid rule along it behaves like the periodic trapezoid rule of the theory's [Global and local](../theory/03_global_and_local.md) chapter.

## A column that propagates NaNs

The last case runs the other way. A model column integrated over its vertical levels should be NaN if any level is NaN, since a column with a missing level is not a column (xrops' `column_integral` makes that choice on purpose). That is the default `nan_policy="propagate"`, and `where=` can still drop levels that are known to be invalid, such as those below the surface.

```python
# Shapes: 3 columns × N_z = 5 levels
# level thicknesses in hPa
level_widths = jnp.array([50.0, 100.0, 100.0, 100.0, 50.0])   # → (N_z,)
# ρ_cl: one NaN in the middle column
density = jnp.array(                                          # → (3, N_z)
    [
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, jnp.nan, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
    ]
)
# m_cl = 0 on the last column's lowest level, below the surface
above_ground = jnp.ones((3, 5), bool).at[2, 4].set(False)     # → (3, N_z)
# today (integrix)
# a NaN at any kept level makes its column NaN
# I_c = ∑ₗ m_cl wₗ ρ_cl
columns = rule.integrate(                                     # → (3,)
    level_widths, density, where=above_ground
).value
print("columns:", [float(column) for column in columns])
```

```text
columns: [400.0, nan, 350.0]
```

The middle column is NaN because one of its levels is, and the last column leaves out its level below the surface, which is a different thing from a missing value.
