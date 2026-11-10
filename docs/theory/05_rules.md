---
numbering:
  enumerator: "5.%s"
---

# Rules

## A global mean on a latitude–longitude grid

Every climate report has a global mean surface temperature in it, and it is easy to get it wrong by a few kelvin. The field comes on a regular latitude–longitude grid, and a plain average of the grid values counts a cell next to the pole as much as a cell on the equator, even though the polar cell is a tiny sliver of the sphere. The fix is the area element of [](#eq-global-mean): each value is weighted by the area of its cell, which falls like $\cos \phi$ toward the poles. So the global mean is a rule, with weights that come from the geometry of the grid, and it is the first operation integrix offers.

In this chapter we look at rules over one axis and several, at the two ways a grid on the sphere can be weighted (global and local), at weighted means, and at one detail that matters for long sums in single precision.

## Rules over several axes

A rule over one axis is [](#eq-rule). A field on a grid has several axes, and when the domain is a product $\Omega = \Omega_1 \times \Omega_2$ and the measure is a product too, $\mathrm{d}\mu = \mathrm{d}\mu_1 \mathrm{d}\mu_2$, a rule on each axis gives a rule on the grid,

$$\hat{I}[f] = \sum_{i=1}^{N_1} \sum_{j=1}^{N_2} w^{(1)}_i w^{(2)}_j f(s^{(1)}_i, s^{(2)}_j)$$ (eq-tensor-rule)

where $w^{(1)}_i$ and $w^{(2)}_j$ are the weights of the two rules, with $N_1$ and $N_2$ nodes. This is a tensor-product rule, and its weights are the outer product of the two weight vectors. The sphere in latitude and longitude is such a product: the measure $\cos \phi \mathrm{d}\phi \mathrm{d}\vartheta$ splits into a latitude factor $\cos \phi \mathrm{d}\phi$ and a longitude factor $\mathrm{d}\vartheta$.

There are a few ways to weight each factor, and they put the sphere on either side of chapter 3.

- **Longitude** is periodic, so the plain rule with equal weights $2 \pi / N_\vartheta$ on equally spaced longitudes is the trapezoid rule on a circle, which is already spectral ([](#eq-euler-maclaurin)). There is nothing to gain from anything else.
- **Latitude, local.** A regular grid stores cell values at the centres of $N_\phi$ equal latitude bands. Weighting each by the exact area of its band, $\sin \phi_{j+1/2} - \sin \phi_{j-1/2}$ between the band's edges, is a local rule with error $O(h^2)$, and it is what a finite-volume model does.
- **Latitude, global.** A spectral model puts its latitudes at the Gauss–Legendre nodes in $\mu = \sin \phi$, with the Gauss weights. The measure $\cos \phi \mathrm{d}\phi$ is $\mathrm{d}\mu$, so this is the Gauss rule of chapter 3, exact for polynomials of degree $2 N_\phi - 1$ in $\mu$.
- **Latitude, unweighted.** Equal weights on equal latitude bands, which is what a plain average does. It is not a rule for the sphere at all, since it integrates $\mathrm{d}\phi$ instead of $\cos \phi \mathrm{d}\phi$.

We check all three on a toy temperature field with a pole-to-equator gradient and a wave in longitude, $T(\phi, \vartheta) = 288 - 30 \sin^2 \phi + 5 \cos \phi \cos \vartheta$ (in K). The wave averages to zero around every latitude circle, and $\sin^2 \phi$ averages to $1/3$ over the sphere, so the exact global mean is $\bar{T} = 278$ K. integrix takes a tuple of rules as a tensor rule, one per axis. Until functax's rules land (functax#44), a rule is any object with `nodes` and `weights`, so we define a two-field module for it.

```python
import jax

jax.config.update("jax_enable_x64", True)

import einx
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from integrix import rule

# Shapes: N_φ latitudes; N_ϑ = 2 N_φ longitudes


class Rule(eqx.Module):
    """Nodes and weights: a stand-in for functax's rules (functax#44)."""

    # user
    nodes: jax.Array
    weights: jax.Array


def temperature(latitude, longitude):
    """T(φ, ϑ) = 288 − 30 sin²φ + 5 cos φ cos ϑ, in K."""
    # user
    zonal = 288.0 - 30.0 * jnp.sin(latitude) ** 2  # → (N_φ,)
    wave = einx.multiply("lat, lon -> lat lon", jnp.cos(latitude), jnp.cos(longitude))
    return einx.add("lat, lat lon -> lat lon", zonal, 5.0 * wave)  # → (N_φ, N_ϑ)


labels = ("unweighted", "cell areas", "Gauss")
print("N_φ  " + "  ".join(f"{label:>11}" for label in labels))
for n_lat in (4, 8, 16, 32):
    n_lon = 2 * n_lat
    # ϑₖ = 2π k / N_ϑ, wₖ = 2π / N_ϑ
    longitude = 2.0 * jnp.pi * jnp.arange(n_lon) / n_lon  # → (N_ϑ,)
    longitude_rule = Rule(longitude, jnp.full(n_lon, 2.0 * jnp.pi / n_lon))
    # local: band centres, wⱼ = sin φⱼ₊½ − sin φⱼ₋½
    edges = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, n_lat + 1)  # → (N_φ + 1,)
    centres = 0.5 * (edges[1:] + edges[:-1])  # → (N_φ,)
    cell_rule = Rule(centres, jnp.diff(jnp.sin(edges)))
    # global: Gauss–Legendre in μ = sin φ
    mu_nodes, mu_weights = np.polynomial.legendre.leggauss(n_lat)
    gauss_rule = Rule(jnp.arcsin(mu_nodes), jnp.asarray(mu_weights))
    # a plain average of the grid values, for comparison
    unweighted = jnp.mean(temperature(centres, longitude))  # → ()
    errors = [float(unweighted) - 278.0]
    for latitude_rule in (cell_rule, gauss_rule):
        field = temperature(latitude_rule.nodes, longitude)  # → (N_φ, N_ϑ)
        # today (integrix)
        # T̄ = (1/4π) ∑ⱼ ∑ₖ wⱼ wₖ T(φⱼ, ϑₖ)
        total = rule.integrate((latitude_rule, longitude_rule), field, axis=(0, 1))
        errors.append(float(total.value) / (4.0 * np.pi) - 278.0)
    print(f"{n_lat:3d}  " + "  ".join(f"{error:+11.2e}" for error in errors))
```

```text
N_φ   unweighted   cell areas        Gauss
  4    -5.00e+00    -6.07e-01    +0.00e+00
  8    -5.00e+00    -1.34e-01    +0.00e+00
 16    -5.00e+00    -3.24e-02    +0.00e+00
 32    -5.00e+00    -8.05e-03    -5.68e-14
```

The plain average is $5$ K too cold at every resolution, because it counts the cold polar bands as much as the tropics, and refining the grid does nothing for it. The cell-area rule converges like $h^2$, losing about a factor of four per doubling, as a local rule should. The Gauss rule is exact from the start, up to round-off. That is not luck: the field is a polynomial of degree 2 in $\mu$ times a wave in longitude, and a Gaussian grid with $N_\phi$ latitudes integrates every such product up to degree $2 N_\phi - 1$ exactly, which is why spectral models compute their transforms on it.

## Batching

The axes that a rule does not integrate are batch axes, and integrix keeps them. A field with a time axis, $T(t, \phi, \vartheta)$, integrated over `axis=(1, 2)` returns one global mean per time step, and an ensemble axis gives one per member. In the same way, the tensor rule above never builds the outer product of the weights explicitly in user code: integrix forms $w^{(1)}_i w^{(2)}_j$ once and contracts it with the samples over the integrated axes.

## Weighted means

The band radiance of [](#eq-band-radiance) is the other common rule in Earth observation. A multispectral sensor has a handful of bands, each with its own response $R_b(\lambda)$, and each band radiance is the mean of the spectrum under that response,

$$\bar{f} = \frac{\sum_{n=1}^{N} w_n R_n f_n}{\sum_{n=1}^{N} w_n R_n}$$ (eq-weighted-mean)

where $w_n$ are the quadrature weights in wavelength, $R_n = R_b(\lambda_n)$ the response and $f_n = L(\lambda_n)$ the spectrum. The point of [`weighted_mean`](xref:api#integrix.rule.weighted_mean) is that both sums use the same $w_n$ and the same summation, so a rule that is a little off in the numerator is off in the same way in the denominator, and the two errors cancel for a flat spectrum. A response stacked per band, with the shape of the values, gives every band in one call.

```python
# Shapes: N_λ = 601 wavelengths; N_b = 3 bands
wavelength = jnp.linspace(2200.0, 2400.0, 601)  # → (N_λ,)
# L(λ) = 1 − 0.4 exp(−(λ − 2300)²/2): one line at 2300 nm
spectrum = 1.0 - 0.4 * jnp.exp(-0.5 * (wavelength - 2300.0) ** 2)
# R_b(λ) = exp(−(λ − λ_b)²/(2 · 10²)), bands at 2250, 2300 and 2350 nm
band_centres = jnp.array([2250.0, 2300.0, 2350.0])  # → (N_b,)
offset = einx.subtract("band, wl -> band wl", band_centres, wavelength)
response = jnp.exp(-0.5 * (offset / 10.0) ** 2)  # → (N_b, N_λ)
spectra = einx.id("wl -> band wl", spectrum, band=3)  # → (N_b, N_λ)
spacing = wavelength[1] - wavelength[0]
weights = jnp.full(601, spacing).at[jnp.array([0, -1])].set(spacing / 2)
# today (integrix)
# L_b = ∑ₙ wₙ R_b(λₙ) L(λₙ) / ∑ₙ wₙ R_b(λₙ)
bands = rule.weighted_mean(weights, spectra, weight=response)  # → (N_b,)
print("L_b =", [round(float(band), 4) for band in bands])
```

```text
L_b = [1.0, 0.9602, 1.0]
```

Only the middle band sees the line, and it loses about 4 % of its light, half as much as the 5 nm channel of chapter 1, since this band is twice as wide.

## Summing in single precision

The last detail of a rule is the sum itself. A left-to-right sum keeps a running total and adds each term to it, so the total is rounded $N$ times, and once the total is large the small terms lose most of their bits. The rounding error of such a sum can grow like $O(\varepsilon N)$, where $\varepsilon$ is the unit round-off ($\varepsilon \approx 6 \times 10^{-8}$ in float32). A pairwise sum adds neighbours, then pairs of pairs, so each term is rounded only $O(\log_2 N)$ times and the error bound is $O(\varepsilon \log_2 N)$ (Higham, 2002). In float64 the difference rarely matters, but a float32 sum over a long time series, a fine spectral grid or a large image can be visibly wrong. integrix always sums pairwise.

We add $2^{20}$ copies of $0.1$ in float32, once with a left-to-right `jax.lax.scan` and once with [`integrate`](xref:api#integrix.rule.integrate) and unit weights. The exact answer, in float32's nearest value of $0.1$, is printed last.

```python
# Shapes: N = 2²⁰ terms
values = jnp.full(2**20, 0.1, dtype=jnp.float32)  # → (N,)


def add(total, value):
    """One step of a left-to-right sum."""
    # user
    return total + value, None


# S = ((v₁ + v₂) + v₃) + …
left_to_right, _ = jax.lax.scan(add, jnp.float32(0.0), values)
# today (integrix)
# S = ∑ₙ 1 · vₙ, summed pairwise
pairwise = rule.integrate(jnp.ones(2**20, dtype=jnp.float32), values).value
exact = 2**20 * float(np.float32(0.1))
print(f"left to right: {float(left_to_right):.1f}")
print(f"pairwise:      {float(pairwise):.1f}")
print(f"exact:         {exact:.1f}")
```

```text
left to right: 105891.8
pairwise:      104857.6
exact:         104857.6
```

The left-to-right sum is about 1 % too large, and the pairwise sum is exact to the last digit printed. Chapter 11 comes back to precision with a budget that has to close.
