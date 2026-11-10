---
numbering:
  enumerator: "11.%s"
---

# Precision

## Closing an energy budget in float32

The Earth's energy budget is a very small number made of very large ones. About 340 W m⁻² of sunlight arrives at the top of the atmosphere, about 100 W m⁻² is reflected and about 240 W m⁻² leaves as infrared, and the imbalance that drives climate change is under 1 W m⁻². A climate model that wants to report that imbalance has to integrate fluxes of hundreds of W m⁻² over the sphere and keep the result accurate to a few hundredths of a W m⁻², and more and more of these models run in single precision on GPUs, where the unit round-off is $\varepsilon \approx 6 \times 10^{-8}$.

Every integral in this section is a weighted sum, so its rounding errors are the rounding errors of a sum. This chapter collects what integrix does about them: which dtype the weights take, how the sum is ordered, why cancellation is the real enemy, and where the float32 limits of point sets and rules hide.

## The weights take the values' dtype

integrix never casts silently. A rule sums in the dtype of its samples (promoted with the weights' dtype only when the samples are not floating point), so float64 weights against float32 samples give a float32 integral, and the caller decides the precision by choosing the dtype of the data. That is the right default for a model: its state is float32 for a reason (memory, bandwidth, the speed of the accelerator), and an integral that quietly promoted to float64 would break `jit` caches and double the memory of every batched diagnostic.

## Order of summation

Chapter 5 showed the first lesson: a long left-to-right sum rounds its running total $N$ times, with an error bound that grows like $O(\varepsilon N)$, while a pairwise sum rounds each term $O(\log_2 N)$ times. integrix always sums pairwise, so the order of the sum is not something a caller has to think about.

## Cancellation is the real enemy

The second lesson is about what we sum. The error bound of a pairwise sum is relative to $\sum_n \vert w_n f_n \vert$, not to the result $\vert \sum_n w_n f_n \vert$, so the relative error of the integral is roughly

$$\frac{\vert \hat{I}_\text{float32} - \hat{I} \vert}{\vert \hat{I} \vert} \lesssim \kappa \varepsilon \log_2 N, \quad \kappa = \frac{\sum_n \vert w_n f_n \vert}{\vert \sum_n w_n f_n \vert}$$ (eq-sum-condition)

where $\kappa \ge 1$ is the condition number of the sum. A positive integrand on positive weights has $\kappa = 1$. A small net flux made of large terms of both signs has a large $\kappa$, and so does an oscillatory integrand whose positive and negative lobes nearly cancel. The cure is to cancel before integrating: integrate the net flux field, not each flux separately.

We build three toy flux fields on a 1° latitude–longitude grid, each with a pole-to-equator shape and the last two with a wave in longitude, whose global means are 340 W m⁻² of incoming sunlight $F_\text{in}$, 100 W m⁻² reflected $F_\text{refl}$ and 239.3 W m⁻² of outgoing longwave $F_\text{olr}$, so the net $F_\text{net} = F_\text{in} - F_\text{refl} - F_\text{olr}$ is about 0.7 W m⁻². The reference is the same cell-area rule in float64, and every float32 estimate is compared against it, so only rounding is measured.

```python
import jax

jax.config.update("jax_enable_x64", True)

import einx
import jax.numpy as jnp
from integrix import adapt, rule

# Shapes: N_φ = 180 latitudes; N_ϑ = 360 longitudes

edges = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 181)  # → (N_φ + 1,)
latitude = 0.5 * (edges[1:] + edges[:-1])  # → (N_φ,)
longitude = jnp.arange(360) * 2.0 * jnp.pi / 360  # → (N_ϑ,)
# wⱼₖ = (sin φⱼ₊½ − sin φⱼ₋½) Δϑ / 4π, so ∑ wⱼₖ = 1
band_areas = jnp.diff(jnp.sin(edges))  # → (N_φ,)
area = einx.multiply("lat, lon -> lat lon", band_areas, jnp.full(360, 1.0 / 360)) / 2.0
shape = 1.0 - 3.0 * jnp.sin(latitude) ** 2  # → (N_φ,)
wave = einx.multiply("lat, lon -> lat lon", jnp.cos(latitude), jnp.cos(3.0 * longitude))


def flux(mean, pole_to_equator, wave_amplitude):
    """F(φ, ϑ) = F̄ + a (1 − 3 sin²φ) + b cos φ cos 3ϑ, in W m⁻²."""
    # user
    zonal = mean + pole_to_equator * shape  # → (N_φ,)
    return einx.add("lat, lat lon -> lat lon", zonal, wave_amplitude * wave)


incoming = flux(340.0, 100.0, 0.0)
reflected = flux(100.0, 30.0, 5.0)
outgoing = flux(239.3, 40.0, 8.0)
net = incoming - reflected - outgoing  # → (N_φ, N_ϑ)
# today (integrix)
# F̄_net = ∑ⱼₖ wⱼₖ F_net(φⱼ, ϑₖ), in float64
reference = float(rule.integrate(area, net, axis=(0, 1)).value)


def single(x):
    """The same array in float32."""
    # user
    return x.astype(jnp.float32)


area32 = single(area)
net32 = single(incoming) - single(reflected) - single(outgoing)
# today (integrix)
# integrate the net field: cancel first, then sum
direct = rule.integrate(area32, net32, axis=(0, 1)).value
# today (integrix)
# integrate each flux, then subtract: sum first, then cancel
means = [
    rule.integrate(area32, single(f), axis=(0, 1)).value
    for f in (incoming, reflected, outgoing)
]
separate = means[0] - means[1] - means[2]
print(
    f"net flux (float64):         {reference:.4f} W/m2, dtype of direct {direct.dtype}"
)
print(f"float32, net field:         error {abs(float(direct) - reference):.1e} W/m2")
print(f"float32, fluxes separately: error {abs(float(separate) - reference):.1e} W/m2")
kappa_net = float(jnp.sum(area * jnp.abs(net))) / abs(reference)
kappa_separate = float(jnp.sum(area * incoming)) / abs(reference)
print(f"condition number: net field {kappa_net:.0f}, largest flux {kappa_separate:.0f}")
```

```text
net flux (float64):         0.6992 W/m2, dtype of direct float32
float32, net field:         error 8.6e-07 W/m2
float32, fluxes separately: error 4.1e-05 W/m2
condition number: net field 34, largest flux 486
```

Both float32 estimates are far inside what the budget needs, since pairwise summation keeps $\log_2 N$ small, but integrating each flux and then subtracting is about fifty times worse than integrating the net field. The condition numbers say why: the incoming flux alone is about 490 times larger than the net, and every one of its rounding errors survives the subtraction. A left-to-right sum would also replace the $\log_2 N$ in [](#eq-sum-condition) by $N$, the 64 800 cells of the grid.

## Point sets and rules near their limits

The last place precision bites is inside the rules, at the edges of the domain, where a node can round onto a point it must not touch. integrix handles two cases explicitly.

- **Quasi-Monte Carlo points.** A Sobol or Halton point is a fraction with up to 32 bits, and a float32 holds 24, so rounding can put a point exactly on 0 or 1. A transform such as the inverse normal CDF then returns $\pm \infty$. [`qmc`](xref:api#integrix.expect.qmc) keeps only the leading bits of a Sobol point that the dtype can hold and puts the point at the midpoint of that cell, and it clips a rotated Halton point into the open interval, so every point stays strictly inside the cube.
- **Tanh–sinh near a singular end.** The double-exponential map of chapter 8 sends nodes extremely close to the endpoints, and $a + (b - a)(1 + \tanh u)/2$ rounds onto $a$ in float32 long before the weights become negligible. [`TanhSinh`](xref:api#integrix.adapt.TanhSinh) forms the distance to the nearer end directly, as $(b - a)/(1 + e^{2 \vert u \vert})$, and gives zero weight to any node that still lands on the end, so an integrand like $s^{-1/2}$ is never evaluated at its singularity.

```python
# today (integrix)
# ∫₀¹ s^(−1/2) ds = 2, in float32
singular = adapt.TanhSinh()(
    lambda s: 1.0 / jnp.sqrt(s), jnp.float32(0.0), jnp.float32(1.0)
)
print(
    f"tanh-sinh in {singular.value.dtype}: error {abs(float(singular.value) - 2.0):.1e}, "
    f"converged {bool(singular.converged)}"
)
```

```text
tanh-sinh in float32: error 2.4e-07, converged True
```

The error is one unit in the last place of 2 in float32 ($2^{-22} \approx 2.4 \times 10^{-7}$), so the singular integral is as accurate as the dtype allows.
