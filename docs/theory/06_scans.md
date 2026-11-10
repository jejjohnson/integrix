---
numbering:
  enumerator: "6.%s"
---

# Scans

## Optical depth from the top of the atmosphere

A radiative transfer model needs to know how much of the incoming sunlight survives down to each level of the atmosphere. Light travelling down from the top is attenuated by every layer above it, so the quantity at height $z$ is the optical depth from the top,

$$\tau(z) = \int_z^{\infty} \beta(z') \mathrm{d}z'$$ (eq-optical-depth)

where $\beta(z) \ge 0$ (in km⁻¹) is the absorption coefficient, the product of an absorber's density and its cross-section, and the transmittance from the top down to $z$ is $e^{-\tau(z)}$. We do not want one number here but one per level, so the integral is a function of its lower limit. The same thing happens in time: a rain gauge reports a rain rate, and the accumulated precipitation over a storm is the integral of that rate from the start of the storm up to every report.

These running integrals are scans. In integrix they are [`cumulative`](xref:api#integrix.scan.cumulative), which turns the samples of an integrand into its integral up to every node.

## The running integral

Take $N$ nodes $s_1 < \ldots < s_N$ along one axis and the samples $f_n = f(s_n)$. The running integral $F_k = \int_{s_1}^{s_k} f(s) \mathrm{d}s$ splits, by additivity ([](#eq-additivity)), into one piece per interval, and each piece is estimated with the trapezoid rule on its own two ends,

$$F_k = F_{k-1} + \Delta_k, \quad \Delta_k = \frac{1}{2} (s_k - s_{k-1}) (f_{k-1} + f_k)$$ (eq-scan)

with $F_1 = 0$ and $k = 2, \ldots, N$. So $F_k$ is the prefix sum of the interval increments $\Delta_2, \ldots, \Delta_k$.

This is the global and local thread again, in its plainest form. Each increment is local: it uses two neighbouring nodes and nothing else, so a bad sample at one level only changes the two increments that touch it. The last entry is global: $F_N$ is the integral over the whole axis, the same number that [`integrate`](xref:api#integrix.rule.integrate) gives with trapezoid weights on the same nodes. The error at node $k$ is the sum of the interval errors up to $k$, each $O(h^3)$ for a smooth integrand, so it grows along the scan but stays $O(h^2)$ at the end. The same bookkeeping, local errors that add up to a global one, comes back for time steppers in chapter 7.

## Scanning from the far end

The optical depth runs the other way, from the top down. A reversed scan sums the same increments from the last one backwards,

$$\tau_k = \tau_{k+1} + \Delta_{k+1}, \quad \tau_N = 0$$ (eq-reverse-scan)

so $\tau_k = \sum_{j > k} \Delta_j$ is the integral from node $k$ to the top. The detail that matters is that the reversed scan reverses the increments, not the nodes. If we flipped the heights and scanned forward, every interval width $s_k - s_{k-1}$ would change sign and the optical depth would come out negative. integrix keeps the nodes as they are and only changes the order of the sum.

We check both directions on an exponential atmosphere, with $\beta(z) = \beta_0 e^{-z/H_\beta}$, a scale height $H_\beta = 8$ km and $\beta_0 = 0.125$ km⁻¹, on 41 levels from the surface to 80 km. The exact optical depth is $\tau(z) = \beta_0 H_\beta (e^{-z/H_\beta} - e^{-z_\text{top}/H_\beta})$, so the surface sees $\tau(0) \approx 1$ and a transmittance of about $e^{-1}$.

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from functax import quad
from integrix import rule, scan
from jaxtyping import Array, Float

# Shapes: N = 41 levels


# zₙ, km, from the surface to the top
height = jnp.linspace(0.0, 80.0, 41)                            # → (N,)
# today (functax)
# wₙ = Δz, with Δz/2 at the ends
levels = quad.trapezoid(height)                                 # → (N,)
# s = z/H_β
scaled_height = height / 8.0                                    # → (N,)
# β(z) = β₀ exp(−s), km⁻¹
absorption = 0.125 * jnp.exp(-scaled_height)                    # → (N,)
# e_s = exp(−s)
surface_term = jnp.exp(-scaled_height)                          # → (N,)
# e_t = exp(−z_top/H_β)
top_term = jnp.exp(-80.0 / 8.0)                                 # → ()
# τ(z) = β₀ H_β (e_s − e_t)
exact = 0.125 * 8.0 * (surface_term - top_term)                 # → (N,)

# today (integrix)
# τₖ = τₖ₊₁ + Δₖ₊₁, from the top down
optical_depth = scan.cumulative(levels, absorption, reverse=True)  # → (N,)
for level in (40, 20, 10, 5, 0):
    # e = τ̂ − τ
    error = float(optical_depth[level] - exact[level])
    print(
        f"z = {float(height[level]):4.0f} km: tau = {float(optical_depth[level]):.4f}"
        f", error = {error:+.1e}"
    )
print(f"surface transmittance: {float(jnp.exp(-optical_depth[0])):.4f}")
```

```text
z =   80 km: tau = 0.0000, error = +0.0e+00
z =   40 km: tau = 0.0067, error = +3.5e-05
z =   20 km: tau = 0.0825, error = +4.3e-04
z =   10 km: tau = 0.2879, error = +1.5e-03
z =    0 km: tau = 1.0052, error = +5.2e-03
surface transmittance: 0.3660
```

The top level has zero optical depth by construction, and the error grows downwards as the interval errors add up, reaching $5.2 \times 10^{-3}$ at the surface. That is the leading term of [](#eq-euler-maclaurin), $\frac{h^2}{12} (\beta'(z_\text{top}) - \beta'(0)) \approx \frac{h^2}{12} \frac{\beta_0}{H_\beta}$ with $h = 2$ km, so the error is the $O(h^2)$ of the trapezoid rule and nothing else. The forward scan of the same samples, the reversed one and a plain rule all agree on the total, since they sum the same increments.

```python
# Shapes: N = 41 levels
# today (integrix)
# Fₖ = Fₖ₋₁ + Δₖ, from the surface up
upward = scan.cumulative(levels, absorption)                    # → (N,)
# today (integrix)
# I = ∑ₙ wₙ β(zₙ)
total = rule.integrate(levels, absorption).value                # → ()
print(f"forward, last:  {float(upward[-1]):.12f}")
print(f"reverse, first: {float(optical_depth[0]):.12f}")
print(f"rule:           {float(total):.12f}")
```

```text
forward, last:  1.005157279905
reverse, first: 1.005157279905
rule:           1.005157279905
```

## Associative scans

A scan looks sequential, since $F_k$ needs $F_{k-1}$. However, addition is associative, so the prefix sums can be computed in $O(\log_2 N)$ parallel steps instead of $N$ sequential ones (Blelloch, 1990): first add neighbouring pairs of increments, then pairs of pairs, and then fill in the partial sums on the way back down. The total work stays $O(N)$. integrix computes [](#eq-scan) with `jax.lax.associative_scan`, so a scan over a long time series or a fine vertical grid runs as a tree on an accelerator. As a side benefit, the tree adds numbers of similar size, like the pairwise sum of chapter 5.

The gradient of a scan is a scan too. Since $F_k$ is linear in the increments, $\partial F_k / \partial \Delta_j$ is $1$ for $j \le k$ and $0$ otherwise, and the vector-Jacobian product of a forward prefix sum is a reversed prefix sum. Chapter 9 uses this for adjoints.
