---
numbering:
  enumerator: "3.%s"
---

# Global and local

## Two models, two kinds of rule

A spectral atmosphere model and a finite-volume ocean model can sit in the same coupled system and still integrate in two very different ways. The atmosphere stores its fields as spherical harmonics and evaluates them on a Gaussian grid: the latitudes are the nodes of a Gauss–Legendre rule in $\mu = \sin \phi$, and the longitudes are equally spaced. Every node and every weight depends on the whole grid, and changing the resolution moves all of them. The ocean stores the average of each tracer over each cell, and the integral of a tracer is the sum of cell averages times cell volumes. Each weight belongs to one cell, and refining one region leaves the other cells alone.

These are the two families of rules, and almost every choice in integrix (and a fair amount in the models built on it) sits on one side or the other.

- A **global rule** uses one set of nodes and weights over the whole domain: Gauss–Legendre, Clenshaw–Curtis, the trapezoid rule on periodic data.
- A **local rule**, or composite rule, cuts the domain into tiles with [](#eq-additivity) and applies a low-degree rule on each one: the composite trapezoid and Simpson rules, the midpoint rule on cell averages.

The difference is not only where the nodes are. It changes how fast the error falls, what a single bad sample does, whether a budget closes cell by cell, and what the gradient looks like. This chapter goes through the first three; the gradient comes back in chapter 9.

## Global rules: spectral convergence

If we may place the nodes, the $N$ weights and $N$ nodes of [](#eq-rule) are $2N$ unknowns, and we can ask for $2N$ moment equations [](#eq-moments) instead of $N$. On $[-1, 1]$ with the ordinary length, the solution is the Gauss–Legendre rule: the nodes are the roots of the Legendre polynomial of degree $N$, the weights are all positive, and the degree of exactness is $m^\star = 2N - 1$.

Exactness for polynomials says little about a function that is not one, so the useful result is about analytic functions. If $f$ extends analytically to the interior of the Bernstein ellipse $E_\rho$ (the ellipse with foci $\pm 1$ and semi-axes summing to $\rho > 1$) and is bounded there, the error of the $N$-point Gauss rule falls like

$$\vert E_N[f] \vert = O(\rho^{-2N})$$ (eq-gauss-rate)

which is spectral, or geometric, convergence: each extra node multiplies the error by a fixed factor $\rho^{-2}$. Trefethen (2008) gives the constant and shows that Clenshaw–Curtis, whose nodes are the Chebyshev extrema, converges at almost the same rate in practice, even though its degree is only $N - 1$. The larger the region where $f$ is smooth, the larger $\rho$, so a field that varies on the scale of the domain is integrated to machine precision with a handful of nodes.

## Local rules: algebraic convergence

A composite rule on $N_\text{tile}$ tiles of width $h$ has an error that is the sum of the tile errors. For the composite trapezoid rule on $[a, b]$,

$$E_N[f] = -\frac{(b - a) h^2}{12} f''(\xi)$$ (eq-trapezoid-error)

for some $\xi \in [a, b]$, and for the composite Simpson rule the power is $h^4$. In general a composite rule built on a tile rule of degree $m^\star$ has $E_N = O(h^q)$ with $q = m^\star + 1$ (the trapezoid rule has $m^\star = 1$ and $q = 2$; Simpson's has $m^\star = 3$ and $q = 4$). This is algebraic convergence, and $q$ is the order of the rule: halving the tile width divides the error by $2^q$, whatever the integrand, as long as it has $q$ derivatives. A smoother integrand does not make it converge faster.

So the trade is as follows. A global rule is very fast for smooth integrands and gives nothing extra for rough ones. A local rule is never very fast but it does not care where the roughness is, since each tile only sees its own part of the integrand.

## The trapezoid rule on a circle

There is one case where a local rule is also a global one. The Euler–Maclaurin formula writes the error of the composite trapezoid rule $T_h$ as a series in $h^2$,

$$T_h[f] - I[f] = \sum_{k=1}^{K} \frac{B_{2k} h^{2k}}{(2k)!} \left( f^{(2k-1)}(b) - f^{(2k-1)}(a) \right) + R_K$$ (eq-euler-maclaurin)

where $B_{2k}$ are the Bernoulli numbers ($B_2 = 1/6$, which gives back [](#eq-trapezoid-error)) and $R_K$ is a remainder. If $f$ is periodic on $[a, b]$, every bracket vanishes, and for a periodic $f$ that is analytic in a strip of half-width $\alpha$ about the real axis, the error falls like $O(e^{-\alpha N})$ (Trefethen and Weideman, 2014). So equally spaced samples of a periodic field (every longitude circle on every global grid) are already a spectral rule, and nothing beats the plain sum there.

We check it on $\int_0^{2 \pi} e^{\cos \vartheta} \mathrm{d}\vartheta = 2 \pi I_0(1)$, a smooth function of longitude $\vartheta$, where $I_0$ is the modified Bessel function of order zero.

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from integrix import rule
from scipy.special import i0

# Shapes: N nodes on a circle of longitude
exact_circle = 2.0 * np.pi * i0(1.0)
for n_nodes in (2, 4, 8, 16):
    # ϑₙ = 2π n / N, wₙ = 2π / N
    longitude = 2.0 * jnp.pi * jnp.arange(n_nodes) / n_nodes  # → (N,)
    weights = jnp.full(n_nodes, 2.0 * jnp.pi / n_nodes)  # → (N,)
    # today (integrix)
    # Î = ∑ₙ wₙ exp(cos ϑₙ)
    estimate = rule.integrate(weights, jnp.exp(jnp.cos(longitude))).value
    print(f"N = {n_nodes:2d}: error = {abs(float(estimate) - exact_circle):.1e}")
```

```text
N =  2: error = 1.7e+00
N =  4: error = 3.4e-02
N =  8: error = 1.3e-06
N = 16: error = 0.0e+00
```

Each doubling of $N$ roughly squares the error until it hits the round-off of float64.

## A smooth band and an ice edge

Now the two families side by side, on two zonal-mean fields in $\mu = \sin \phi \in [-1, 1]$, where the area of the sphere is uniform in $\mu$. The first is a smooth band of tropical moisture, $f_\text{band}(\mu) = e^{-\mu^2 / (2 \cdot 0.3^2)}$, with an exact integral from the error function. The second is a sea-ice fraction that is zero equatorward of $|\mu| = 0.8$ (about 53°) and grows linearly toward each pole, $f_\text{ice}(\mu) = \max(0, |\mu| - 0.8)$, whose integral is exactly $0.04$. It is continuous but has a kink at each ice edge. The global rule is Gauss–Legendre, from NumPy's nodes and weights; the local rule is the composite trapezoid rule on equally spaced $\mu$. In the table, GL is Gauss–Legendre and TR the trapezoid rule.

```python
from scipy.special import erf

# Shapes: N nodes in μ = sin φ ∈ [−1, 1]


def band(mu):
    """f_band(μ) = exp(−μ²/(2 · 0.3²))."""
    # user
    return jnp.exp(-(mu**2) / (2.0 * 0.3**2))


def ice(mu):
    """f_ice(μ) = max(0, |μ| − 0.8)."""
    # user
    return jnp.maximum(0.0, jnp.abs(mu) - 0.8)


def trapezoid(n_nodes):
    """Equally spaced nodes on [−1, 1], with h/2 at each end."""
    # user
    nodes = jnp.linspace(-1.0, 1.0, n_nodes)  # → (N,)
    spacing = nodes[1] - nodes[0]
    weights = jnp.full(n_nodes, spacing).at[jnp.array([0, -1])].set(spacing / 2)
    return nodes, weights


exact = {
    "band": 0.3 * np.sqrt(2.0 * np.pi) * erf(1.0 / (0.3 * np.sqrt(2.0))),
    "ice": 0.04,
}
labels = ("GL band", "TR band", "GL ice", "TR ice")
print(" N  " + "  ".join(f"{label:>8}" for label in labels))
for n_nodes in (4, 8, 16, 32, 64):
    gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(n_nodes)
    trap_nodes, trap_weights = trapezoid(n_nodes)
    errors = []
    for name, field in (("band", band), ("ice", ice)):
        for nodes, weights in (
            (gauss_nodes, gauss_weights),
            (trap_nodes, trap_weights),
        ):
            # today (integrix)
            # Î = ∑ₙ wₙ f(μₙ)
            estimate = rule.integrate(jnp.asarray(weights), field(nodes)).value
            errors.append(abs(float(estimate) - exact[name]))
    print(f"{n_nodes:2d}  " + "  ".join(f"{error:.2e}" for error in errors))
```

```text
 N   GL band   TR band    GL ice    TR ice
 4  5.38e-02  2.96e-02  2.53e-03  9.33e-02
 8  9.34e-05  5.19e-04  7.55e-03  1.71e-02
16  2.14e-12  1.24e-04  7.77e-04  4.44e-03
32  8.88e-16  2.96e-05  2.87e-04  3.75e-04
64  1.11e-15  7.20e-06  7.04e-05  2.12e-04
```

On the smooth band the Gauss rule reaches round-off with 32 nodes, while the trapezoid rule falls by about four with each doubling, as [](#eq-trapezoid-error) says it should. On the ice edge the Gauss rule loses its advantage: both rules fall at an algebraic rate, about $N^{-2}$, and neither falls smoothly, since the error depends on where the nodes sit relative to the kink. The [convergence notebook](../notebooks/theory_convergence.ipynb) plots the same sweep on a finer range of $N$.

The local rule has one more card to play. Its error comes from the tile that holds the kink, so a tile boundary placed on the kink removes it. With eleven equally spaced nodes there is a node at $\mu = \pm 0.8$, every tile sees a straight line, and the trapezoid rule is exact.

```python
trap_nodes, trap_weights = trapezoid(11)
# today (integrix)
estimate = rule.integrate(trap_weights, ice(trap_nodes)).value
print(f"N = 11, nodes on the edges: error = {abs(float(estimate) - 0.04):.1e}")
```

```text
N = 11, nodes on the edges: error = 2.1e-17
```

A global rule cannot do this without becoming local, by splitting the domain at the kink and using one rule on each side.

## Which one to use

| | Global rule | Local (composite) rule |
| --- | --- | --- |
| nodes | fixed by the rule (Gauss latitudes, Chebyshev points) | wherever the data are, on tiles |
| smooth integrand | spectral, $O(\rho^{-2N})$ | algebraic, $O(h^q)$ |
| kink, front, coastline | algebraic, error spread over the domain | algebraic, error kept in a few tiles, and a tile edge on the kink removes it |
| one bad sample | moves the estimate by its weight, anywhere | moves only its own tile |
| budgets | one number for the domain | one number per tile, which add up to the domain's ([](#eq-additivity)) |
| refinement | all nodes move, the old samples are not reused | split the tiles that need it (chapter 8) |

Neither column wins. Spectral dynamical cores, the trapezoid rule in longitude and the Gauss rules of gaussx's Gaussian expectations live in the left column, because their integrands are smooth on the whole domain and the nodes are ours to choose. Finite-volume models, satellite swaths, retrievals on fixed levels and anything with a coastline live in the right column, because the data come on their own grid and the budgets must close cell by cell. The same split comes back in time (a Runge–Kutta step is a local rule in time, chapter 7), in adaptivity (local error estimates against a global tolerance, chapter 8) and in high dimensions, where Monte Carlo trades both for a rate that does not depend on the dimension (chapter 10).
