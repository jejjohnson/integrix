---
numbering:
  enumerator: "4.%s"
---

# Bases and stencils

## A field is more than its samples

Most fields in a geoscience workflow are not stored as plain samples. A spectral model stores spherical-harmonic coefficients, a finite-element ocean stores the coefficients of hat functions on a mesh, a finite-volume model stores cell averages, and a Gaussian-process emulator stores the weights of its kernel functions. When we integrate such a field, the honest thing is to integrate the representation itself, not to sample it on some grid and start again. It turns out that this gives us the weights of chapter 2 from a different direction, and it explains where the stencils of an interpolation scheme fit in.

## Integrating a basis expansion

Let the integrand be an expansion in $D_z$ basis functions $\phi_d: \Omega \to \mathbb{R}$,

$$f(\vect{s}) = \sum_{d=1}^{D_z} z_d \phi_d(\vect{s})$$ (eq-basis-expansion)

with coefficients $\vect{z} \in \mathbb{R}^{D_z}$. The integral is linear in the coefficients,

$$I[f] = \sum_{d=1}^{D_z} z_d \int_\Omega \phi_d(\vect{s}) \mathrm{d}\mu(\vect{s}) = \vect{v}^\top \vect{z}$$ (eq-basis-integral)

where $\vect{v} \in \mathbb{R}^{D_z}$ holds the integrals of the basis functions, $v_d = \int_\Omega \phi_d \mathrm{d}\mu$. So a basis comes with its own weights, and they are computed once, from the basis alone. As seen above, this is [](#eq-rule) again with the coefficients in place of the samples.

The two families of chapter 3 show up here as two kinds of basis.

- **Global (spectral) bases** have functions that are non-zero on the whole domain: Legendre polynomials, Fourier modes, spherical harmonics, the eigenfunctions of a kernel. Their integral weights are often trivial, because the functions are orthogonal to the constant. On $[-1, 1]$, $\int P_d(\mu) \mathrm{d}\mu = 2$ for $d = 0$ and $0$ otherwise, so the zonal mean of a Legendre expansion is its first coefficient, and the global mean of a spherical-harmonic expansion is its $(0, 0)$ coefficient. However, every coefficient touches every point, so a change in one coefficient changes the field everywhere.
- **Compact-support (local) bases** have functions that are non-zero on a few tiles only: the hat functions of linear finite elements, B-splines, the indicator functions of finite-volume cells. Their integral weights are local too. The integral of the indicator of a cell is its area (the finite-volume rule), and the integral of a hat function is the trapezoid weight of its node.

The second statement is worth checking, since it says that the trapezoid rule is nothing but the exact integral of the piecewise-linear interpolant. We integrate the five hat functions on $[0, 4]$ with a fine Gauss rule on each tile, as one batched call.

```python
import jax

jax.config.update("jax_enable_x64", True)

import einx
import jax.numpy as jnp
import numpy as np
from integrix import rule

# Shapes: D_z = 5 hat functions; N = 4 tiles × 8 Gauss nodes = 32 nodes

knots = jnp.arange(5.0)  # → (D_z,)
# 8 Gauss–Legendre nodes on each tile [k, k + 1]
reference_nodes, reference_weights = np.polynomial.legendre.leggauss(8)
tile_nodes = 0.5 * (reference_nodes + 1.0)  # → (8,)
nodes = einx.add("tile, gauss -> (tile gauss)", knots[:-1], tile_nodes)
weights = jnp.tile(0.5 * reference_weights, 4)  # → (N,)
# φ_d(s) = max(0, 1 − |s − s_d|), the hat on knot s_d
distance = einx.subtract("node, basis -> node basis", nodes, knots)
hats = jnp.maximum(0.0, 1.0 - jnp.abs(distance))  # → (N, D_z)
# today (integrix)
# v_d = ∫ φ_d(s) ds ≈ ∑ₙ wₙ φ_d(sₙ)
basis_weights = rule.integrate(weights, hats, axis=0).value  # (N, D_z) → (D_z,)
print("v =", jnp.round(basis_weights, 12).tolist())
```

```text
v = [0.5, 1.0, 1.0, 1.0, 0.5]
```

These are the trapezoid weights for a unit spacing, so a piecewise-linear field integrated exactly and its samples integrated by the trapezoid rule give the same number.

## From samples to coefficients

If the field is given by its samples $\vect{f} \in \mathbb{R}^N$ at nodes $\vect{s}_n$, with as many nodes as basis functions ($N = D_z$), interpolation finds the coefficients by solving $\gmat{\Phi} \vect{z} = \vect{f}$, where $\gmat{\Phi} \in \mathbb{R}^{N \times D_z}$ is the basis matrix with $[\gmat{\Phi}]_{nd} = \phi_d(\vect{s}_n)$. Then

$$I[f] \approx \vect{v}^\top \gmat{\Phi}^{-1} \vect{f} = \vect{w}^\top \vect{f}, \quad \gmat{\Phi}^\top \vect{w} = \vect{v}$$ (eq-basis-weights)

so the rule's weights solve a linear system with the transposed basis matrix. With monomials as the basis, $\gmat{\Phi}^\top$ is the matrix $\mat{V}$ of [](#eq-moments) and $\vect{v}$ the moments $\vect{b}$, which is chapter 2's construction. With hat functions, $\gmat{\Phi}$ is the identity on the knots and $\vect{w} = \vect{v}$, which is the check above. Any basis gives a rule this way, and the rule is global or local exactly when the basis is.

## Stencils: integrating a read-out

The last piece is the read-out. Interpolation schemes (nearest, linear, cubic, conservative remapping) all map values on a source grid to values at target points with a linear map $\mat{H} \in \mathbb{R}^{N_\text{target} \times N_\text{source}}$, so that $\hat{\vect{f}} = \mat{H} \vect{c}$ with $\vect{c} \in \mathbb{R}^{N_\text{source}}$ and $\hat{\vect{f}} \in \mathbb{R}^{N_\text{target}}$. For a local scheme each row of $\mat{H}$ has a few non-zero taps, the stencil, which functax stores as a sparse `Stencil` of indices and weights. For a global scheme (a kernel read-out $k(\vect{s}_*, \vect{s})$) the rows are dense.

If the targets carry a rule with weights $\vect{w}_\text{target} \in \mathbb{R}^{N_\text{target}}$, then

$$\vect{w}_\text{target}^\top \mat{H} \vect{c} = (\mat{H}^\top \vect{w}_\text{target})^\top \vect{c}$$ (eq-stencil-weights)

so integrating the read-out with the target rule is the same as integrating the source values with the weights $\mat{H}^\top \vect{w}_\text{target}$. The adjoint of the read-out moves a rule from one grid to another. A sparse $\mat{H}$ moves a local rule to a local rule, and a dense one spreads every target weight over every source.

Here is the smallest example: linear interpolation to the midpoints of four samples, then the midpoint rule on the three midpoints. The weights it puts on the samples are the trapezoid weights again.

```python
from functax.interp import Stencil

# Shapes: N_source = 4 samples; N_target = 3 midpoints, 2 taps each

# today (functax)
# row k reads (cₖ + cₖ₊₁)/2
midpoints = Stencil(
    jnp.array([[0, 1], [1, 2], [2, 3]]),
    jnp.full((3, 2), 0.5),
    n_source=4,
)
# the midpoint rule on unit tiles: w_target = (1, 1, 1)
target_weights = jnp.ones(3)  # → (N_target,)
# today (functax)
# w_source = Hᵀ w_target
source_weights = midpoints.apply_adjoint(target_weights)  # → (N_source,)
print("H^T w =", source_weights.tolist())
```

```text
H^T w = [0.5, 1.0, 1.0, 0.5]
```

This is the same identity that chapter 9 uses for gradients: the gradient of an integral with respect to the source values is $\mat{H}^\top \vect{w}_\text{target}$, computed by the same adjoint.

## Who owns what

Bases, stencils and rules are data, and the procedures that use them are code, so the stack splits them. functax owns the bases (`functax.basis`), the read-outs (`Stencil`, and `Plan` to build them) and the rules (`functax.quad`, functax#44). integrix owns what is done with them: the sums of [](#eq-rule) and [](#eq-basis-integral), the scans, the adaptive refinement, the marches and their adjoints. gaussx owns the Gaussian expectations that come with a closed rule (Gauss–Hermite, unscented). Today integrix's pin of functax has `Stencil` but not yet `Plan` or the rules, so the pages in this section build their Gauss nodes with NumPy; integrix accepts any object with `nodes` and `weights`, or a plain weights array, so the code does not change when the rules land.
