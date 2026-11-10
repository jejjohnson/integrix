# Notation

Every page of the mathematical reference uses the symbols below. The letter says what kind of object a symbol is, the subscript says which variable it belongs to, and a bold symbol is a vector or a matrix. Equations use $\vect{s}$ on the pages and Unicode, such as wₙ or Σ_x, in docstrings and code comments.

One departure from the stack's notation standard is deliberate. There, $f$ is the dynamics of a state-space model; here it is the integrand. The two meet in chapter 7, where the integrand of a march is the right-hand side of an ODE, so $f$ keeps one meaning across the section.

## Domains, nodes and sizes

| Symbol | Shape | Meaning | In code |
| --- | --- | --- | --- |
| $\Omega$, $\vect{s}$ | $\vect{s} \in \Omega \subset \mathbb{R}^{D_s}$ | the domain of integration and a point in it | |
| $\mu$, $\rho_\mu$ | | a measure on $\Omega$ and its density, when it has one | `weight=` in `weighted_mean` |
| $N$, $n$ | | the number of nodes of a rule, and one node | `node` dim |
| $\vect{s}_n$ | $\Omega$ | the nodes | `rule.nodes` |
| $\Omega^{(r)}$, $N_\text{tile}$ | | a tile of a partition of $\Omega$, and the number of tiles | |
| $h$ | $\mathbb{R}_{>0}$ | the width of a tile, or a step | |
| $D_z$, $d$ | | the number of basis functions, and one of them | `basis` dim |

## Integrals and rules

| Symbol | Shape | Meaning | In code |
| --- | --- | --- | --- |
| $f$ | $\Omega \to \mathbb{R}^{D_f}$ | the integrand, batched over $D_f$ entries | `values_or_fn` |
| $I[f]$ | $\mathbb{R}^{D_f}$ | the integral [](#eq-integral) | `Integral.value` |
| $\hat{I}_N[f]$ | $\mathbb{R}^{D_f}$ | the rule's estimate [](#eq-rule) | `Integral.value` |
| $E_N[f]$ | $\mathbb{R}^{D_f}$ | the error $I[f] - \hat{I}_N[f]$ | `Integral.error`, when estimated |
| $\vect{w}$, $w_n$ | $\mathbb{R}^N$ | the weights | `rule.weights`, or a weights array |
| $\vect{f}$, $f_n$ | $\mathbb{R}^N$ | the samples $f(\vect{s}_n)$ | `values` |
| $m$, $m^\star$ | | a polynomial degree, and a rule's degree of exactness | |
| $\mat{V}$, $\vect{b}$ | $N \times N$, $\mathbb{R}^N$ | the moment matrix $[\mat{V}]_{mn} = s_n^m$ and the moments $b_m$ [](#eq-moments) | |
| $q$ | | the order of a local rule, $E_N = O(h^q)$ | |
| $\rho$ | $\mathbb{R}_{>1}$ | the Bernstein-ellipse parameter of a global rule's rate [](#eq-gauss-rate) | |

## Bases and read-outs

| Symbol | Shape | Meaning | In code |
| --- | --- | --- | --- |
| $\phi_d$ | $\Omega \to \mathbb{R}$ | a basis function | `functax.basis` |
| $\vect{z}$ | $\mathbb{R}^{D_z}$ | the coefficients of a basis expansion [](#eq-basis-expansion) | |
| $\gmat{\Phi}$ | $N \times D_z$ | the basis matrix, $[\gmat{\Phi}]_{nd} = \phi_d(\vect{s}_n)$ | |
| $\vect{v}$ | $\mathbb{R}^{D_z}$ | the integrals of the basis functions [](#eq-basis-integral) | |
| $\mat{H}$ | $N_\text{target} \times N_\text{source}$ | a read-out from source values to target values | `functax.interp.Stencil` |
| $\vect{c}$ | $\mathbb{R}^{N_\text{source}}$ | the source values of a read-out | |

## Physical symbols

Each page declares the physical symbols it uses in its own table. The ones that recur are wavelength $\lambda$ (nm), pressure $p$ and surface pressure $p_\text{s}$ (hPa), latitude $\phi$ with $\mu = \sin \phi$, longitude $\vartheta$, and height $z$ (m).
