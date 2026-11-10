# Notation

Every page of the mathematical reference uses the symbols below. The letter says what kind of object a symbol is, the subscript says which variable it belongs to, and a bold symbol is a vector or a matrix. Equations use $\vect{s}$ on the pages and Unicode, such as wₙ or Σ_x, in docstrings and code comments.

One departure from the stack's notation standard is deliberate. There, $f$ is the dynamics of a state-space model; here it is the integrand. The two meet in chapter 7, where the integrand of a march is the right-hand side of an ODE, so $f$ keeps one meaning across the section.

## Domains, nodes and sizes

:::{table} The domain, its nodes and the sizes used on every page.
:label: tbl-notation-domains

| Symbol | Shape | Meaning | In code |
| --- | --- | --- | --- |
| $\Omega$, $\vect{s}$ | $\vect{s} \in \Omega \subset \mathbb{R}^{D_s}$ | the domain of integration and a point in it | |
| $\mu$, $\rho_\mu$ | | a measure on $\Omega$ and its density, when it has one | `weight=` in `weighted_mean` |
| $N$, $n$ | | the number of nodes of a rule, and one node | `node` dim |
| $\vect{s}_n$ | $\Omega$ | the nodes | `rule.nodes` |
| $\Omega^{(r)}$, $N_\text{tile}$ | | a tile of a partition of $\Omega$, and the number of tiles | |
| $h$ | $\mathbb{R}_{>0}$ | the width of a tile, or a step | |
| $D_z$, $d$ | | the number of basis functions, and one of them | `basis` dim |
:::

## Integrals and rules

:::{table} The integral, a rule's estimate and error, and the symbols of exactness and convergence.
:label: tbl-notation-rules

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
:::

## Bases and read-outs

:::{table} Bases, coefficients and read-outs from chapter 4.
:label: tbl-notation-bases

| Symbol | Shape | Meaning | In code |
| --- | --- | --- | --- |
| $\phi_d$ | $\Omega \to \mathbb{R}$ | a basis function | `functax.basis` |
| $\vect{z}$ | $\mathbb{R}^{D_z}$ | the coefficients of a basis expansion [](#eq-basis-expansion) | |
| $\gmat{\Phi}$ | $N \times D_z$ | the basis matrix, $[\gmat{\Phi}]_{nd} = \phi_d(\vect{s}_n)$ | |
| $\vect{v}$ | $\mathbb{R}^{D_z}$ | the integrals of the basis functions [](#eq-basis-integral) | |
| $\mat{H}$ | $N_\text{target} \times N_\text{source}$ | a read-out from source values to target values | `functax.interp.Stencil` |
| $\vect{c}$ | $\mathbb{R}^{N_\text{source}}$ | the source values of a read-out | |
:::

## Time stepping

:::{table} Time stepping, from chapter 7.
:label: tbl-notation-marches

| Symbol | Shape | Meaning | In code |
| --- | --- | --- | --- |
| $\vect{x}(t)$, $\vect{x}_k$ | $\mathbb{R}^{D_x}$ | the state, and its value at $t_k = t_0 + k h$ | `y0`, `state` |
| $f(\vect{x}, t)$ | $\mathbb{R}^{D_x}$ | the tendency, which is the integrand of a march | `rhs`, `rhs_fn` |
| $h$, $K$ | | the step, and the number of steps to the end time $T = K h$ | `dt`, `n_steps` |
| $a_{ij}$, $b_i$, $c_i$, $N_\text{stage}$ | | the Butcher coefficients of a Runge–Kutta method, and its number of stages | |
| $\vect{k}_i$ | $\mathbb{R}^{D_x}$ | the stage tendencies | |
| $\gamma$, $\zeta = h \gamma$, $R(\zeta)$ | $\mathbb{C}$ | an eigenvalue of the linearised tendency, its scaled form, and a stepper's stability function | |
:::

The Butcher weights $b_i$ are a rule's weights on $[0, 1]$ and have nothing to do with the moments $b_m$ of chapter 2; the two never meet on one page.

## Adaptivity, adjoints and expectations

:::{table} Error estimates, adjoints and random nodes, from chapters 8 to 10.
:label: tbl-notation-cross

| Symbol | Shape | Meaning | In code |
| --- | --- | --- | --- |
| $G_n$, $K_{2n+1}$, $E$ | | a Gauss rule, its Kronrod extension, and the error estimate of the pair | `GaussKronrod`, `Integral.error` |
| $\vect{x}^\dagger_k$ | $\mathbb{R}^{D_x}$ | the adjoint state at step $k$ | |
| $\mat{J}_{\Phi,k}$ | $D_x \times D_x$ | the Jacobian of one step map $\Phi$ at $\vect{x}_k$ | |
| $\mathcal{J}$ | $\mathbb{R}$ | a cost over the state, such as a 4D-Var misfit | |
| $B$ | | the block length of checkpointing, the spacing of the stored states | `every` |
| $D$ | | the dimension of an expectation's integration space | `dim` |
| $\vect{x}^{(e)}$, $N_\mathcal{E}$ | $\mathbb{R}^{D}$ | a draw, and the number of draws | `n` |
| $\hat{\sigma}$ | | the standard error of an estimate | `Expectation.standard_error` |
| $w^{(e)}$, $N_\text{eff}$ | | importance weights, and the effective sample size | `Expectation.effective_sample_size` |
:::

## Precision and the methane example

:::{table} Symbols of chapters 11 and 13.
:label: tbl-notation-applied

| Symbol | Shape | Meaning | In code |
| --- | --- | --- | --- |
| $\varepsilon$ | | the unit round-off of a dtype ($2^{-24}$ in float32) | `jnp.finfo(dtype).eps / 2` |
| $\kappa$ | | the condition number of a sum, $\sum_n \vert w_n f_n \vert / \vert \sum_n w_n f_n \vert$ | |
| $\Omega(x, y)$, $\Omega_\text{bg}$ | kg m⁻² | the methane column mass and its background, as in the IME literature (chapter 13 only; elsewhere $\Omega$ is the domain) | |
| $\delta$ | | a column enhancement relative to the background | |
| $Q$, $U$, $L_P$ | kg s⁻¹, m s⁻¹, m | the emission rate, the wind speed and the plume length along the wind | |
:::

## Physical symbols

Each page declares the physical symbols it uses in its own table. The ones that recur are wavelength $\lambda$ (nm), pressure $p$ and surface pressure $p_\text{s}$ (hPa), latitude $\phi$ with $\mu = \sin \phi$, longitude $\vartheta$, and height $z$ (m).
