---
numbering:
  enumerator: "2.%s"
---

# Weighted summation

## The column is already a weighted sum

A methane retrieval does not hand out a continuous profile $c(p)$. It reports the mole fraction on a few model levels $p_1, \ldots, p_{N}$ and, next to it, a pressure weighting function $h_1, \ldots, h_{N}$, so that the column average is

$$\mathrm{XCH_4} \approx \sum_{n=1}^{N} h_n c(p_n)$$ (eq-xch4-sum)

where $h_n \ge 0$ and $\sum_n h_n = 1$. So the integral of chapter 1, $\frac{1}{p_\text{s}} \int_0^{p_\text{s}} c(p) \mathrm{d}p$, has become a dot product, and the weighting function is the part of the product that holds the numerics. Anyone who compares two retrievals, or a retrieval against a model, has to apply the same $h_n$ to both, or the comparison picks up a difference in the weights instead of a difference in methane.

This is the whole idea of the section. Every integral integrix computes ends up as samples of the integrand times weights, summed, and the different modules are different ways of choosing those weights.

## Rules: nodes and weights

A rule with $N$ nodes is a set of points $\vect{s}_n \in \Omega$ and weights $w_n \in \mathbb{R}$, $n = 1, \ldots, N$. It approximates [](#eq-integral) by

$$\hat{I}_N[f] = \sum_{n=1}^{N} w_n f(\vect{s}_n) = \vect{w}^\top \vect{f}$$ (eq-rule)

where $\vect{w} \in \mathbb{R}^N$ holds the weights and $\vect{f} \in \mathbb{R}^N$ the samples $f_n = f(\vect{s}_n)$. The error of the rule is $E_N[f] = I[f] - \hat{I}_N[f]$. The measure of chapter 1 goes into the weights: a rule for $\int_0^{p_\text{s}} c(p) \mathrm{d}p / p_\text{s}$ is a rule for $\mathrm{d}p$ divided by $p_\text{s}$, which is how the $h_n$ of [](#eq-xch4-sum) come about.

There are only a few ways to choose the weights, and the six modules of integrix use all of them.

| Module | How the weights are chosen | Geoscience example |
| --- | --- | --- |
| `rule` | in advance, from the nodes: interpolation, Gauss, or cell areas | the $\cos \phi$ of a latitude band, a pressure weighting function |
| `scan` | the same weights, summed up to every node | optical depth from the top of the atmosphere to each level |
| `march` | the weights of a Runge–Kutta or Adams method, on samples the stepper computes itself | an ocean model stepping its tracers |
| `adapt` | after looking at $f$: more nodes where the local error is large | a narrow absorption line |
| `expect` | random nodes, with weights $1/N$ or importance ratios | an ensemble of emission rates |
| `adjoint` | the same weights, read backwards, since $\partial \hat{I}_N / \partial f_n = w_n$ | the sensitivity of a column to each level |

The rest of this chapter is about the first row, because the others are built from it.

## Interpolatory weights

The oldest way to get weights is to interpolate the samples and integrate the interpolant. Take $N$ distinct nodes on an interval and the polynomial $\pi_{N-1}$ of degree $N - 1$ through the samples. Integrating it gives a weighted sum of the samples, with weights that depend only on the nodes, and those weights are the ones that integrate every monomial $s^m$ of degree $m \le N - 1$ exactly:

$$\sum_{n=1}^{N} w_n s_n^m = \int_\Omega s^m \mathrm{d}\mu(s), \quad m = 0, \ldots, N - 1$$ (eq-moments)

These are the moment equations. In matrix form they are $\mat{V} \vect{w} = \vect{b}$, where $\mat{V} \in \mathbb{R}^{N \times N}$ is the transposed Vandermonde matrix with $[\mat{V}]_{mn} = s_n^m$, and $\vect{b} \in \mathbb{R}^N$ holds the moments $b_m = \int_\Omega s^m \mathrm{d}\mu(s)$. On $[0, 1]$ with the ordinary length, $b_m = 1/(m + 1)$.

We solve them below for equally spaced nodes on $[0, 1]$ (the Newton–Cotes rules). With three nodes we get Simpson's rule, and with eleven we get something we should not use.

```python
import jax

jax.config.update("jax_enable_x64", True)

import einx
import jax.numpy as jnp
import lineax as lx

# Shapes: N nodes on [0, 1]


def newton_cotes_weights(n_nodes):
    """Solve the moment equations V w = b for N equally spaced nodes."""
    # user
    nodes = jnp.linspace(0.0, 1.0, n_nodes)  # → (N,)
    # [V]ₘₙ = sₙᵐ
    vandermonde = jnp.vander(nodes, increasing=True)  # (N,) → (N, N)
    moment_matrix = einx.id("n m -> m n", vandermonde)  # (N, N) → (N, N)
    # bₘ = ∫₀¹ sᵐ ds = 1/(m + 1)
    moments = 1.0 / jnp.arange(1.0, n_nodes + 1.0)  # → (N,)
    # w = V⁻¹ b, by a solve
    operator = lx.MatrixLinearOperator(moment_matrix)
    return nodes, lx.linear_solve(operator, moments).value  # → (N,)


simpson_nodes, simpson = newton_cotes_weights(3)
print("N = 3: ", jnp.round(simpson * 6.0, 10).tolist(), "/ 6")
eleven_nodes, eleven = newton_cotes_weights(11)
print(f"N = 11: min w = {float(eleven.min()):.3f}")
print(f"N = 11: sum |w| = {float(jnp.abs(eleven).sum()):.3f}")
```

```text
N = 3:  [1.0, 4.0, 1.0] / 6
N = 11: min w = -0.435
N = 11: sum |w| = 3.065
```

## Exactness

A rule has degree of exactness $m^\star$ when it integrates every polynomial of degree $m \le m^\star$ exactly and fails for some polynomial of degree $m^\star + 1$. The moment equations give an interpolatory rule with $N$ nodes a degree of at least $N - 1$, and sometimes one more for free: Simpson's three nodes are symmetric, so the odd moment $s^3$ about the midpoint also cancels and $m^\star = 3$.

```python
from integrix import rule

for degree in (2, 3, 4):
    # today (integrix)
    # Î = ∑ₙ wₙ sₙᵐ against ∫₀¹ sᵐ ds = 1/(m + 1)
    estimate = rule.integrate(simpson, simpson_nodes**degree).value
    error = float(estimate) - 1.0 / (degree + 1)
    # round-off below 10⁻¹² counts as exact
    verdict = "exact" if abs(error) < 1e-12 else f"error = {error:.2e}"
    print(f"m = {degree}: {verdict}")
```

```text
m = 2: exact
m = 3: exact
m = 4: error = 8.33e-03
```

Exactness is the language in which rules are compared. With $N$ nodes fixed in advance, $N - 1$ is what the moment equations can promise. If we are also free to place the nodes, we have $2N$ unknowns and can match $2N$ moments, and that is the Gauss rule, with $m^\star = 2N - 1$ (chapter 3). A high degree is only useful when the integrand looks like a polynomial on the whole domain, though, and this is where the global and local thread starts.

## Positive weights and noise

The eleven-node weights above sum to one, as they must, since they integrate the constant exactly. However, several of them are negative and the sum of their absolute values is about 3. That number matters for real data, because samples are never exact. If each sample carries an error $|\varepsilon_n| \le \varepsilon_\text{max}$ (instrument noise, a retrieval error, round-off), the rule passes it on as

$$\Big\vert \sum_{n=1}^{N} w_n (f_n + \varepsilon_n) - \sum_{n=1}^{N} w_n f_n \Big\vert \le \varepsilon_\text{max} \sum_{n=1}^{N} \vert w_n \vert$$ (eq-noise-bound)

so $\sum_n |w_n|$ is the amplification factor of the rule. With positive weights it equals $\sum_n w_n = \mu(\Omega)$, the size of the domain, which is the smallest it can be. With negative weights it grows, and for equally spaced nodes it grows without bound as $N$ increases (the Runge phenomenon, seen through the weights). Negative weights also break the positivity of chapter 1: a positive methane profile can come out with a negative column.

So we have three practical rules for weights, which the later chapters keep returning to.

1. Prefer positive weights. Gauss, Clenshaw–Curtis, the composite trapezoid and Simpson rules, cell areas and Monte Carlo all have them.
2. Never raise the degree by adding equally spaced nodes to one polynomial. Either move the nodes (Gauss, chapter 3) or cut the domain into pieces and use a low-degree rule on each (composite rules, chapter 3).
3. Keep the weights next to the data they came from. A pressure weighting function, a cell-area grid or a spectral response is part of the dataset, and integrix takes it as a plain array for exactly that reason.
