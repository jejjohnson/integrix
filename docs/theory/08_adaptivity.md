---
numbering:
  enumerator: "8.%s"
---

# Adaptivity

## A narrow methane line

Methane absorbs in thousands of narrow lines, and in the lower atmosphere each line has a Lorentzian shape whose half-width is a small fraction of a nanometre. Integrating one line across a 40 nm band with a fixed grid is a bad deal: the grid spends almost all of its nodes on the flat wings, where nothing happens, and still under-samples the core, where all the absorption is. We would like the nodes to go where the integrand is hard, and we do not know in advance where that is (in a real spectrum the line positions shift with pressure and temperature, and there are many of them).

This is adaptive quadrature. A rule is applied, its error is estimated from the samples it already has, and the nodes are refined where that estimate is large, until the whole integral meets a tolerance. Every step of this is still a weighted sum, so adaptivity is a statement about weights: it compares two weightings of the same samples, and uses the difference to decide where to sample next. In integrix it is the [`GaussKronrod`](xref:api#integrix.adapt.GaussKronrod), [`TanhSinh`](xref:api#integrix.adapt.TanhSinh) and [`Romberg`](xref:api#integrix.adapt.Romberg) rules.

## Two rules, one error estimate

The trick behind every adaptive rule is to compute two estimates with different accuracy and take their difference as the error of the worse one. The Gauss–Kronrod pair does this for free. A Gauss rule $G_n$ with $n$ nodes is extended by $n + 1$ more nodes to a Kronrod rule $K_{2n+1}$ that reuses all of the Gauss nodes, so both estimates come from the same $2n + 1$ samples, and

$$E \approx \vert K_{2n+1}[f] - G_n[f] \vert$$ (eq-gauss-kronrod)

is an estimate of the error of $G_n$ that is, in practice, a very conservative bound on the error of $K_{2n+1}$, which is the value we keep (Piessens et al., 1983). integrix's default pair is $G_7$ and $K_{15}$.

The same idea appears in time as an embedded Runge–Kutta pair. `RK2Heun` carries forward Euler inside it, since both use the same stages, and the difference between the two updates estimates the local error of the step; diffrax's step-size controller then shrinks or grows the step to keep it under a tolerance. A Gauss–Kronrod pair refines in space and an embedded pair refines in time, but they are one algorithm.

## Local estimates, a global tolerance

This is the global and local thread in its most practical form. The error estimates are local, one per subinterval. The tolerance is global: we ask for the total error of the integral to be below $\max(\text{atol}, \text{rtol} \vert I \vert)$. Global adaptive bisection connects the two. It keeps a list of subintervals with their values and error estimates, and as long as the summed error is too large, it bisects the subinterval with the largest error, wherever that is, and evaluates the pair on both halves. The refinement is local (only the worst piece is split) and the decision is global (the worst piece over the whole domain, and a stopping rule on the whole integral).

We integrate the Lorentzian line $\ell(\lambda) = \frac{\gamma_\text{L} / \pi}{(\lambda - \lambda_0)^2 + \gamma_\text{L}^2}$, with centre $\lambda_0 = 2300$ nm and half-width $\gamma_\text{L} = 0.05$ nm, over the band $[2280, 2320]$ nm, where its exact integral is $\frac{1}{\pi} \left( \arctan \frac{20}{\gamma_\text{L}} + \arctan \frac{20}{\gamma_\text{L}} \right)$, just under 1. We compare the adaptive rule with the trapezoid rule on fixed grids.

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from integrix import adapt, rule

# Shapes: N wavelengths on a fixed grid

centre, half_width = 2300.0, 0.05
lower, upper = 2280.0, 2320.0


def line(wavelength):
    """ℓ(λ) = (γ_L/π) / ((λ − λ₀)² + γ_L²), in nm⁻¹."""
    # user
    return (half_width / jnp.pi) / ((wavelength - centre) ** 2 + half_width**2)


exact = (
    np.arctan((upper - centre) / half_width) - np.arctan((lower - centre) / half_width)
) / np.pi

# today (integrix)
# I ≈ ∑ over subintervals of K₁₅, refined until ∑ E ≤ rtol |I|
adaptive = adapt.GaussKronrod(rtol=1e-8)(line, lower, upper)
print(
    f"Gauss-Kronrod: {int(adaptive.n_evaluations):5d} evaluations, "
    f"error {abs(float(adaptive.value) - exact):.1e}, "
    f"estimate {float(adaptive.error):.1e}, converged {bool(adaptive.converged)}"
)
for n_nodes in (401, 4001):
    wavelength = jnp.linspace(lower, upper, n_nodes)  # → (N,)
    spacing = wavelength[1] - wavelength[0]
    weights = jnp.full(n_nodes, spacing).at[jnp.array([0, -1])].set(spacing / 2)
    # today (integrix)
    estimate = rule.integrate(weights, line(wavelength)).value
    print(
        f"trapezoid:     {n_nodes:5d} evaluations, "
        f"error {abs(float(estimate) - exact):.1e}"
    )
```

```text
Gauss-Kronrod:   585 evaluations, error 2.7e-14, estimate 4.4e-11, converged True
trapezoid:       401 evaluations, error 9.0e-02
trapezoid:      4001 evaluations, error 9.0e-11
```

The adaptive rule gets to round-off with about a seventh of the evaluations that the fixed grid needs, and the 401-point grid, which has nodes every 0.1 nm, is still wrong in the second digit because only one node falls inside the line's core. The error estimate is far larger than the true error, as it usually is for Gauss–Kronrod: it is the error of $G_7$, and we keep $K_{15}$.

## Richardson and the double-exponential map

Two more adaptive rules sit next to Gauss–Kronrod, each for a different kind of integrand.

- **Romberg.** The trapezoid rule's error has the expansion [](#eq-euler-maclaurin), $T(h) = I + c_1 h^2 + c_2 h^4 + \ldots$, so two estimates at $h$ and $h/2$ can be combined to cancel the $h^2$ term, then the $h^4$ term, and so on (Richardson extrapolation). It uses the samples it already has, so it is cheap on a smooth integrand such as a Gaussian spectral response. On a narrow line it is slow, since the expansion only starts to hold once $h$ resolves the line.
- **Tanh–sinh.** The substitution $\lambda = c + d \tanh(\frac{\pi}{2} \sinh t)$ maps an interval to the whole real line in $t$ and makes the transformed integrand decay double-exponentially at both ends (Takahasi and Mori, 1974). The trapezoid rule in $t$ then converges very fast, even when the original integrand has an endpoint singularity ($\log \lambda$, $\lambda^{-1/2}$) or the interval is infinite.

Tanh–sinh gives the strength of the whole line, over all wavelengths. It wants the feature to sit on the unit scale, since its nodes cluster around the origin, so we integrate in units of the half-width, $u = (\lambda - \lambda_0) / \gamma_\text{L}$, where the line is $\frac{1}{\pi (1 + u^2)}$ and its integral over the real line is exactly 1.

```python
# today (integrix)
# ∫ from −∞ to ∞ of 1/(π (1 + u²)) du = 1
strength = adapt.TanhSinh()(lambda u: 1.0 / (jnp.pi * (1.0 + u**2)), -jnp.inf, jnp.inf)
print(
    f"tanh-sinh: error {abs(float(strength.value) - 1.0):.1e}, "
    f"converged {bool(strength.converged)}"
)

# a smooth Gaussian response: R(λ) = exp(−(λ − λ₀)²/(2 · 5²))
response_integral = (
    5.0
    * np.sqrt(2.0 * np.pi)
    * float(jax.scipy.special.erf(20.0 / (5.0 * np.sqrt(2.0))))
)
# today (integrix)
romberg = adapt.Romberg()(
    lambda lam: jnp.exp(-0.5 * ((lam - centre) / 5.0) ** 2), lower, upper
)
print(
    f"Romberg on the response: error {abs(float(romberg.value) - response_integral):.1e}, "
    f"converged {bool(romberg.converged)}"
)
```

```text
tanh-sinh: error 2.2e-16, converged True
Romberg on the response: error 8.3e-14, converged True
```

## Adaptivity under `jit`

An adaptive rule decides at run time how many subintervals it needs, and `jit` wants every shape fixed at trace time. integrix squares this in three ways.

1. **A static buffer.** `GaussKronrod` keeps a buffer of `max_intervals` subintervals (64 by default) and refines inside it with a `while_loop`, so the shapes never change.
2. **`converged` is a flag, not an exception.** If the buffer fills before the tolerance is met, the result comes back with `converged=False`; a traced function cannot raise on a value, so the caller checks the flag.
3. **The gradient of the final rule.** Differentiating through the refinement loop would differentiate the decisions too. integrix instead differentiates the final Kronrod rule on the final partition, held fixed, which is the derivative of a weighted sum and is exact to the accuracy of the integral (chapter 9).

The second point is easy to trip over, so we show it: with room for only four subintervals the line integral stops early and says so, and the derivative with respect to the line width is checked against a finite difference of the exact integral.

```python
# today (integrix)
cramped = adapt.GaussKronrod(rtol=1e-8, max_intervals=4)(line, lower, upper)
print(
    f"4 subintervals: error {abs(float(cramped.value) - exact):.1e}, "
    f"converged {bool(cramped.converged)}"
)


def band_integral(width):
    """∫ ℓ(λ; γ_L) dλ over the band, adaptively."""
    # user
    profile = lambda lam: (width / jnp.pi) / ((lam - centre) ** 2 + width**2)
    # today (integrix)
    return adapt.GaussKronrod(rtol=1e-10)(profile, lower, upper).value


def exact_band(width):
    """(1/π)(arctan((b − λ₀)/γ_L) − arctan((a − λ₀)/γ_L))."""
    # user
    return (
        np.arctan((upper - centre) / width) - np.arctan((lower - centre) / width)
    ) / np.pi


derivative = float(jax.jit(jax.grad(band_integral))(half_width))
finite_difference = (
    exact_band(half_width + 1e-7) - exact_band(half_width - 1e-7)
) / 2e-7
print(f"d/dγ_L: {derivative:.6f} (finite difference {finite_difference:.6f})")
```

```text
4 subintervals: error 6.1e-02, converged False
d/dγ_L: -0.031831 (finite difference -0.031831)
```

The band loses a little of the line as the line widens, since more of its wings fall outside the 40 nm window, and the gradient through the jitted adaptive rule says exactly how much.
