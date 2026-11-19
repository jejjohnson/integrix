# Sums against a rule

Role: ① primitive (`role-ladder`, `references/primitive.md`).
Unit: integrals.

Open this when an integral is a weighted sum of samples you already have, or of an integrand at a rule's nodes.

## Names

All of them live in `integrix.rule`.

| Name | What it is |
|---|---|
| `integrate(rule, values_or_fn, *, axis=None, args=None, where=None, nan_policy="propagate")` | I ≈ ∑ₙ wₙ f(xₙ), contracted over one axis or several and batched over the rest. Returns an `Integral`. |
| `weighted_mean(rule, values, *, weight, axis=None, where=None, nan_policy="propagate")` | ∑ₙ wₙ Rₙ Lₙ / ∑ₙ wₙ Rₙ with one rule for both sums, such as a band radiance from a spectrum L and a response R. Returns an array. |
| `log_integrate(rule, log_values_or_fn, *, axis=None, args=None, where=None)` | log I from log f by log-sum-exp, for integrands whose values underflow. Returns an array. |
| `Integral` | The result: `value`, `error` (`None` for a plain rule), `n_evaluations`, `converged` (`True` for a plain rule). |

The `rule` argument takes three forms.

| Form | Weights |
|---|---|
| A rule with `weights`, such as a `functax.quad.Rule` | the rule's own, over as many axes as the weights have |
| A tuple of 1-D rules | their outer product, one axis each |
| A plain array | as given: a spectral response, cell areas |

## Use it

```python
import jax.numpy as jnp
from functax.quad import gauss_legendre, trapezoid

from integrix.rule import Integral, integrate, log_integrate, weighted_mean

# an integrand at a rule's nodes
rule = gauss_legendre(8, (0.0, 1.0))
result = integrate(rule, jnp.exp)
assert isinstance(result, Integral)
assert jnp.allclose(result.value, jnp.e - 1.0, rtol=1e-5)
assert result.error is None and bool(result.converged)

# samples you already have: 5 spectra on 101 wavelengths, integrated over the last axis
wavelength = jnp.linspace(1.0, 2.0, 101)
spectra = jnp.ones((5, 101)) * jnp.arange(1.0, 6.0)[:, None]
band = integrate(trapezoid(wavelength), spectra).value  # (5,)
assert jnp.allclose(band, jnp.arange(1.0, 6.0), rtol=1e-5)

# two axes at once with a tuple of rules; the node axes of a callable come first
x = gauss_legendre(6, (0.0, 1.0))
y = gauss_legendre(6, (0.0, 2.0))
field = x.nodes[:, None] * y.nodes[None, :]  # (6, 6)
assert jnp.allclose(integrate((x, y), field).value, 1.0, rtol=1e-5)

# a mask leaves samples out, and gives them a zero gradient
keep = wavelength <= 1.5
half = integrate(trapezoid(wavelength), spectra, where=keep).value
assert bool((half < band).all())

# a band radiance: the mean of a spectrum under a response
response = jnp.exp(-0.5 * ((wavelength - 1.5) / 0.1) ** 2)
radiance = weighted_mean(trapezoid(wavelength), spectra, weight=response)  # (5,)
assert jnp.allclose(radiance, jnp.arange(1.0, 6.0), rtol=1e-5)

# log I when f underflows: log ∫₀¹ e^(−800) dx = −800
log_total = log_integrate(rule, lambda nodes: jnp.full(nodes.shape, -800.0))
assert jnp.allclose(log_total, -800.0)
```

## Rules

- The rule is data: its nodes and weights come from functax, or the weights are an array you hold. integrix never builds a rule functax already has.
- `axis` names one axis of the samples per weights axis; `None` takes the last ones. A callable returns its node axes first, so it takes no `axis`.
- `where` is a boolean mask that broadcasts against the samples. With `nan_policy="omit"` NaN samples are left out too, like `np.nansum`.
- `weighted_mean` renormalises under a mask: both sums run over the kept samples. A weight with extra leading axes is a batch of weight functions, and their axes come last in the result.
- `log_integrate` needs positive weights and raises `ValidationError` otherwise, under `jit` too. With no node kept it returns −∞.
- Read `result.value`; an `Integral` is not an array.
