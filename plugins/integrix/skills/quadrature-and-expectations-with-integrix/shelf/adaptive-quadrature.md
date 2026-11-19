# Adaptive quadrature

Role: ② strategy (`role-ladder`, `references/strategy.md`).
Unit: integrals.

Open this when you can evaluate the integrand anywhere and want the integral to a tolerance, with an error estimate, inside a traced function.

## Names

All of them live in `integrix.adapt`. Each is configuration only, and is called as `rule(fn, lower, upper, *, args=None)` to return an `integrix.rule.Integral` with `error` and `converged` filled in.

| Name | Use it for | How it works |
|---|---|---|
| `GaussKronrod(order=15, rtol=1e-8, atol=0.0, max_intervals=64)` | a finite interval; a peaked integrand | Bisects the subinterval with the largest error, in a static buffer of `max_intervals`. `order` is 15, 21, 31, 41, 51 or 61. |
| `TanhSinh(rtol=1e-10, atol=0.0, max_levels=6, min_levels=3)` | a singularity at an end; an infinite limit | A double-exponential change of variable, then the trapezoid rule, halving the step per level. |
| `Romberg(rtol=1e-10, atol=0.0, max_levels=10, min_levels=4)` | a smooth integrand on a finite interval | Richardson extrapolation of trapezoid sums on 2^L + 1 points. |

## Use it

```python
import jax
import jax.numpy as jnp

from integrix.adapt import GaussKronrod, Romberg, TanhSinh

# a peak far narrower than the interval
peak = lambda x, width: jnp.exp(-0.5 * (x / width) ** 2)
result = GaussKronrod(rtol=1e-6)(peak, -1.0, 1.0, args=0.01)
assert bool(result.converged)
assert jnp.allclose(result.value, 0.01 * jnp.sqrt(2.0 * jnp.pi), rtol=1e-4)
assert float(result.error) <= 1e-4 * float(result.value)

# it runs under jit and differentiates in the integrand's parameters
area = jax.jit(lambda width: GaussKronrod(rtol=1e-6)(peak, -1.0, 1.0, args=width).value)
slope = jax.grad(area)(0.01)
assert jnp.allclose(slope, jnp.sqrt(2.0 * jnp.pi), rtol=1e-3)

# a singular end: ∫₀¹ x^(−1/2) dx = 2
singular = TanhSinh(rtol=1e-6)(lambda x: 1.0 / jnp.sqrt(x), 0.0, 1.0)
assert jnp.allclose(singular.value, 2.0, rtol=1e-4)

# an infinite limit: ∫₀^∞ e^(−x) dx = 1
tail = TanhSinh(rtol=1e-6)(lambda x: jnp.exp(-x), 0.0, float("inf"))
assert jnp.allclose(tail.value, 1.0, rtol=1e-4)

# a smooth integrand
smooth = Romberg(rtol=1e-6)(jnp.exp, 0.0, 1.0)
assert jnp.allclose(smooth.value, jnp.e - 1.0, rtol=1e-5) and bool(smooth.converged)
```

## Rules

- Check `result.converged`. A rule that runs out of intervals or levels returns its best value with `converged` false; nothing is raised, because under `jit` it is a flag.
- The tolerance is `max(atol, rtol · |I|)` on the summed error.
- `GaussKronrod` differentiates through the final Kronrod rule on the final partition held fixed, in the parameters passed through `args` or closed over, and in the limits.
- An endpoint singularity needs `TanhSinh`, not `GaussKronrod`. Put the singularity at 0 by shifting the variable: the integrand sees x, so near a singular end at a ≠ 0 the distance x − a loses accuracy.
- An infinite limit is a Python or NumPy `inf`, for `TanhSinh` only.
- `min_levels` stops a coarse grid whose nodes all fall on the integrand's zeros from being accepted; raise it for an oscillating integrand.
- For samples on fixed nodes there is nothing to adapt: use `integrix.rule.integrate`.
