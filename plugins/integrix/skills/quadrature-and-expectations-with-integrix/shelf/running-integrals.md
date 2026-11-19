# Running integrals

Role: ① primitive (`role-ladder`, `references/primitive.md`).
Unit: integrals.

Open this when you need the integral up to every node, not only the total: optical depth down a column, a cumulative distribution, a travel time along a path.

## Names

| Name | What it is |
|---|---|
| `integrix.scan.cumulative(rule, values, *, reverse=False, axis=-1, initial=0.0)` | The running trapezoid integral Fₖ of samples along one axis, by `jax.lax.associative_scan` in O(log N) depth. Returns an array with the samples' shape. |
| `integrix.adapt.cumulative_adaptive(rule, fn, nodes, *, args=None)` | ∫ from x₀ to each node of an integrand, each interval by adaptive Gauss–Kronrod. Returns an `Integral` whose `value` has shape `(N, …)`. |

## Use it

```python
import jax
import jax.numpy as jnp
from functax.quad import trapezoid

from integrix.adapt import GaussKronrod, cumulative_adaptive
from integrix.rule import integrate
from integrix.scan import cumulative

# optical depth from the top of a column: extinction on 50 levels, 3 columns
height = jnp.linspace(0.0, 10.0, 50)
extinction = jnp.exp(-height / 2.0) * jnp.array([[1.0], [2.0], [3.0]])  # (3, 50)
rule = trapezoid(height)
from_top = cumulative(rule, extinction, reverse=True)  # (3, 50)
assert from_top.shape == (3, 50)
assert jnp.allclose(from_top[:, -1], 0.0)  # nothing above the top level
# the entry at the ground is the whole column
assert jnp.allclose(from_top[:, 0], integrate(rule, extinction).value, rtol=1e-5)

# forward, from the first node, starting at a given value
forward = cumulative(rule, extinction[0], initial=1.0)
assert jnp.allclose(forward[0], 1.0)

# an integrand to a tolerance at each node
nodes = jnp.array([0.0, 0.5, 1.0, 2.0])
running = cumulative_adaptive(GaussKronrod(rtol=1e-6), jnp.exp, nodes)
assert jnp.allclose(running.value, jnp.exp(nodes) - 1.0, rtol=1e-5)
assert bool(running.converged)

# it differentiates like any prefix sum
grad = jax.grad(lambda scale: cumulative(rule, scale * extinction[0])[-1])(1.0)
assert jnp.allclose(grad, forward[-1] - 1.0, rtol=1e-5)
```

## Rules

- `cumulative` reads only the rule's `nodes`: the increments are the trapezoid rule on them, whatever weights the rule carries. Its last entry equals `integrate` with a trapezoid rule on the same nodes.
- The nodes may ascend or descend. `reverse=True` integrates from the last node towards the first without reversing the nodes, so every width keeps its sign.
- `cumulative_adaptive` takes a `GaussKronrod` only, and raises `ValidationError` for another rule. Its `error` is the running sum of the interval errors, and `converged` says every interval converged.
- Use `cumulative` for samples and `cumulative_adaptive` for an integrand you can evaluate anywhere.
