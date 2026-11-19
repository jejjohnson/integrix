# Errors, the version and what to install

Role: setup.
Unit: integrals.

Open this when a call raises, when you write an `except` clause around integrix, or when you set up a project that uses it.

## Names

| Name | What it is |
|---|---|
| `integrix.IntegrixError` | The base class of every exception integrix raises. |
| `integrix.ValidationError` | An argument failed validation. Also a `ValueError`. |
| `integrix.exceptions.IntegrixError`, `integrix.exceptions.ValidationError` | The same two classes at the module that defines them. |
| `integrix.__version__` | The version string, `major.minor.patch`. |

The top level holds only these names. Everything else is reached through its area module: `integrix.rule`, `integrix.scan`, `integrix.adapt`, `integrix.expect`, `integrix.march`, `integrix.adjoint`.

## Use it

```python
import jax.numpy as jnp

import integrix
from integrix import IntegrixError, ValidationError
from integrix.adapt import GaussKronrod
from integrix.rule import integrate

assert integrix.exceptions.ValidationError is ValidationError
assert integrix.exceptions.IntegrixError is IntegrixError
assert issubclass(ValidationError, IntegrixError) and issubclass(ValidationError, ValueError)
assert all(part.isdigit() for part in integrix.__version__.split(".")[:3])

# two weights against three samples
try:
    integrate(jnp.array([0.5, 0.5]), jnp.array([1.0, 2.0, 3.0]))
except ValidationError as err:
    message = str(err)
assert message

# configuration is checked when the object is built
try:
    GaussKronrod(order=17)
except ValueError as err:  # a ValidationError is a ValueError
    assert isinstance(err, IntegrixError)
```

## Rules

- Catch `IntegrixError` to trap the package, or `ValueError` in code that must not import integrix.
- Configuration is checked when the object is built: `GaussKronrod(order=17)` raises before any integrand is evaluated.
- A check on a traced value, such as a positive weight in `log_integrate` or a spacing in `stable_step`, still runs under `jax.jit`.
- An adaptive rule that misses its tolerance does not raise: it returns `converged=False`.
- integrix installs JAX, Equinox, diffrax, lineax, optimistix, einx, functax and gaussx with it. SciPy and quadax are test references only, never imported by the library.
- The library never calls `jax.jit`: jit once at the top of your own code.
- The references in integrix's own tests run in 64-bit mode; turn on `jax_enable_x64` when you need tolerances below single precision.
