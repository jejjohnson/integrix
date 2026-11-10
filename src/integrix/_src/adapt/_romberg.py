"""Romberg integration: trapezoid estimates extrapolated by Richardson."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike, ScalarLike

from integrix._src.adapt._levels import first_converged
from integrix._src.adapt._tanh_sinh import check_levels, check_tolerances
from integrix._src.rule._pairwise import pairwise_sum
from integrix._src.rule._rule import Integral
from integrix.exceptions import ValidationError


__all__ = [
    "Romberg",
]


class Romberg(eqx.Module):
    """Romberg integration of a smooth integrand on a finite interval.

    The trapezoid estimates T(h), T(h/2), … have the error expansion
    T(h) = I + c₁h² + c₂h⁴ + …, so Richardson extrapolation
    Rℓ,ₘ = Rℓ,ₘ₋₁ + (Rℓ,ₘ₋₁ − Rℓ₋₁,ₘ₋₁)/(4ᵐ − 1) removes one term per column.
    It suits smooth and periodic-like integrands; a singular one needs
    tanh–sinh and a peaked one Gauss–Kronrod.

    Algorithm:
        1. Evaluate f once on 2^L + 1 equally spaced points.
        2. For each level ℓ = 0, …, L, the trapezoid sum Tℓ = Rℓ,₀ on every
           2^{L−ℓ}-th point, and its extrapolations Rℓ,ₘ, m = 1, …, ℓ.
        3. The diagonal Rℓ,ℓ is level ℓ's estimate and |Rℓ,ℓ − Rℓ₋₁,ℓ₋₁| its
           error; return the first level within max(atol, rtol |I|), or the
           last, with ``converged``.

    Attributes:
        rtol: The relative tolerance, ≥ 0.
        atol: The absolute tolerance, ≥ 0.
        max_levels: The number of halvings L ≥ 1 (2^L + 1 evaluations).
        min_levels: The coarsest level ℓ ≥ 1 that may be accepted. A periodic
            integrand whose zeros fall on a coarse grid (sin²(8πx) on [0, 1]
            vanishes at every node up to 9 points) fools any level difference,
            so levels below this are never accepted.

    Raises:
        ValidationError: If a tolerance is negative or not finite, or
            ``max_levels`` or ``min_levels`` is out of range.

    References:
        Romberg, W. (1955). Vereinfachte numerische Integration. Det Kongelige
        Norske Videnskabers Selskab Forhandlinger 28, 30–36.

    Examples:
        >>> import jax.numpy as jnp
        >>> result = Romberg(rtol=1e-6)(jnp.exp, 0.0, 1.0)
        >>> bool(jnp.allclose(result.value, jnp.e - 1.0)), bool(result.converged)
        (True, True)
    """

    rtol: float = eqx.field(static=True, default=1e-10)
    atol: float = eqx.field(static=True, default=0.0)
    max_levels: int = eqx.field(static=True, default=10)
    min_levels: int = eqx.field(static=True, default=4)

    def __check_init__(self) -> None:
        check_tolerances(self.rtol, self.atol)
        check_levels(self.max_levels, self.min_levels)

    def __call__(
        self,
        fn: Callable[..., ArrayLike],
        lower: ScalarLike,
        upper: ScalarLike,
        *,
        args: Any = None,
    ) -> Integral:
        """Integrate ``fn`` from ``lower`` to ``upper``.

        Args:
            fn: The integrand f(x), or f(x, args) when ``args`` is given, for a
                scalar x; it may return an array.
            lower: The lower limit a, finite.
            upper: The upper limit b, finite.
            args: Parameters passed to ``fn``, differentiable.

        Returns:
            An `integrix.rule.Integral` with the value, the change of the
            diagonal as the error, the number of evaluations and
            ``converged``.

        Raises:
            ValidationError: If a limit is not a scalar or a host limit is not
                finite.
        """
        for name, value in (("lower", lower), ("upper", upper)):
            if np.ndim(value) != 0:
                raise ValidationError(
                    f"'{name}' must be a scalar, got {np.shape(value)}"
                )
            if not isinstance(value, jax.Array) and not np.isfinite(np.asarray(value)):
                raise ValidationError(
                    f"'{name}' must be finite (tanh–sinh handles infinite limits), "
                    f"got {value!r}"
                )
        call = (lambda x: fn(x)) if args is None else (lambda x: fn(x, args))
        dtype = jnp.result_type(lower, upper, float)
        a = jnp.asarray(lower, dtype=dtype)
        b = jnp.asarray(upper, dtype=dtype)
        n_points = 2**self.max_levels + 1
        points = a + (b - a) * jnp.linspace(0.0, 1.0, n_points, dtype=dtype)
        samples = jax.vmap(lambda x: jnp.asarray(call(x)))(points)
        width = (b - a).astype(samples.dtype)

        table: list[list[Any]] = []
        for level in range(self.max_levels + 1):
            stride = 2 ** (self.max_levels - level)
            level_samples = samples[::stride]
            step = width / 2**level
            # Tℓ = hℓ (∑ⱼ fⱼ − (f₀ + f_N)/2)
            interior = pairwise_sum(level_samples) - 0.5 * (
                level_samples[0] + level_samples[-1]
            )
            row = [step * interior]
            for column in range(1, level + 1):
                # Rℓ,ₘ = Rℓ,ₘ₋₁ + (Rℓ,ₘ₋₁ − Rℓ₋₁,ₘ₋₁)/(4ᵐ − 1)
                previous = table[level - 1][column - 1]
                row.append(row[-1] + (row[-1] - previous) / (4**column - 1))
            table.append(row)
        diagonal = jnp.stack([row[-1] for row in table])
        errors = jnp.concatenate(
            [
                jnp.full_like(diagonal[:1], jnp.inf),
                jnp.abs(diagonal[1:] - diagonal[:-1]),
            ]
        )
        value, error, _, converged = first_converged(
            diagonal,
            errors,
            rtol=self.rtol,
            atol=self.atol,
            min_level=self.min_levels,
        )
        return Integral(
            value=value,
            error=error,
            n_evaluations=jnp.asarray(n_points),
            converged=converged,
        )
