"""Pick the first refinement level whose estimate meets the tolerance."""

from __future__ import annotations

import einx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Inexact


def first_converged(
    estimates: Inexact[Array, " level *out"],
    errors: Inexact[Array, " level *out"],
    *,
    rtol: float,
    atol: float,
    min_level: int = 1,
) -> tuple[Inexact[Array, "*out"], Inexact[Array, "*out"], Array, Bool[Array, ""]]:
    """The estimate and error of the first level within max(atol, rtol |I|).

    Every level is computed (static shapes); the first one from ``min_level``
    on whose every entry is finite and meets the tolerance is returned, or
    the last when none does. A non-finite estimate or error never passes,
    and the floor guards against a coarse grid whose nodes all land on the
    integrand's zeros.

    Returns:
        ``(value, error, level, converged)``.
    """
    n_levels = estimates.shape[0]
    bound = jnp.maximum(atol, rtol * jnp.abs(estimates))
    # a level passes when every output entry is within its bound
    within = (errors <= bound) & jnp.isfinite(errors) & jnp.isfinite(estimates)
    passes = einx.all("level ... -> level", within)
    passes = passes & (jnp.arange(n_levels) >= min_level)
    converged = jnp.any(passes)
    level = jnp.where(converged, jnp.argmax(passes), n_levels - 1)
    return estimates[level], errors[level], level, converged
