"""The running integral of an integrand to each node, adaptively per interval."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from integrix._src.adapt._gauss_kronrod import GaussKronrod
from integrix._src.rule._rule import Integral
from integrix.exceptions import ValidationError


__all__ = [
    "cumulative_adaptive",
]


def cumulative_adaptive(
    rule: GaussKronrod,
    fn: Callable[..., ArrayLike],
    nodes: ArrayLike,
    *,
    args: Any = None,
) -> Integral:
    """∫ from x₀ to each node xₖ of f, each interval integrated adaptively.

    Fₖ = ∑ⱼ≤ₖ ∫ from xⱼ₋₁ to xⱼ of f, with F₀ = 0: each interval between
    consecutive nodes gets its own adaptive Gauss–Kronrod integral (run under
    `jax.vmap`), and the interval integrals are prefix-summed with
    `jax.lax.associative_scan`, as `integrix.scan.cumulative` does with
    trapezoid increments.

    Args:
        rule: The adaptive rule for each interval, a `GaussKronrod`.
        fn: The integrand f(x), or f(x, args) when ``args`` is given.
        nodes: The N ≥ 1 nodes x₀, …, x_{N−1}, ascending or descending.
        args: Parameters passed to ``fn``, differentiable.

    Returns:
        An `integrix.rule.Integral` whose ``value`` is F, shape (N, …), whose
        ``error`` is the running sum of the interval errors, whose
        ``n_evaluations`` is their total and whose ``converged`` says every
        interval converged.

    Raises:
        ValidationError: If ``rule`` is not a `GaussKronrod` or ``nodes`` is
            not a non-empty vector.

    Examples:
        >>> import jax.numpy as jnp
        >>> from integrix.adapt import GaussKronrod
        >>> nodes = jnp.array([0.0, 0.5, 1.0, 2.0])
        >>> running = cumulative_adaptive(GaussKronrod(rtol=1e-6), jnp.exp, nodes)
        >>> bool(jnp.allclose(running.value, jnp.exp(nodes) - 1.0))
        True
    """
    if not isinstance(rule, GaussKronrod):
        raise ValidationError(
            f"'rule' must be a GaussKronrod, got {type(rule).__name__}"
        )
    nodes = jnp.asarray(nodes)
    if nodes.ndim != 1 or nodes.shape[0] == 0:
        raise ValidationError(
            f"'nodes' must be a vector of at least one node, got shape "
            f"{np.shape(nodes)}"
        )
    nodes = nodes.astype(jnp.result_type(nodes, float))
    if nodes.shape[0] == 1:
        # one node: F₀ = 0, with the integrand's output shape
        out = jax.eval_shape(lambda x: rule(fn, x, x, args=args).value, nodes[0])
        zeros = jnp.zeros((1, *out.shape), dtype=out.dtype)
        return Integral(
            value=zeros,
            error=zeros,
            n_evaluations=jnp.asarray(0),
            converged=jnp.asarray(True),
        )
    pieces = jax.vmap(lambda a, b: rule(fn, a, b, args=args))(nodes[:-1], nodes[1:])
    # Fₖ = ∑ⱼ≤ₖ Iⱼ, with F₀ = 0
    running = jax.lax.associative_scan(jnp.add, pieces.value)
    running_error = jax.lax.associative_scan(jnp.add, jnp.asarray(pieces.error))
    value = jnp.concatenate([jnp.zeros_like(running[:1]), running])
    error = jnp.concatenate([jnp.zeros_like(running_error[:1]), running_error])
    return Integral(
        value=value,
        error=error,
        n_evaluations=jnp.sum(pieces.n_evaluations),
        converged=jnp.all(pieces.converged),
    )
