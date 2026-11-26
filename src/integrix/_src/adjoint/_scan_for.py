"""The scan that runs a loop of one's own under an adjoint: `scan_for`."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax

from integrix._src.adjoint._checkpoint import Checkpointed, checkpointed_scan
from integrix._src.adjoint._truncated import Truncated, truncated_scan
from integrix.exceptions import ValidationError


__all__ = ["scan_for"]


def scan_for(adjoint: Any) -> Callable[..., tuple[Any, Any]]:
    """The scan that runs a loop under an adjoint.

    ``None`` gives `jax.lax.scan`, `Checkpointed` gives `checkpointed_scan` with
    its ``every``, and `Truncated` gives `truncated_scan` with its ``k``. Forward
    values are those of `jax.lax.scan` in every case; only reverse mode differs.

    Args:
        adjoint: `Checkpointed`, `Truncated`, or ``None`` for plain reverse mode,
            which stores every carry. Anything else is refused.

    Returns:
        A function ``scan(f, init, xs) -> (carry, ys)``, called as `jax.lax.scan`
        is.

    Raises:
        ValidationError: If ``adjoint`` is anything else, such as a diffrax
            adjoint (which `to_diffrax_adjoint` maps for a diffrax solve) or the
            class `Truncated` in place of an instance.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from integrix.adjoint import Checkpointed, Truncated, scan_for
        >>> def step(carry, x):
        ...     new = 0.9 * carry + x
        ...     return new, new
        >>> xs = jnp.arange(6.0)
        >>> _, plain = jax.lax.scan(step, 1.0, xs)

        The forward values are the same under each adjoint:

        >>> _, ys = scan_for(None)(step, 1.0, xs)
        >>> bool(jnp.array_equal(ys, plain))
        True
        >>> _, ys = scan_for(Checkpointed(every=2))(step, 1.0, xs)
        >>> bool(jnp.array_equal(ys, plain))
        True

        `Truncated` cuts the gradient through all but the last k steps:

        >>> def last(init, adjoint):
        ...     carry, _ = scan_for(adjoint)(step, init, xs)
        ...     return carry
        >>> round(float(jax.grad(last)(1.0, None)), 6)  # 0.9⁶
        0.531441
        >>> float(jax.grad(last)(1.0, Truncated(k=2)))
        0.0
    """
    if adjoint is None:
        return jax.lax.scan
    if isinstance(adjoint, Checkpointed):
        every = adjoint.every

        def checkpointed(f: Any, init: Any, xs: Any) -> tuple[Any, Any]:
            return checkpointed_scan(f, init, xs, every=every)

        return checkpointed
    if isinstance(adjoint, Truncated):
        k = adjoint.k

        def truncated(f: Any, init: Any, xs: Any) -> tuple[Any, Any]:
            return truncated_scan(f, init, xs, k=k)

        return truncated
    name = adjoint.__name__ if isinstance(adjoint, type) else type(adjoint).__name__
    raise ValidationError(
        f"'adjoint' of a scan must be None, Checkpointed or Truncated (an "
        f"instance), got {name}; a diffrax adjoint applies to a diffrax solve, "
        f"through integrix.adjoint.to_diffrax_adjoint"
    )
