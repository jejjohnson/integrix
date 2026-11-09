"""Truncated backpropagation through time: `Truncated` and `truncated_scan`.

Moved from pipekit-jax (``packages/pipekit-jax/src/pipekit_jax/adjoints.py``
at pipekit ``7d02d8e``) with its interface unchanged. A truncated rollout cuts
the gradient through the carry before the last k steps, so reverse mode keeps
O(k) of the history; for chaotic dynamics it is also a regulariser, since the
exact adjoint grows like e^{λ_max T}.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from integrix.exceptions import ValidationError


__all__ = [
    "Truncated",
    "truncated_scan",
]


class Truncated(eqx.Module):
    """The truncated adjoint: gradients flow through the carry of the last k steps.

    The adjoint option for `integrix.march.run` and the integrix form of
    pipekit-cycle's ``TruncatedAdjoint``; `truncated_scan` is its function
    form for any `jax.lax.scan` rollout. ``k ≥ T`` is the exact adjoint.

    Attributes:
        k: The number of trailing steps whose carry propagates gradients,
            at least 1.

    Raises:
        ValidationError: If ``k`` is not a positive integer.

    Examples:
        >>> Truncated(k=5).k
        5
    """

    k: int = eqx.field(static=True, default=1)

    def __check_init__(self) -> None:
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k < 1:
            raise ValidationError(f"'k' must be a positive integer, got {self.k!r}")


def truncated_scan[Carry, X, Y](
    f: Callable[[Carry, X], tuple[Carry, Y]],
    init: Carry,
    xs: X,
    *,
    k: int = 1,
    length: int | None = None,
) -> tuple[Carry, Y]:
    """`jax.lax.scan` with gradients truncated to the trailing k steps.

    The carry leaving every step before T − k is wrapped in
    `jax.lax.stop_gradient`, so reverse mode sees only the last k transitions
    through the carry. Each step's own output yₜ still receives gradients with
    respect to that step's inputs; only the flow through history is cut (the
    ROAD-EnKF and truncated-BPTT training regime). ``k ≥ T`` is a plain scan.
    Forward values are identical to `jax.lax.scan`'s.

    Args:
        f: The scan body ``(carry, x) -> (carry, y)``.
        init: The initial carry.
        xs: The stacked per-step inputs, leading axis T, or ``None`` with
            ``length``.
        k: The number of trailing steps that propagate gradients through the
            carry, at least 1.
        length: The number of steps T when ``xs`` has no leaves.

    Returns:
        ``(final_carry, ys)``, exactly as `jax.lax.scan` returns them.

    Raises:
        ValueError: If ``k < 1``, or ``xs`` has no leading axis and no
            ``length`` is given (a `ValidationError`, which is a
            `ValueError`).

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> def step(carry, x):
        ...     new = 0.9 * carry + x
        ...     return new, new
        >>> _, ys = truncated_scan(step, 1.0, jnp.zeros(4), k=2)
        >>> _, plain = jax.lax.scan(step, 1.0, jnp.zeros(4))
        >>> bool(jnp.array_equal(ys, plain))
        True
    """
    if k < 1:
        raise ValidationError(f"truncated_scan needs k >= 1; got k={k}.")
    leaves = jax.tree.leaves(xs)
    if leaves:
        steps = leaves[0].shape[0]
    elif length is not None:
        steps = length
    else:
        raise ValidationError(
            "truncated_scan needs at least one stacked input in xs, or a length."
        )
    cutoff = steps - k

    def body(carry: Carry, indexed: tuple[Any, X]) -> tuple[Carry, Y]:
        i, x = indexed
        new_carry, y = f(carry, x)
        # jnp.where keeps the graph differentiable in the trailing window and
        # routes earlier steps through stop_gradient.
        cut = i < cutoff
        new_carry = jax.tree.map(
            lambda c: jnp.where(cut, jax.lax.stop_gradient(c), c), new_carry
        )
        return new_carry, y

    return jax.lax.scan(body, init, (jnp.arange(steps), xs), length=steps)
