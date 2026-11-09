"""Checkpointed backward sweeps: `Checkpointed` and `checkpointed_scan`.

A checkpointed rollout stores the carry every k steps on the forward run and
recomputes the k steps between two checkpoints on the backward run: memory
O(T/k + k) instead of O(T), minimised near k ≈ √T, for about one extra forward
pass. It is the integrix loop's counterpart of diffrax's
`diffrax.RecursiveCheckpointAdjoint`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import einx
import equinox as eqx
import jax
import jax.numpy as jnp

from integrix._src.adjoint._scan import positive_count, scan_length


__all__ = [
    "Checkpointed",
    "checkpointed_scan",
]


class Checkpointed(eqx.Module):
    """The checkpointed adjoint: store every k-th state, recompute between them.

    The adjoint option for `integrix.march.run`: an `AbstractStepper`'s loop
    runs as `checkpointed_scan`, and a diffrax solver gets
    ``diffrax.RecursiveCheckpointAdjoint(checkpoints=⌈N / k⌉)``, since
    diffrax counts checkpoints rather than spacing them.

    Attributes:
        every: The spacing k ≥ 1 of the stored states; k ≈ √N minimises the
            memory O(N/k + k).

    Raises:
        ValidationError: If ``every`` is not a positive integer.

    Examples:
        >>> Checkpointed(every=100).every
        100
    """

    every: int = eqx.field(static=True, default=50)

    def __check_init__(self) -> None:
        positive_count(self.every, "every")


def checkpointed_scan[Carry, X, Y](
    f: Callable[[Carry, X], tuple[Carry, Y]],
    init: Carry,
    xs: X,
    *,
    every: int,
    length: int | None = None,
) -> tuple[Carry, Y]:
    """`jax.lax.scan` that stores the carry only every k steps for reverse mode.

    The T steps run as ⌊T/k⌋ blocks of k, an outer `jax.lax.scan` over blocks
    whose body is a `jax.checkpoint`-ed inner scan, then a checkpointed tail
    of T mod k steps. Reverse mode keeps the carries at block boundaries and
    recomputes each block's interior, so it stores O(T/k + k) states instead
    of T. Forward values, and gradients, are those of the plain scan.

    Args:
        f: The scan body ``(carry, x) -> (carry, y)``.
        init: The initial carry.
        xs: The stacked per-step inputs, leading axis T, or ``None`` with
            ``length``.
        every: The block length k ≥ 1.
        length: The number of steps T when ``xs`` has no leaves.

    Returns:
        ``(final_carry, ys)``, exactly as `jax.lax.scan` returns them.

    Raises:
        ValidationError: If ``every`` is not a positive integer, a leaf of
            ``xs`` has no leading axis, the leaves or ``length`` disagree, or
            there are no stacked inputs and no ``length``.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> def step(carry, x):
        ...     new = jnp.sin(carry) + x
        ...     return new, new
        >>> xs = jnp.linspace(0.0, 1.0, 10)
        >>> _, ys = checkpointed_scan(step, 0.5, xs, every=3)
        >>> _, plain = jax.lax.scan(step, 0.5, xs)
        >>> bool(jnp.array_equal(ys, plain))
        True
    """
    block = positive_count(every, "every")
    steps = scan_length(xs, length, "checkpointed_scan")
    if steps == 0:
        # nothing to checkpoint: the empty scan, with its empty outputs
        return jax.lax.scan(f, init, xs, length=0)
    n_blocks, remainder = divmod(steps, block)
    head = n_blocks * block

    @jax.checkpoint
    def run_block(carry: Carry, block_xs: Any) -> tuple[Carry, Y]:
        return jax.lax.scan(f, carry, block_xs, length=block)

    @jax.checkpoint
    def run_tail(carry: Carry, tail_xs: Any) -> tuple[Carry, Y]:
        return jax.lax.scan(f, carry, tail_xs, length=remainder)

    carry = init
    pieces = []
    if n_blocks:
        # (T, …) → (blocks, k, …) for the head of whole blocks
        blocked = jax.tree.map(
            lambda x: einx.id("(blocks k) ... -> blocks k ...", x[:head], k=block),
            xs,
        )
        carry, blocked_ys = jax.lax.scan(run_block, carry, blocked, length=n_blocks)
        # (blocks, k, …) → (blocks · k, …)
        pieces.append(
            jax.tree.map(
                lambda y: einx.id("blocks k ... -> (blocks k) ...", y), blocked_ys
            )
        )
    if remainder:
        tail = jax.tree.map(lambda x: x[head:], xs)
        carry, tail_ys = run_tail(carry, tail)
        pieces.append(tail_ys)
    if len(pieces) == 1:
        return carry, pieces[0]
    ys = jax.tree.map(lambda a, b: jnp.concatenate([a, b]), *pieces)
    return carry, ys
