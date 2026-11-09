"""The step count of a scan from its stacked inputs or an explicit length."""

from __future__ import annotations

import operator
from typing import Any

import jax
import numpy as np

from integrix.exceptions import ValidationError


def positive_count(value: Any, name: str) -> int:
    """``value`` as a positive Python integer (not a bool), else `ValidationError`."""
    if isinstance(value, bool):
        raise ValidationError(f"'{name}' must be a positive integer, got {value!r}")
    try:
        count = operator.index(value)
    except TypeError as err:
        raise ValidationError(
            f"'{name}' must be a positive integer, got {value!r}"
        ) from err
    if count < 1:
        raise ValidationError(f"'{name}' must be a positive integer, got {value!r}")
    return count


def scan_length(xs: Any, length: int | None, caller: str) -> int:
    """The number of steps T: the shared leading axis of ``xs``, or ``length``.

    Raises:
        ValidationError: If a leaf of ``xs`` has no leading axis, the leaves
            disagree, ``length`` disagrees with them, or there are no leaves
            and no ``length``.
    """
    leaves = jax.tree.leaves(xs)
    if not leaves:
        if length is None:
            raise ValidationError(
                f"{caller} needs at least one stacked input in xs, or a length."
            )
        return length
    if any(np.ndim(leaf) == 0 for leaf in leaves):
        raise ValidationError(
            f"{caller} needs every leaf of xs to have a leading (step) axis"
        )
    lengths = {leaf.shape[0] for leaf in leaves}
    if len(lengths) != 1:
        raise ValidationError(
            f"{caller} needs every leaf of xs to share one leading length, "
            f"got {sorted(lengths)}"
        )
    (steps,) = lengths
    if length is not None and length != steps:
        raise ValidationError(f"{caller} got length={length} but xs has {steps} steps")
    return steps
