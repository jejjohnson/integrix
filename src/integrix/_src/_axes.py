"""Bookkeeping shared by the areas: axes, moving them first, and working dtypes."""

from __future__ import annotations

import operator
from typing import Any

import einx
import jax.numpy as jnp
from jaxtyping import Array, Inexact

from integrix._typing import Axes
from integrix.exceptions import ValidationError


def normalise_axes(axis: Axes | None, ndim: int, n_axes: int) -> tuple[int, ...]:
    """The integrated axes as non-negative positions, ``n_axes`` of them.

    ``None`` takes the last ``n_axes`` axes.

    Raises:
        ValidationError: If an axis is not an integer, the count is wrong, an
            axis is out of range, or an axis repeats.
    """
    if axis is None:
        axis = tuple(range(-n_axes, 0))
    try:
        if isinstance(axis, tuple):
            axes = tuple(operator.index(position) for position in axis)
        else:
            axes = (operator.index(axis),)
    except TypeError as err:
        raise ValidationError(
            f"'axis' must be an integer or a tuple of integers, got {axis!r}"
        ) from err
    if len(axes) != n_axes:
        raise ValidationError(
            f"'axis' must name one axis per weights axis ({n_axes}), got {axis}"
        )
    normalised = []
    for position in axes:
        if not -ndim <= position < ndim:
            raise ValidationError(
                f"'axis' must be within the values' {ndim} axes, got {position}"
            )
        normalised.append(position % ndim)
    if len(set(normalised)) != len(normalised):
        raise ValidationError(f"'axis' must not repeat an axis, got {axis}")
    return tuple(normalised)


def check_node_shape(
    values: Array, axes: tuple[int, ...], node_shape: tuple[int, ...]
) -> None:
    """Raise `ValidationError` unless ``values`` has one entry per node on ``axes``."""
    shape = tuple(values.shape[position] for position in axes)
    if shape != node_shape:
        raise ValidationError(
            f"'values' must have one entry per node along axes {axes}: got shape "
            f"{shape}, expected {node_shape}"
        )


def as_inexact(values: Array) -> Inexact[Array, " ..."]:
    """The values unchanged when floating or complex, else in the default float."""
    if jnp.issubdtype(values.dtype, jnp.inexact):
        return values
    return values.astype(jnp.result_type(float))


def working_dtype(values_dtype: Any, *coefficient_dtypes: Any) -> Any:
    """The dtype a sum runs in: the values' precision, complex if any input is.

    Floating values keep their dtype (precision is the caller's), so float64
    weights never promote float32 values. Integer values take the weights'
    floating dtype. A complex weight promotes real values to the complex type
    of their own precision rather than losing its imaginary part.
    """
    if jnp.issubdtype(values_dtype, jnp.inexact):
        dtype = values_dtype
    else:
        dtype = coefficient_dtypes[0]
    if any(jnp.issubdtype(other, jnp.complexfloating) for other in coefficient_dtypes):
        dtype = jnp.promote_types(dtype, jnp.complex64)
    return dtype


def _names(ndim: int) -> list[str]:
    """One einx axis name per position: d0, d1, …"""
    return [f"d{position}" for position in range(ndim)]


def nodes_first(values: Array, axes: tuple[int, ...]) -> Inexact[Array, " node ..."]:
    """Move ``axes`` first and flatten them into one node axis, (N, …)."""
    names = _names(values.ndim)
    node_names = " ".join(names[position] for position in axes)
    rest_names = " ".join(
        name for position, name in enumerate(names) if position not in axes
    )
    pattern = f"{' '.join(names)} -> ({node_names}) {rest_names}"
    return as_inexact(einx.id(pattern, values))


def flatten(weights: Array) -> Inexact[Array, " node"]:
    """Flatten a weights array of any rank into one node axis."""
    names = " ".join(_names(weights.ndim))
    return einx.id(f"{names} -> ({names})", weights)


def move_first(values: Array, axis: int) -> Array:
    """Move one (non-negative) axis to the front, keeping the others in order."""
    names = _names(values.ndim)
    moved = [names[axis], *(name for name in names if name != names[axis])]
    return einx.id(f"{' '.join(names)} -> {' '.join(moved)}", values)


def move_back(values: Array, axis: int) -> Array:
    """Undo `move_first`: send the leading axis back to position ``axis``."""
    names = _names(values.ndim)
    moved = [names[axis], *(name for name in names if name != names[axis])]
    return einx.id(f"{' '.join(moved)} -> {' '.join(names)}", values)
