"""The number and the lengths of the steps of a fixed-step run over a span."""

from __future__ import annotations

import math

import numpy as np

from integrix.exceptions import ValidationError


__all__ = [
    "count_steps",
    "step_lengths",
]

type Number = int | float | np.integer | np.floating
"""A real number on the host: a Python or NumPy scalar."""

#: The relative slack within which a quotient is taken as a whole number.
_SNAP = 1e-9


def _checked(span: Number, step: Number) -> tuple[float, float]:
    """The span and the step as Python floats, checked.

    Raises:
        ValidationError: If ``span`` is negative or not finite, or ``step`` is
            not positive and finite.
    """
    total, size = float(span), float(step)
    if not math.isfinite(total) or total < 0.0:
        raise ValidationError(f"'span' must be finite and at least 0, got {span!r}")
    if not math.isfinite(size) or size <= 0.0:
        raise ValidationError(f"'step' must be finite and positive, got {step!r}")
    return total, size


def count_steps(span: Number, step: Number) -> int:
    """K = ⌈T/Δt⌉, with a quotient within 10⁻⁹ K of a whole number taken as it.

    Plain ⌈T/Δt⌉ adds a step of zero length when the quotient is a whole number
    that floating point rounds just above: 2.1 s in steps of 0.15 s is 14 steps,
    and not 15. Take the span of times counted from an epoch near the run: the
    difference of two Unix-epoch times has already rounded.

    Args:
        span: T ≥ 0, the length of the run, a Python number.
        step: Δt > 0, the length of a step, a Python number.

    Returns:
        K: 0 for T = 0, and at least 1 for any T > 0, so a span much shorter than
        a step is still one step.

    Raises:
        ValidationError: If ``span`` is negative or not finite, or ``step`` is not
            positive and finite.

    Examples:
        >>> import math
        >>> count_steps(2.1, 0.15), math.ceil(2.1 / 0.15)
        (14, 15)
        >>> count_steps(122.0, 5.0)
        25
        >>> count_steps(0.0, 5.0), count_steps(1e-12, 5.0)
        (0, 1)
    """
    total, size = _checked(span, step)
    # q = T/Δt, and n its nearest whole number
    quotient = total / size
    nearest = round(quotient)
    if abs(quotient - nearest) <= _SNAP * max(1.0, nearest):
        return nearest if total == 0.0 else max(nearest, 1)
    return math.ceil(quotient)


def step_lengths(span: Number, step: Number) -> np.ndarray:
    """The K step lengths: Δt, and T − (K − 1) Δt for the last, on the host.

    They sum to T to rounding; K is `count_steps`. Each is in (0, Δt], except
    that the last is up to 10⁻⁹ K Δt longer where the quotient was taken as a
    whole number.

    Args:
        span: T ≥ 0, as in `count_steps`.
        step: Δt > 0, as in `count_steps`.

    Returns:
        The K lengths, in float64; empty for T = 0.

    Raises:
        ValidationError: As `count_steps` does.

    Examples:
        >>> step_lengths(12.0, 5.0).tolist()
        [5.0, 5.0, 2.0]
        >>> step_lengths(2.1, 0.15).shape
        (14,)
        >>> step_lengths(0.0, 5.0).shape
        (0,)
    """
    total, size = _checked(span, step)
    n_steps = count_steps(total, size)
    lengths = np.full(n_steps, size, dtype=np.float64)
    if n_steps:
        # Δt_K = T − (K − 1) Δt
        lengths[-1] = total - size * (n_steps - 1)
    return lengths
