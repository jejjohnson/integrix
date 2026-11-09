"""Resolve an adjoint spec, pipekit-cycle's or diffrax's, to a diffrax adjoint.

Moved from pipekit-jax (``adjoints.py`` at pipekit ``7d02d8e``). pipekit-cycle
sits above integrix (Core tier 2), so its specs are matched structurally, by
class name and fields, and never imported.
"""

from __future__ import annotations

from typing import Any

import diffrax as dfx

from integrix.exceptions import ValidationError


__all__ = [
    "to_diffrax_adjoint",
]


def to_diffrax_adjoint(spec: Any) -> dfx.AbstractAdjoint:
    """Map an adjoint spec onto a `diffrax.AbstractAdjoint`.

    Specs are matched by class name, so pipekit-cycle's and any structurally
    identical spec are accepted interchangeably:

    - ``RecursiveCheckpointAdjoint(checkpoints)`` →
      `diffrax.RecursiveCheckpointAdjoint` with the same checkpoints, the
      recommended default;
    - ``DirectAdjoint()`` → `diffrax.DirectAdjoint`;
    - ``BacksolveAdjoint()`` → `diffrax.BacksolveAdjoint`. Continuous
      adjoints diverge for chaotic or stiff dynamics; never a default;
    - ``ImplicitAdjoint()`` → `diffrax.ImplicitAdjoint`, for steady-state
      solves only (a derivative at a fixed point is argminax's business);
    - ``TruncatedAdjoint`` raises: truncation belongs to an outer rollout,
      not to a single solve; use `integrix.adjoint.Truncated` with
      `integrix.march.run`, or `truncated_scan`, at that layer.

    A ready-made `diffrax.AbstractAdjoint` passes through unchanged.

    Args:
        spec: A spec from pipekit-cycle's ``adjoints``, any structurally
            identical object, or a diffrax adjoint.

    Returns:
        The diffrax adjoint.

    Raises:
        ValueError: For ``TruncatedAdjoint`` or an unrecognised spec (a
            `ValidationError`, which is a `ValueError`).

    Examples:
        >>> import diffrax as dfx
        >>> class RecursiveCheckpointAdjoint:
        ...     checkpoints = 8
        >>> adjoint = to_diffrax_adjoint(RecursiveCheckpointAdjoint())
        >>> isinstance(adjoint, dfx.RecursiveCheckpointAdjoint), adjoint.checkpoints
        (True, 8)
    """
    if isinstance(spec, dfx.AbstractAdjoint):
        return spec
    name = type(spec).__name__
    if name == "RecursiveCheckpointAdjoint":
        return dfx.RecursiveCheckpointAdjoint(
            checkpoints=getattr(spec, "checkpoints", None)
        )
    if name == "DirectAdjoint":
        return dfx.DirectAdjoint()
    if name == "BacksolveAdjoint":
        return dfx.BacksolveAdjoint()
    if name == "ImplicitAdjoint":
        return dfx.ImplicitAdjoint()
    if name in ("TruncatedAdjoint", "Truncated"):
        raise ValidationError(
            "TruncatedAdjoint applies to an outer rollout (assimilation cycle "
            "or solver iteration), not to a single ODE/SDE solve — use "
            "integrix.adjoint.Truncated with integrix.march.run, or "
            "integrix.adjoint.truncated_scan, at that layer."
        )
    raise ValidationError(f"Unrecognised adjoint spec: {spec!r}")
