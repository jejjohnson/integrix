"""Adjoints through time: truncated and checkpointed backward sweeps.

`Truncated` (with its function form `truncated_scan`) cuts the gradient
through the carry before the last k steps. `to_diffrax_adjoint` resolves an
adjoint spec (pipekit-cycle's, matched structurally, or a diffrax adjoint) to
a diffrax adjoint. Adjoints at an optimum (implicit, k-step) are argminax's;
backward sweeps through a forward run are integrix's.
"""

from __future__ import annotations

from integrix._src.adjoint._diffrax import to_diffrax_adjoint
from integrix._src.adjoint._truncated import Truncated, truncated_scan


__all__ = [
    "Truncated",
    "to_diffrax_adjoint",
    "truncated_scan",
]
