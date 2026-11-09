"""Shared type aliases and protocols.

The aliases are written with [PEP 695](https://peps.python.org/pep-0695/)
``type`` statements (Python 3.12+). Array annotations use jaxtyping and live
with the modules that need them.
"""

from __future__ import annotations

from typing import Any, Protocol, SupportsIndex, runtime_checkable


__all__ = [
    "Axes",
    "RuleLike",
]


type Axes = SupportsIndex | tuple[SupportsIndex, ...]
"""One array axis, or several, given by position (negative counts from the end).

Any integer works, a Python ``int`` or a NumPy integer, since positions are
read with `operator.index`.
"""


@runtime_checkable
class RuleLike(Protocol):
    """Anything with quadrature ``weights`` (and, to be evaluated, ``nodes``).

    functax's rules (``functax.quad``) satisfy it structurally: integrix reads
    their ``weights`` wₙ and, when it evaluates an integrand, their ``nodes``
    xₙ, and never builds a rule itself.

    Examples:
        >>> import jax.numpy as jnp
        >>> import equinox as eqx
        >>> class Midpoint(eqx.Module):
        ...     nodes: jnp.ndarray
        ...     weights: jnp.ndarray
        >>> isinstance(Midpoint(jnp.array([0.5]), jnp.array([1.0])), RuleLike)
        True
    """

    @property
    def weights(self) -> Any:
        """The weights wₙ, one per node."""
        ...
