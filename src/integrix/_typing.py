"""Shared type aliases.

The aliases are written with [PEP 695](https://peps.python.org/pep-0695/)
``type`` statements (Python 3.12+). Array annotations use jaxtyping and live
with the modules that need them.
"""

from __future__ import annotations


__all__ = [
    "Axes",
]


type Axes = int | tuple[int, ...]
"""One array axis, or several, given by position (negative counts from the end)."""
