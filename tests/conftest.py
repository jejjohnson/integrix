"""Shared pytest configuration.

The reference comparisons run in float64, so JAX's 64-bit mode is enabled for
the whole suite, and jaxtyping checks every annotated array at runtime through
its import hook (`python-style`).
"""

from __future__ import annotations

import jax
from jaxtyping import install_import_hook


jax.config.update("jax_enable_x64", True)
# Installed before pytest imports integrix for doctest collection, so every
# integrix module imported afterwards is checked with beartype.
install_import_hook("integrix", "beartype.beartype")
