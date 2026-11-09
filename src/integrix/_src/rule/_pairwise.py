"""Pairwise (tree) summation over the leading axis."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Inexact


def pairwise_sum(terms: Inexact[Array, " node *rest"]) -> Inexact[Array, "*rest"]:
    """Sum over the leading axis by halving, so each term is rounded O(log N) times.

    A left-to-right sum rounds the running total N times, and in float32 the
    late terms of a long spectrum fall below its last bit; adding neighbours in
    pairs, then pairs of pairs, keeps the error at O(ε log N) instead of O(ε N).

    Algorithm:
        1. Pad the terms with zero rows to the next power of two, 2^⌈log₂ N⌉.
        2. Add each even row to the odd row after it, halving the count,
           until one row is left.

    Args:
        terms: The N ≥ 1 terms to add, along the leading axis.

    Returns:
        Their sum, with the leading axis removed.

    Examples:
        >>> import jax.numpy as jnp
        >>> bool(pairwise_sum(jnp.arange(5.0)) == 10.0)
        True
        >>> pairwise_sum(jnp.ones((3, 2))).shape
        (2,)
    """
    n_terms = terms.shape[0]
    padded_length = 1 << (n_terms - 1).bit_length()
    if padded_length > n_terms:
        padding = jnp.zeros((padded_length - n_terms, *terms.shape[1:]), terms.dtype)
        terms = jnp.concatenate([terms, padding])
    while terms.shape[0] > 1:
        # sₖ = t₂ₖ + t₂ₖ₊₁
        terms = terms[0::2] + terms[1::2]
    return terms[0]
