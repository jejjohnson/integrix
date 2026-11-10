"""Low-discrepancy point sets on [0, 1)^D: Sobol (Joe–Kuo) and Halton.

The points are built on the host with NumPy, as constants: they depend only
on the static size n and dimension D, never on traced values. Randomisation
(a digital shift for Sobol, a Cranley–Patterson rotation for Halton) happens
in JAX, from a key.
"""

from __future__ import annotations

import functools

import numpy as np

from integrix._src.expect._sobol_table import M_INIT, POLY
from integrix.exceptions import ValidationError


__all__ = [
    "SOBOL_BITS",
    "SOBOL_MAX_DIM",
    "halton_points",
    "sobol_integers",
]


SOBOL_BITS = 32
SOBOL_MAX_DIM = len(POLY)


@functools.cache
def _directions(dim: int) -> np.ndarray:
    """The direction numbers vₖ = mₖ 2^{32−k} of the first ``dim`` dimensions."""
    directions = np.zeros((dim, SOBOL_BITS), dtype=np.uint64)
    for k in range(SOBOL_BITS):
        directions[0, k] = 1 << (SOBOL_BITS - 1 - k)
    for d in range(1, dim):
        polynomial = POLY[d]
        degree = polynomial.bit_length() - 1
        m = list(M_INIT[d][:degree])
        for k in range(degree, SOBOL_BITS):
            # mₖ = 2ˢ mₖ₋ₛ ⊕ mₖ₋ₛ ⊕ ⨁ⱼ aⱼ 2ʲ mₖ₋ⱼ (Joe and Kuo, 2008)
            new = m[k - degree] ^ (m[k - degree] << degree)
            for j in range(1, degree):
                if (polynomial >> (degree - j)) & 1:
                    new ^= m[k - j] << j
            m.append(new)
        for k in range(SOBOL_BITS):
            directions[d, k] = m[k] << (SOBOL_BITS - 1 - k)
    return directions


def sobol_integers(n: int, dim: int) -> np.ndarray:
    """The first n Sobol points as 32-bit integers, in Gray-code order, (n, D).

    Raises:
        ValidationError: If ``dim`` exceeds the 64 tabulated dimensions.
    """
    if dim > SOBOL_MAX_DIM:
        raise ValidationError(
            f"'dim' must be at most {SOBOL_MAX_DIM} for Sobol (the tabulated "
            f"dimensions), got {dim}; use sequence='halton'"
        )
    directions = _directions(dim)
    index = np.arange(n, dtype=np.uint64)
    # Gray code: consecutive points differ in one direction number
    gray = index ^ (index >> np.uint64(1))
    points = np.zeros((n, dim), dtype=np.uint64)
    for bit in range(SOBOL_BITS):
        selected = ((gray >> np.uint64(bit)) & np.uint64(1)) == 1
        points ^= np.where(selected[:, None], directions[None, :, bit], np.uint64(0))
    return points.astype(np.uint32)


def _primes(count: int) -> list[int]:
    """The first ``count`` primes."""
    primes: list[int] = []
    candidate = 2
    while len(primes) < count:
        if all(candidate % p for p in primes if p * p <= candidate):
            primes.append(candidate)
        candidate += 1
    return primes


def halton_points(n: int, dim: int) -> np.ndarray:
    """The first n Halton points: radical inverses in the first D primes, (n, D)."""
    points = np.zeros((n, dim), dtype=np.float64)
    index = np.arange(n)
    for d, base in enumerate(_primes(dim)):
        # φ_b(i) = ∑ⱼ dⱼ(i) b^{−(j+1)}, the digits of i in base b reflected
        remaining = index.copy()
        scale = 1.0 / base
        while np.any(remaining > 0):
            points[:, d] += (remaining % base) * scale
            remaining //= base
            scale /= base
    return points
