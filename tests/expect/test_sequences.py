"""The low-discrepancy point sets against SciPy's, bit for bit (#27)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import qmc as scipy_qmc

import integrix as ix
from integrix._src.expect._sequences import (
    SOBOL_BITS,
    SOBOL_MAX_DIM,
    halton_points,
    sobol_integers,
)


def test_sobol_matches_scipy_in_all_tabulated_dimensions() -> None:
    reference = scipy_qmc.Sobol(d=SOBOL_MAX_DIM, scramble=False, bits=32).random(1024)
    ours = sobol_integers(1024, SOBOL_MAX_DIM).astype(np.float64) / 2.0**SOBOL_BITS
    assert np.array_equal(ours, reference)


@pytest.mark.parametrize("dim", [1, 5, 13])
def test_halton_matches_scipy(dim: int) -> None:
    reference = scipy_qmc.Halton(d=dim, scramble=False).random(500)
    assert np.allclose(halton_points(500, dim), reference, rtol=0.0, atol=1e-15)


def test_sobol_refuses_untabulated_dimensions() -> None:
    with pytest.raises(ix.ValidationError, match="halton"):
        sobol_integers(8, SOBOL_MAX_DIM + 1)
