"""Tests for `count_steps` and `step_lengths`, the steps of a span."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

import integrix as ix
from integrix.march import count_steps, step_lengths


# K steps of Δt, Δt to 1 ms in [10⁻³, 10³]
STEPS = st.integers(min_value=1, max_value=10**9)
SIZES = st.integers(min_value=1, max_value=10**6).map(lambda n: n / 1000.0)


@pytest.mark.slow
@settings(max_examples=2_000)
@given(k=STEPS, dt=SIZES)
def test_a_whole_number_of_steps_is_counted_exactly(k: int, dt: float) -> None:
    assert count_steps(k * dt, dt) == k


@pytest.mark.slow
@settings(max_examples=500)
@given(k=st.integers(min_value=1, max_value=10**6), dt=SIZES, part=st.floats(0.0, 1.0))
def test_the_lengths_lie_in_the_step_and_sum_to_the_span(
    k: int, dt: float, part: float
) -> None:
    span = (k - 1 + part) * dt
    lengths = step_lengths(span, dt)
    if span == 0.0:
        assert lengths.shape == (0,)
        return
    assert lengths.shape == (count_steps(span, dt),)
    # a few units of rounding of T
    tolerance = 8.0 * np.spacing(max(span, dt)) * max(1, lengths.size)
    # and the last step takes the slack of a quotient snapped to a whole number
    snapped = 1e-9 * max(1, lengths.size) * dt
    assert (lengths > 0.0).all()
    assert (lengths[:-1] <= dt).all()
    assert lengths[-1] <= dt + snapped + tolerance
    assert abs(float(np.sum(lengths)) - span) <= tolerance


def test_the_rounding_that_adds_a_step_to_a_plain_ceiling() -> None:
    assert math.ceil(2.1 / 0.15) == 15
    assert count_steps(2.1, 0.15) == 14
    assert step_lengths(2.1, 0.15).shape == (14,)


def test_a_partial_last_step() -> None:
    assert count_steps(12.0, 5.0) == 3
    assert step_lengths(12.0, 5.0).tolist() == [5.0, 5.0, 2.0]


def test_an_empty_span_and_a_tiny_one() -> None:
    assert count_steps(0.0, 5.0) == 0
    assert step_lengths(0.0, 5.0).shape == (0,)
    assert count_steps(1e-12, 5.0) == 1
    assert step_lengths(1e-12, 5.0).tolist() == [1e-12]


def test_numpy_numbers_are_taken() -> None:
    assert count_steps(np.float64(2.1), np.float32(0.25)) == 9
    assert step_lengths(np.float64(1.0), 1).dtype == np.float64


@pytest.mark.parametrize(
    ("span", "step", "name"),
    [
        (-1.0, 1.0, "span"),
        (math.inf, 1.0, "span"),
        (math.nan, 1.0, "span"),
        (1.0, 0.0, "step"),
        (1.0, -1.0, "step"),
        (1.0, math.inf, "step"),
    ],
)
def test_bad_numbers_are_refused(span, step, name: str) -> None:
    with pytest.raises(ix.ValidationError, match=f"'{name}'"):
        count_steps(span, step)
    with pytest.raises(ix.ValidationError, match=f"'{name}'"):
        step_lengths(span, step)
