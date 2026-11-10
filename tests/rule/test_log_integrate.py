"""Tests for `integrix.rule.log_integrate` (#69).

log ∑ₙ wₙ fₙ from log fₙ: equal to log(integrate(w, f)) wherever the latter does
not underflow, finite where it does, with masks that drop nodes and keep the
gradient finite, and the same jit/vmap invariance as `integrate`.
"""

from __future__ import annotations

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_traces
from _rules import gauss_legendre_rule, trapezoid_rule
from hypothesis import given, settings
from jaxtyping import TypeCheckError

import integrix as ix
from integrix.rule import integrate, log_integrate


class TestAgainstIntegrate:
    # each list length compiles anew, so the property sweep is slow
    @pytest.mark.slow
    @settings(max_examples=30, deadline=None)
    @given(
        log_values=st.lists(st.floats(-30.0, 30.0), min_size=2, max_size=12),
        seed=st.integers(0, 2**16),
    )
    def test_equals_log_of_integrate(self, log_values: list[float], seed: int) -> None:
        rng = np.random.default_rng(seed)
        weights = jnp.asarray(rng.uniform(0.1, 2.0, len(log_values)))
        log_f = jnp.asarray(log_values)
        expected = jnp.log(integrate(weights, jnp.exp(log_f)).value)
        np.testing.assert_allclose(log_integrate(weights, log_f), expected, rtol=1e-12)

    def test_a_callable_and_a_rule_with_nodes(self) -> None:
        rule = gauss_legendre_rule(8, (1.0, 2.0))
        # log ∫₁² x³ dx = log(15/4)
        value = log_integrate(rule, lambda x: 3.0 * jnp.log(x))
        np.testing.assert_allclose(value, np.log(15.0 / 4.0), rtol=1e-13)

    def test_batched_over_the_other_axes(self) -> None:
        log_f = jnp.log(jnp.arange(1.0, 13.0)).reshape(3, 4)
        weights = jnp.array([0.5, 1.0, 1.0, 0.5])
        values = log_integrate(weights, log_f)
        expected = jnp.log(integrate(weights, jnp.exp(log_f)).value)
        assert values.shape == (3,)
        np.testing.assert_allclose(values, expected, rtol=1e-13)


class TestUnderflow:
    def test_values_below_the_smallest_float_stay_finite(self) -> None:
        rule = trapezoid_rule(jnp.linspace(0.0, 1.0, 11))
        log_f = jnp.full(11, -800.0)
        value = log_integrate(rule, log_f)
        assert bool(jnp.isfinite(value))
        # ∑ wₙ e^(−800) = e^(−800) · 1
        np.testing.assert_allclose(value, -800.0, rtol=1e-14)

    def test_one_large_term_dominates_without_overflow(self) -> None:
        log_f = jnp.array([1000.0, 0.0])
        value = log_integrate(jnp.array([1.0, 1.0]), log_f)
        np.testing.assert_allclose(value, 1000.0, rtol=1e-14)


class TestMask:
    def test_drops_nodes(self) -> None:
        log_f = jnp.log(jnp.array([1.0, 2.0, 4.0]))
        keep = jnp.array([True, False, True])
        value = log_integrate(jnp.ones(3), log_f, where=keep)
        np.testing.assert_allclose(value, np.log(5.0), rtol=1e-14)

    def test_nothing_kept_is_minus_infinity(self) -> None:
        value = log_integrate(jnp.ones(2), jnp.zeros(2), where=jnp.zeros(2, bool))
        assert float(value) == -np.inf

    def test_a_nan_outside_the_mask_does_not_leak(self) -> None:
        log_f = jnp.array([0.0, jnp.nan, 1.0])
        keep = jnp.array([True, False, True])

        def masked(values: jax.Array) -> jax.Array:
            return log_integrate(jnp.ones(3), values, where=keep)

        assert bool(jnp.isfinite(masked(log_f)))
        gradient = jax.grad(masked)(log_f)
        assert bool(jnp.all(jnp.isfinite(gradient)))
        assert float(gradient[1]) == 0.0
        # ∂/∂ log fₙ = wₙ fₙ / I, the softmax of the kept terms
        np.testing.assert_allclose(jnp.sum(gradient), 1.0, rtol=1e-14)


class TestValidation:
    def test_rejects_a_non_positive_weight(self) -> None:
        with pytest.raises(Exception, match="positive weights"):
            log_integrate(jnp.array([1.0, 0.0]), jnp.zeros(2))

    def test_rejects_complex_log_values(self) -> None:
        with pytest.raises(
            (ix.ValidationError, TypeCheckError), match=r"real|log_values"
        ):
            log_integrate(jnp.ones(2), jnp.zeros(2, dtype=jnp.complex128))

    def test_rejects_a_callable_without_nodes(self) -> None:
        with pytest.raises(ix.ValidationError, match="'nodes'"):
            log_integrate(jnp.ones(2), jnp.log)

    def test_rejects_the_wrong_node_count(self) -> None:
        with pytest.raises(ix.ValidationError, match="one entry per node"):
            log_integrate(jnp.ones(3), jnp.zeros(4))


@pytest.mark.slow
class TestTransforms:
    def test_gradient_in_the_log_values_and_the_weights(self) -> None:
        log_f = jnp.array([-1.0, 0.5, 2.0, -0.3])
        weights = jnp.array([0.2, 0.5, 0.9, 0.4])
        assert_grad_matches_fd(lambda v: log_integrate(weights, v), log_f)
        assert_grad_matches_fd(lambda w: log_integrate(w, log_f), weights)

    def test_jit_and_vmap_agree_with_a_loop(self) -> None:
        rng = np.random.default_rng(0)
        weights = jnp.asarray(rng.uniform(0.1, 1.0, (3, 5)))
        log_f = jnp.asarray(rng.normal(size=(3, 5)))
        assert_traces(log_integrate, weights, log_f)
