"""Tests for `integrix.expect.effective_sample_size` and
`pareto_smoothed_log_weights` (#70).

Oracles: equal weights give ESS = S and k̂ = NaN; a Pareto tail of known shape is
recovered within the estimator's bias; and on fixed seeds at several S the JAX
smoothing equals plumax's NumPy version (``tests/_psis_reference.py``, from plumax
feb5e86) to 1e-10, NaN cases included. Smoothed weights are normalised and keep
the order of the raw weights; jit and vmap over a batch agree with a loop.
"""

from __future__ import annotations

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_traces
from _psis_reference import pareto_smoothed_log_weights as reference_smoothing
from hypothesis import given, settings
from jaxtyping import TypeCheckError

import integrix as ix
from integrix.expect import (
    effective_sample_size,
    importance,
    pareto_smoothed_log_weights,
)


class TestEffectiveSampleSize:
    def test_equal_weights_give_the_draw_count(self) -> None:
        np.testing.assert_allclose(effective_sample_size(jnp.zeros(50)), 50.0)

    def test_matches_the_linear_formula(self) -> None:
        weights = np.random.default_rng(0).uniform(0.1, 3.0, 40)
        expected = weights.sum() ** 2 / np.sum(weights**2)
        value = effective_sample_size(jnp.log(jnp.asarray(weights)))
        np.testing.assert_allclose(value, expected, rtol=1e-12)

    def test_is_invariant_to_scaling_the_weights(self) -> None:
        log_weights = jnp.asarray(np.random.default_rng(1).normal(size=30))
        np.testing.assert_allclose(
            effective_sample_size(log_weights + 700.0),
            effective_sample_size(log_weights),
            rtol=1e-12,
        )

    def test_batches_over_leading_axes(self) -> None:
        log_weights = jnp.asarray(np.random.default_rng(2).normal(size=(3, 2, 20)))
        assert effective_sample_size(log_weights).shape == (3, 2)


def _log_weights(kind: str, size: int, seed: int) -> np.ndarray:
    """Fixed-seed log weights: light-tailed, heavy-tailed, or equal."""
    rng = np.random.default_rng(seed)
    if kind == "light":
        return rng.normal(0.0, 0.3, size)
    if kind == "heavy":
        # log of |Student-t(2)| weights: a tail with k ≈ 0.5
        return np.log(np.abs(rng.standard_t(2.0, size)) + 1e-3)
    return np.zeros(size)


class TestAgainstPlumax:
    def test_one_light_tailed_case(self) -> None:
        log_weights = _log_weights("light", 1000, 0)
        smoothed, k_hat = pareto_smoothed_log_weights(jnp.asarray(log_weights))
        expected_smoothed, expected_k = reference_smoothing(log_weights)
        np.testing.assert_allclose(k_hat, expected_k, rtol=1e-10)
        np.testing.assert_allclose(smoothed, expected_smoothed, rtol=1e-10, atol=1e-12)

    # every shape compiles anew, so the sweep is slow
    @pytest.mark.slow
    @pytest.mark.parametrize("size", [10, 100, 1000, 4000])
    @pytest.mark.parametrize("kind", ["light", "heavy", "equal"])
    @pytest.mark.parametrize("seed", [0, 1])
    def test_matches_the_numpy_reference(self, size: int, kind: str, seed: int) -> None:
        log_weights = _log_weights(kind, size, seed)
        smoothed, k_hat = pareto_smoothed_log_weights(jnp.asarray(log_weights))
        expected_smoothed, expected_k = reference_smoothing(log_weights)
        if np.isnan(expected_k):
            assert bool(jnp.isnan(k_hat))
        else:
            np.testing.assert_allclose(k_hat, expected_k, rtol=1e-10)
        np.testing.assert_allclose(smoothed, expected_smoothed, rtol=1e-10, atol=1e-12)

    def test_a_single_weight_has_no_tail(self) -> None:
        smoothed, k_hat = pareto_smoothed_log_weights(jnp.array([3.0]))
        assert bool(jnp.isnan(k_hat))
        np.testing.assert_allclose(smoothed, [0.0])


class TestProperties:
    def test_recovers_a_known_pareto_shape(self) -> None:
        # w ~ GPD(σ = 1, ξ = 0.5) by inversion: w = ((1 − u)^(−ξ) − 1)/ξ
        uniforms = np.random.default_rng(3).uniform(size=4000)
        weights = ((1.0 - uniforms) ** -0.5 - 1.0) / 0.5 + 1e-9
        _, k_hat = pareto_smoothed_log_weights(jnp.log(jnp.asarray(weights)))
        # the tail of 190 exceedances, shrunk towards ½: well within ±0.15
        assert abs(float(k_hat) - 0.5) < 0.15

    @pytest.mark.slow
    @settings(max_examples=20, deadline=None)
    @given(seed=st.integers(0, 2**16), spread=st.floats(0.1, 3.0))
    def test_normalised_and_order_preserving(self, seed: int, spread: float) -> None:
        log_weights = spread * np.random.default_rng(seed).standard_t(3.0, 200)
        smoothed, _ = pareto_smoothed_log_weights(jnp.asarray(log_weights))
        np.testing.assert_allclose(jax.nn.logsumexp(smoothed), 0.0, atol=1e-12)
        order = np.argsort(log_weights, kind="stable")
        assert bool(jnp.all(jnp.diff(smoothed[order]) >= -1e-12))

    @pytest.mark.slow
    def test_batches_over_leading_axes(self) -> None:
        log_weights = jnp.asarray(np.random.default_rng(4).normal(size=(3, 500)))
        smoothed, k_hat = pareto_smoothed_log_weights(log_weights)
        assert smoothed.shape == (3, 500)
        assert k_hat.shape == (3,)
        for row, (row_smoothed, row_k) in enumerate(zip(smoothed, k_hat, strict=True)):
            expected, expected_k = reference_smoothing(np.asarray(log_weights[row]))
            np.testing.assert_allclose(row_smoothed, expected, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(row_k, expected_k, rtol=1e-10)

    def test_rejects_a_scalar(self) -> None:
        # the import hook checks the annotation before the body runs
        with pytest.raises((ix.ValidationError, TypeCheckError)):
            pareto_smoothed_log_weights(jnp.asarray(1.0))


class TestImportanceDiagnostics:
    @pytest.mark.slow
    def test_importance_reports_pareto_k_on_request(self) -> None:
        def proposal(key: jax.Array, n: int) -> jax.Array:
            return 2.0 * jax.random.normal(key, (n,))

        estimate = importance(
            jnp.square,
            proposal,
            lambda x: jax.scipy.stats.norm.logpdf(x, scale=2.0),
            jax.scipy.stats.norm.logpdf,
            2000,
            diagnostics=True,
            key=jax.random.key(0),
        )
        assert estimate.pareto_k is not None
        # a wider proposal than the target: light-tailed weights
        assert float(estimate.pareto_k) < 0.7
        plain = importance(
            jnp.square,
            proposal,
            lambda x: jax.scipy.stats.norm.logpdf(x, scale=2.0),
            jax.scipy.stats.norm.logpdf,
            2000,
            key=jax.random.key(0),
        )
        assert plain.pareto_k is None
        np.testing.assert_allclose(estimate.value, plain.value, rtol=1e-14)
        np.testing.assert_allclose(
            estimate.effective_sample_size, plain.effective_sample_size, rtol=1e-12
        )


@pytest.mark.slow
def test_jit_and_vmap_agree_with_a_loop() -> None:
    log_weights = jnp.asarray(np.random.default_rng(5).normal(size=(3, 300)))
    assert_traces(effective_sample_size, log_weights)
    assert_traces(lambda w: pareto_smoothed_log_weights(w)[0], log_weights)
    assert_traces(lambda w: pareto_smoothed_log_weights(w)[1], log_weights)
