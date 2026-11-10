"""Tests for `integrix.rule.integrate` and `weighted_mean`.

Gates: G0-exact (a Gauss–Legendre rule integrates the polynomials it must to
1e-13 relative) and G0-scipy (trapezoid sums equal `scipy.integrate.trapezoid`
to 1e-14 on random non-uniform nodes), plus float32 drift over a 10⁵-point
spectrum, gradients in the values and the weights, and jit/vmap invariance.
"""

from __future__ import annotations

import math

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import (
    assert_exact_on_polynomials,
    assert_float32_drift,
    assert_grad_matches_fd,
    assert_matches_scipy,
    assert_traces,
)
from _rules import StandInRule, gauss_legendre_rule, trapezoid_rule
from hypothesis import given, settings
from jaxtyping import TypeCheckError
from scipy.integrate import trapezoid

import integrix as ix
from integrix.rule import Integral, integrate, weighted_mean


def _random_nodes(n_nodes: int, *, seed: int = 0) -> jax.Array:
    """Sorted, non-uniform nodes on [0, 3]."""
    rng = np.random.default_rng(seed)
    return jnp.asarray(np.sort(rng.uniform(0.0, 3.0, n_nodes)))


class TestExactness:
    @pytest.mark.parametrize("n_nodes", [1, 2, 3, 5, 8, 12])
    def test_gauss_legendre_is_exact_to_degree_2n_minus_1(self, n_nodes: int) -> None:
        rule = gauss_legendre_rule(n_nodes, interval=(0.5, 2.0))
        assert_exact_on_polynomials(
            lambda f: integrate(rule, f).value, 2 * n_nodes - 1, interval=(0.5, 2.0)
        )

    def test_values_and_callable_agree(self) -> None:
        rule = gauss_legendre_rule(6)
        from_values = integrate(rule, jnp.cos(rule.nodes)).value
        from_callable = integrate(rule, jnp.cos).value
        assert_matches_scipy(from_values, from_callable, rtol=1e-15)

    def test_trapezoid_is_exact_for_lines(self) -> None:
        nodes = _random_nodes(9)
        rule = trapezoid_rule(nodes)
        lower, upper = float(nodes[0]), float(nodes[-1])
        assert_exact_on_polynomials(
            lambda f: integrate(rule, f).value, 1, interval=(lower, upper)
        )


class TestMatchesScipy:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_trapezoid_on_random_non_uniform_nodes(self, seed: int) -> None:
        nodes = _random_nodes(40, seed=seed)
        values = jnp.sin(nodes) * jnp.exp(nodes)
        actual = integrate(trapezoid_rule(nodes), values).value
        assert_matches_scipy(actual, trapezoid(values, nodes), rtol=1e-14)

    def test_descending_nodes_flip_the_sign(self) -> None:
        nodes = _random_nodes(20)[::-1]
        values = nodes**2
        actual = integrate(trapezoid_rule(nodes), values).value
        assert_matches_scipy(actual, trapezoid(values, nodes), rtol=1e-14)

    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_batched_over_the_other_axes(self, axis: int) -> None:
        nodes = _random_nodes(7)
        shape = [3, 4, 5]
        shape[axis] = 7
        values = jax.random.normal(jax.random.key(0), tuple(shape))
        actual = integrate(trapezoid_rule(nodes), values, axis=axis).value
        reference = trapezoid(np.asarray(values), np.asarray(nodes), axis=axis)
        assert_matches_scipy(actual, reference, rtol=1e-13, atol=1e-14)

    def test_tensor_rule_matches_nested_trapezoids(self) -> None:
        y_nodes = _random_nodes(6, seed=3)
        x_nodes = _random_nodes(8, seed=4)
        field = jax.random.normal(jax.random.key(1), (2, 6, 8))
        rule = (trapezoid_rule(y_nodes), trapezoid_rule(x_nodes))
        actual = integrate(rule, field, axis=(1, 2)).value
        inner = trapezoid(np.asarray(field), np.asarray(x_nodes), axis=2)
        reference = trapezoid(inner, np.asarray(y_nodes), axis=1)
        assert_matches_scipy(actual, reference, rtol=1e-13, atol=1e-14)

    def test_cell_areas_equal_the_tensor_rule(self) -> None:
        y_rule = trapezoid_rule(_random_nodes(5, seed=5))
        x_rule = trapezoid_rule(_random_nodes(4, seed=6))
        areas = jnp.outer(y_rule.weights, x_rule.weights)
        field = jax.random.normal(jax.random.key(2), (5, 4))
        from_areas = integrate(areas, field).value
        from_tensor = integrate((y_rule, x_rule), field).value
        assert_matches_scipy(from_areas, from_tensor, rtol=1e-14)

    def test_axes_need_not_be_adjacent_or_ordered(self) -> None:
        y_rule = trapezoid_rule(_random_nodes(3, seed=7))
        x_rule = trapezoid_rule(_random_nodes(5, seed=8))
        field = jax.random.normal(jax.random.key(3), (5, 2, 3))
        actual = integrate((y_rule, x_rule), field, axis=(2, 0)).value
        reference = integrate((x_rule, y_rule), field, axis=(0, 2)).value
        assert_matches_scipy(actual, reference, rtol=1e-14)


class TestCallable:
    def test_node_axis_comes_first(self) -> None:
        rule = StandInRule(nodes=jnp.array([1.0, 2.0]), weights=jnp.array([0.25, 0.75]))
        total = integrate(rule, lambda x: jnp.outer(x, jnp.array([1.0, 10.0])))
        assert_matches_scipy(total.value, np.array([1.75, 17.5]), rtol=1e-15)

    def test_args_are_passed_through(self) -> None:
        rule = gauss_legendre_rule(10, interval=(0.0, math.pi))
        total = integrate(rule, lambda x, scale: scale * jnp.sin(x), args=3.0)
        assert_matches_scipy(total.value, 6.0, rtol=1e-13)

    def test_tensor_rule_passes_a_tuple_of_nodes(self) -> None:
        rule = (gauss_legendre_rule(4), gauss_legendre_rule(3, interval=(0.0, 2.0)))

        def integrand(nodes: tuple[jax.Array, jax.Array]) -> jax.Array:
            y, x = nodes
            return jnp.outer(y**2, x)

        # ∫₋₁¹ y² dy ∫₀² x dx = (2/3)(2)
        assert_matches_scipy(integrate(rule, integrand).value, 4.0 / 3.0, rtol=1e-13)


class TestResult:
    def test_is_an_integral_with_no_error_estimate(self) -> None:
        rule = gauss_legendre_rule(5)
        result = integrate(rule, jnp.ones((3, 5)))
        assert isinstance(result, Integral)
        assert result.error is None
        assert bool(result.converged)
        assert int(result.n_evaluations) == 5
        assert result.value.shape == (3,)

    def test_counts_every_node_of_a_tensor_rule(self) -> None:
        rule = (gauss_legendre_rule(3), gauss_legendre_rule(4))
        assert int(integrate(rule, jnp.ones((3, 4))).n_evaluations) == 12

    def test_integer_values_integrate_in_floating_point(self) -> None:
        result = integrate(jnp.array([0.5, 0.5]), jnp.array([1, 2]))
        assert jnp.issubdtype(result.value.dtype, jnp.floating)
        assert_matches_scipy(result.value, 1.5, rtol=1e-15)

    def test_integer_values_take_the_weights_dtype(self) -> None:
        weights = jnp.ones(2, dtype=jnp.float32)
        assert integrate(weights, jnp.array([1, 2])).value.dtype == jnp.float32

    def test_complex_weights_keep_their_imaginary_part(self) -> None:
        weights = jnp.array([1.0 + 1.0j, 1.0 + 1.0j])
        total = integrate(weights, jnp.array([1.0, 2.0])).value
        assert_matches_scipy(total, 3.0 + 3.0j, rtol=1e-15)

    def test_complex_weights_keep_float32_precision(self) -> None:
        weights = jnp.array([1.0j, 1.0j])
        values = jnp.ones(2, dtype=jnp.float32)
        assert integrate(weights, values).value.dtype == jnp.complex64

    def test_accepts_a_numpy_integer_axis(self) -> None:
        values = jnp.arange(6.0).reshape(2, 3)
        by_numpy = integrate(jnp.ones(2), values, axis=np.int64(0)).value
        assert_matches_scipy(
            by_numpy, integrate(jnp.ones(2), values, axis=0).value, rtol=0
        )

    def test_is_reexported_from_the_rule_module(self) -> None:
        assert ix.rule.integrate is integrate


class TestPrecision:
    def test_float32_values_keep_float32_with_float64_weights(self) -> None:
        rule = gauss_legendre_rule(4)
        values = jnp.ones(4, dtype=jnp.float32)
        assert integrate(rule, values).value.dtype == jnp.float32

    def test_float32_drift_over_a_long_spectrum(self) -> None:
        wavelength = jnp.linspace(1000.0, 2500.0, 100_000)
        rule = trapezoid_rule(wavelength)
        radiance = 1.0 + 0.1 * jnp.sin(wavelength / 7.0)

        @jax.jit
        def band_total(values: jax.Array) -> jax.Array:
            return integrate(rule, values).value

        assert_float32_drift(band_total, radiance, rtol=1e-5)


class TestTransforms:
    def test_gradient_in_the_values(self) -> None:
        rule = trapezoid_rule(_random_nodes(6))
        values = jnp.cos(rule.nodes)
        assert_grad_matches_fd(lambda v: integrate(rule, v).value, values)

    def test_gradient_in_the_weights(self) -> None:
        nodes = _random_nodes(6)
        values = jnp.cos(nodes)
        assert_grad_matches_fd(lambda w: integrate(w, values).value, jnp.ones(6))

    def test_jit_and_vmap_agree_with_a_loop(self) -> None:
        nodes = jnp.stack([_random_nodes(8, seed=seed) for seed in range(3)])
        values = jnp.sin(nodes)

        def total(node_row: jax.Array, value_row: jax.Array) -> jax.Array:
            return integrate(trapezoid_rule(node_row), value_row).value

        assert_traces(total, nodes, values)


class TestProperties:
    @pytest.mark.slow
    @settings(max_examples=15, deadline=None)
    @given(
        n_nodes=st.integers(1, 12),
        a=st.floats(-3.0, 3.0),
        b=st.floats(-3.0, 3.0),
        seed=st.integers(0, 2**16),
    )
    def test_linear_in_the_values(
        self, n_nodes: int, a: float, b: float, seed: int
    ) -> None:
        key_f, key_g, key_w = jax.random.split(jax.random.key(seed), 3)
        f = jax.random.normal(key_f, (n_nodes,))
        g = jax.random.normal(key_g, (n_nodes,))
        weights = jax.random.uniform(key_w, (n_nodes,))
        combined = integrate(weights, a * f + b * g).value
        separate = a * integrate(weights, f).value + b * integrate(weights, g).value
        assert_matches_scipy(combined, separate, rtol=1e-12, atol=1e-12)

    @pytest.mark.slow
    @settings(max_examples=15, deadline=None)
    @given(n_nodes=st.integers(1, 40), seed=st.integers(0, 2**16))
    def test_ones_integrate_to_the_sum_of_the_weights(
        self, n_nodes: int, seed: int
    ) -> None:
        weights = jax.random.uniform(jax.random.key(seed), (n_nodes,))
        total = integrate(weights, jnp.ones(n_nodes)).value
        assert_matches_scipy(total, float(np.sum(np.asarray(weights))), rtol=1e-13)


class TestWeightedMean:
    def test_band_radiance_matches_scipy(self) -> None:
        wavelength = _random_nodes(50)
        response = jnp.exp(-((wavelength - 1.5) ** 2))
        radiance = jax.random.uniform(jax.random.key(4), (3, 50))
        band = weighted_mean(trapezoid_rule(wavelength), radiance, weight=response)
        reference = trapezoid(
            np.asarray(radiance * response), np.asarray(wavelength)
        ) / trapezoid(np.asarray(response), np.asarray(wavelength))
        assert_matches_scipy(band, reference, rtol=1e-13)

    def test_per_sample_weight_equals_per_node_weight(self) -> None:
        rule = trapezoid_rule(_random_nodes(9))
        radiance = jax.random.uniform(jax.random.key(5), (4, 9))
        response = jnp.linspace(0.1, 1.0, 9)
        per_node = weighted_mean(rule, radiance, weight=response)
        per_sample = weighted_mean(
            rule, radiance, weight=jnp.broadcast_to(response, (4, 9))
        )
        assert_matches_scipy(per_sample, per_node, rtol=1e-14)

    def test_per_sample_weight_differs_row_by_row(self) -> None:
        wavelength = _random_nodes(9)
        key_l, key_r = jax.random.split(jax.random.key(6))
        radiance = jax.random.uniform(key_l, (4, 9))
        responses = jax.random.uniform(key_r, (4, 9), minval=0.1)
        band = weighted_mean(trapezoid_rule(wavelength), radiance, weight=responses)
        reference = trapezoid(
            np.asarray(radiance * responses), np.asarray(wavelength)
        ) / trapezoid(np.asarray(responses), np.asarray(wavelength))
        assert_matches_scipy(band, reference, rtol=1e-13)

    def test_a_square_weight_is_read_like_the_values(self) -> None:
        # The weight has both the values' shape and the rule's, with the axes
        # permuted: it is read per sample, in the values' layout.
        y_rule = trapezoid_rule(_random_nodes(3, seed=9))
        x_rule = trapezoid_rule(_random_nodes(3, seed=10))
        key_l, key_r = jax.random.split(jax.random.key(7))
        values = jax.random.uniform(key_l, (3, 3))
        response = jax.random.uniform(key_r, (3, 3), minval=0.1)
        band = weighted_mean((y_rule, x_rule), values, weight=response, axis=(1, 0))
        # value axis 1 runs over y and axis 0 over x: w[i, j] = w_x[i] w_y[j]
        areas = jnp.outer(x_rule.weights, y_rule.weights)
        reference = jnp.sum(areas * response * values) / jnp.sum(areas * response)
        assert_matches_scipy(band, reference, rtol=1e-14)

    def test_mean_of_a_constant_is_the_constant(self) -> None:
        rule = gauss_legendre_rule(7)
        band = weighted_mean(rule, jnp.full(7, 4.2), weight=jnp.cos(rule.nodes))
        assert_matches_scipy(band, 4.2, rtol=1e-14)

    def test_gradient_in_the_response(self) -> None:
        rule = trapezoid_rule(_random_nodes(5))
        radiance = jnp.sin(rule.nodes)
        assert_grad_matches_fd(
            lambda r: weighted_mean(rule, radiance, weight=r), jnp.linspace(0.5, 1, 5)
        )

    def test_rejects_a_weight_of_the_wrong_shape(self) -> None:
        with pytest.raises(ix.ValidationError, match="'weight' must have"):
            weighted_mean(jnp.ones(3), jnp.ones((2, 3)), weight=jnp.ones(2))


def _cloudy_scene(seed: int = 0) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Cell areas, an enhancement with NaN (cloudy) pixels, and a plume mask."""
    rng = np.random.default_rng(seed)
    areas = jnp.asarray(rng.uniform(0.5, 1.5, (6, 7)))
    enhancement = rng.normal(size=(6, 7))
    enhancement[rng.uniform(size=(6, 7)) < 0.2] = np.nan
    plume = jnp.asarray(rng.uniform(size=(6, 7)) < 0.6)
    return areas, jnp.asarray(enhancement), plume


class TestMaskAndNan:
    """Gates G6-mask and G6-grad: ``where=`` and ``nan_policy=`` (#48)."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_omit_with_a_mask_matches_nansum(self, seed: int) -> None:
        areas, enhancement, plume = _cloudy_scene(seed)
        total = integrate(
            areas, enhancement, axis=(0, 1), where=plume, nan_policy="omit"
        ).value
        expected = np.nansum(np.asarray(enhancement)[plume] * np.asarray(areas)[plume])
        np.testing.assert_allclose(total, expected, rtol=1e-14)

    def test_omit_without_a_mask_matches_nansum(self) -> None:
        areas, enhancement, _ = _cloudy_scene()
        total = integrate(areas, enhancement, axis=(0, 1), nan_policy="omit").value
        expected = np.nansum(np.asarray(enhancement) * np.asarray(areas))
        np.testing.assert_allclose(total, expected, rtol=1e-14)

    def test_propagate_gives_nan_for_a_nan_inside_the_mask(self) -> None:
        values = jnp.array([1.0, jnp.nan, 3.0])
        total = integrate(jnp.ones(3), values, where=jnp.array([True, True, False]))
        assert bool(jnp.isnan(total.value))

    def test_propagate_ignores_a_nan_outside_the_mask(self) -> None:
        values = jnp.array([1.0, jnp.nan, 3.0])
        total = integrate(jnp.ones(3), values, where=jnp.array([True, False, True]))
        assert float(total.value) == 4.0

    def test_default_propagates_as_before(self) -> None:
        total = integrate(jnp.ones(3), jnp.array([1.0, jnp.nan, 3.0]))
        assert bool(jnp.isnan(total.value))

    def test_a_mask_over_a_batch_axis_drops_whole_columns(self) -> None:
        # rows are columns of the atmosphere, the last axis their levels
        density = jnp.arange(12.0).reshape(3, 4)
        valid = jnp.array([[True], [False], [True]])
        total = integrate(jnp.ones(4), density, where=valid).value
        np.testing.assert_allclose(total, [6.0, 0.0, 38.0])

    def test_a_mask_over_the_nodes_broadcasts_over_the_batch(self) -> None:
        values = jnp.ones((2, 4))
        total = integrate(jnp.ones(4), values, where=jnp.array([1, 0, 1, 1]) > 0)
        np.testing.assert_allclose(total.value, [3.0, 3.0])

    def test_a_callable_integrand_takes_a_mask(self) -> None:
        rule = trapezoid_rule(jnp.linspace(0.0, 1.0, 5))
        keep = rule.nodes < 0.6
        total = integrate(rule, jnp.exp, where=keep).value
        expected = jnp.sum(jnp.where(keep, rule.weights * jnp.exp(rule.nodes), 0.0))
        np.testing.assert_allclose(total, expected, rtol=1e-14)

    @pytest.mark.slow
    def test_gradient_in_the_kept_values_matches_finite_differences(self) -> None:
        areas, enhancement, plume = _cloudy_scene()
        kept = plume & ~jnp.isnan(enhancement)
        filled = jnp.where(jnp.isnan(enhancement), 0.0, enhancement)

        def mass(kept_values: jax.Array) -> jax.Array:
            field = filled.at[kept].set(kept_values)
            return integrate(
                areas, field, axis=(0, 1), where=plume, nan_policy="omit"
            ).value

        assert_grad_matches_fd(mass, filled[kept])

    def test_dropped_samples_get_an_exactly_zero_finite_gradient(self) -> None:
        areas, enhancement, plume = _cloudy_scene()

        def mass(field: jax.Array) -> jax.Array:
            return integrate(
                areas, field, axis=(0, 1), where=plume, nan_policy="omit"
            ).value

        gradient = jax.grad(mass)(enhancement)
        dropped = ~plume | jnp.isnan(enhancement)
        assert bool(jnp.all(jnp.isfinite(gradient)))
        assert bool(jnp.all(gradient[dropped] == 0.0))
        np.testing.assert_allclose(gradient[~dropped], areas[~dropped], rtol=0)

    @pytest.mark.slow
    def test_gradient_in_the_weights_matches_finite_differences(self) -> None:
        areas, enhancement, plume = _cloudy_scene()

        def mass(weights: jax.Array) -> jax.Array:
            return integrate(
                weights, enhancement, axis=(0, 1), where=plume, nan_policy="omit"
            ).value

        assert_grad_matches_fd(mass, areas)
        gradient = jax.grad(mass)(areas)
        assert bool(jnp.all(jnp.isfinite(gradient)))

    @pytest.mark.slow
    def test_jit_and_vmap_agree_with_a_loop(self) -> None:
        scenes = [_cloudy_scene(seed) for seed in range(3)]
        stacked = (jnp.stack(parts) for parts in zip(*scenes, strict=True))
        areas, enhancement, plume = stacked

        def mass(a: jax.Array, f: jax.Array, m: jax.Array) -> jax.Array:
            return integrate(a, f, axis=(0, 1), where=m, nan_policy="omit").value

        assert_traces(mass, areas, enhancement, plume)

    def test_float32_values_stay_float32(self) -> None:
        values = jnp.array([1.0, jnp.nan, 3.0], dtype=jnp.float32)
        total = integrate(jnp.ones(3), values, nan_policy="omit").value
        assert total.dtype == jnp.float32
        assert float(total) == 4.0

    def test_weighted_mean_renormalises_over_the_kept_samples(self) -> None:
        rng = np.random.default_rng(3)
        weights = jnp.asarray(rng.uniform(0.5, 1.5, 8))
        response = jnp.asarray(rng.uniform(0.0, 1.0, 8))
        radiance = rng.normal(size=(4, 8))
        radiance[rng.uniform(size=(4, 8)) < 0.25] = np.nan
        valid = jnp.asarray(rng.uniform(size=(4, 8)) < 0.8)
        band = weighted_mean(
            weights,
            jnp.asarray(radiance),
            weight=response,
            where=valid,
            nan_policy="omit",
        )
        keep = np.asarray(valid) & ~np.isnan(radiance)
        combined = np.asarray(weights * response)
        expected = [
            np.sum(combined[k] * row[k]) / np.sum(combined[k])
            for row, k in zip(radiance, keep, strict=True)
        ]
        np.testing.assert_allclose(band, expected, rtol=1e-13)

    def test_weighted_mean_per_sample_weight_with_a_mask(self) -> None:
        weights = jnp.ones(3)
        values = jnp.array([[1.0, 2.0, 30.0]])
        response = jnp.array([[1.0, 1.0, 1.0]])
        band = weighted_mean(
            weights, values, weight=response, where=jnp.array([True, True, False])
        )
        np.testing.assert_allclose(band, [1.5])

    def test_weighted_mean_of_nothing_kept_is_nan(self) -> None:
        band = weighted_mean(
            jnp.ones(3), jnp.ones(3), weight=jnp.ones(3), where=jnp.zeros(3, bool)
        )
        assert bool(jnp.isnan(band))

    @pytest.mark.slow
    def test_weighted_mean_gradient_in_the_response_is_finite(self) -> None:
        values = jnp.array([1.0, jnp.nan, 3.0, 4.0])

        def band(response: jax.Array) -> jax.Array:
            return weighted_mean(
                jnp.ones(4), values, weight=response, nan_policy="omit"
            )

        response = jnp.array([0.5, 1.0, 1.5, 2.0])
        assert_grad_matches_fd(band, response)
        assert float(jax.grad(band)(response)[1]) == 0.0

    def test_rejects_a_mask_that_is_not_boolean(self) -> None:
        with pytest.raises(ix.ValidationError, match="'where' must be boolean"):
            integrate(jnp.ones(3), jnp.ones(3), where=jnp.ones(3))

    def test_rejects_a_mask_that_does_not_broadcast(self) -> None:
        with pytest.raises(ix.ValidationError, match="'where' must broadcast"):
            integrate(jnp.ones(3), jnp.ones(3), where=jnp.ones(4, bool))

    def test_rejects_a_mask_larger_than_the_values(self) -> None:
        with pytest.raises(ix.ValidationError, match="'where' must broadcast"):
            integrate(jnp.ones(3), jnp.ones(3), where=jnp.ones((2, 3), bool))

    def test_rejects_an_unknown_nan_policy(self) -> None:
        # the import hook in conftest.py checks the Literal before the body runs
        with pytest.raises((ix.ValidationError, TypeCheckError), match="nan_policy"):
            weighted_mean(
                jnp.ones(3), jnp.ones(3), weight=jnp.ones(3), nan_policy="raise"
            )


class TestValidation:
    def test_rejects_an_empty_tensor_rule(self) -> None:
        with pytest.raises(ix.ValidationError, match="at least one rule"):
            integrate((), jnp.ones(3))

    def test_rejects_a_tensor_rule_of_arrays(self) -> None:
        # In the suite, jaxtyping's import hook rejects the annotation first;
        # without it, the boundary check raises ValidationError.
        with pytest.raises((ix.ValidationError, TypeCheckError)):
            integrate((jnp.ones(3),), jnp.ones(3))

    def test_rejects_a_rule_that_is_not_an_array(self) -> None:
        with pytest.raises((ix.ValidationError, TypeCheckError)):
            integrate(object(), jnp.ones(3))

    def test_rejects_a_tensor_rule_that_is_not_1d(self) -> None:
        flat = StandInRule(nodes=jnp.ones((2, 2)), weights=jnp.ones((2, 2)))
        with pytest.raises(ix.ValidationError, match="1-D rules"):
            integrate((flat,), jnp.ones((2, 2)))

    @pytest.mark.parametrize("weights", [jnp.array(1.0), jnp.ones(0)])
    def test_rejects_weights_without_a_node(self, weights: jax.Array) -> None:
        with pytest.raises(ix.ValidationError, match="at least one weight"):
            integrate(weights, jnp.ones(3))

    def test_rejects_a_callable_without_nodes(self) -> None:
        with pytest.raises(ix.ValidationError, match="'nodes'"):
            integrate(jnp.ones(3), jnp.sin)

    def test_rejects_a_callable_with_an_axis(self) -> None:
        with pytest.raises(ix.ValidationError, match="'axis' must be None"):
            integrate(gauss_legendre_rule(3), jnp.sin, axis=0)

    def test_rejects_the_wrong_number_of_axes(self) -> None:
        with pytest.raises(ix.ValidationError, match="one axis per weights axis"):
            integrate(jnp.ones((2, 3)), jnp.ones((2, 3)), axis=0)

    def test_rejects_an_axis_out_of_range(self) -> None:
        with pytest.raises(ix.ValidationError, match="within the values' 2 axes"):
            integrate(jnp.ones(3), jnp.ones((2, 3)), axis=2)

    def test_rejects_an_axis_that_is_not_an_integer(self) -> None:
        with pytest.raises((ix.ValidationError, TypeCheckError)):
            integrate(jnp.ones(3), jnp.ones(3), axis=1.5)

    def test_rejects_a_repeated_axis(self) -> None:
        with pytest.raises(ix.ValidationError, match="must not repeat"):
            integrate(jnp.ones((3, 3)), jnp.ones((3, 3)), axis=(0, -2))

    def test_rejects_values_with_the_wrong_node_count(self) -> None:
        with pytest.raises(ix.ValidationError, match="one entry per node"):
            integrate(jnp.ones(4), jnp.ones((2, 3)))
