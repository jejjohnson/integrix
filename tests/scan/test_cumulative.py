"""Tests for `integrix.scan.cumulative`.

The gate is equality with `scipy.integrate.cumulative_trapezoid` (with
``initial=0``) to 1e-14 on random non-uniform nodes, forward and reversed,
and the last entry equal to `integrix.rule.integrate` on the same rule to
1e-14; plus jit, vmap and grad, and float32 drift down a long column.
"""

from __future__ import annotations

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import (
    assert_float32_drift,
    assert_grad_matches_fd,
    assert_matches_scipy,
    assert_traces,
)
from _rules import StandInRule, trapezoid_rule
from hypothesis import given, settings
from jaxtyping import TypeCheckError
from scipy.integrate import cumulative_trapezoid

import integrix as ix
from integrix.rule import integrate
from integrix.scan import cumulative


def _random_nodes(n_nodes: int, *, seed: int = 0) -> jax.Array:
    """Sorted, non-uniform nodes on [0, 3]."""
    rng = np.random.default_rng(seed)
    return jnp.asarray(np.sort(rng.uniform(0.0, 3.0, n_nodes)))


def _forward_reference(values: jax.Array, nodes: jax.Array, axis: int = -1):
    """SciPy's running trapezoid integral from the first node."""
    return cumulative_trapezoid(
        np.asarray(values), np.asarray(nodes), axis=axis, initial=0.0
    )


def _reverse_reference(values: jax.Array, nodes: jax.Array, axis: int = -1):
    """∫ from xₖ to x_{N−1}: the total less SciPy's running integral."""
    forward = _forward_reference(values, nodes, axis)
    total = np.take(forward, [-1], axis=axis)
    return total - forward


def _scale(reference: np.ndarray) -> float:
    """An absolute floor of 1e-14 of the largest entry, for entries near F₀ = 0."""
    return 1e-14 * float(np.max(np.abs(reference)))


class TestMatchesScipy:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_forward(self, seed: int) -> None:
        nodes = _random_nodes(50, seed=seed)
        values = jnp.exp(nodes) * jnp.cos(3.0 * nodes)
        reference = _forward_reference(values, nodes)
        actual = cumulative(trapezoid_rule(nodes), values)
        assert_matches_scipy(actual, reference, rtol=1e-14, atol=_scale(reference))

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_reversed(self, seed: int) -> None:
        nodes = _random_nodes(50, seed=seed)
        values = jnp.exp(nodes) * jnp.cos(3.0 * nodes)
        reference = _reverse_reference(values, nodes)
        actual = cumulative(trapezoid_rule(nodes), values, reverse=True)
        assert_matches_scipy(actual, reference, rtol=1e-14, atol=_scale(reference))

    def test_descending_nodes(self) -> None:
        nodes = _random_nodes(30)[::-1]
        values = nodes**2
        reference = _forward_reference(values, nodes)
        actual = cumulative(trapezoid_rule(nodes), values)
        assert_matches_scipy(actual, reference, rtol=1e-14, atol=_scale(reference))

    @pytest.mark.parametrize("axis", [0, 1, -1])
    @pytest.mark.parametrize("reverse", [False, True])
    def test_batched_over_the_other_axes(self, axis: int, reverse: bool) -> None:
        nodes = _random_nodes(9)
        shape = [3, 4, 5]
        shape[axis] = 9
        values = jax.random.normal(jax.random.key(0), tuple(shape))
        reference = (_reverse_reference if reverse else _forward_reference)(
            values, nodes, axis
        )
        actual = cumulative(trapezoid_rule(nodes), values, axis=axis, reverse=reverse)
        assert_matches_scipy(actual, reference, rtol=1e-13, atol=_scale(reference))


class TestAgreesWithTheRule:
    def test_last_entry_is_the_integral(self) -> None:
        nodes = _random_nodes(40, seed=5)
        rule = trapezoid_rule(nodes)
        values = jnp.sin(nodes) + 2.0
        total = integrate(rule, values).value
        assert_matches_scipy(cumulative(rule, values)[-1], total, rtol=1e-14)

    def test_reversed_first_entry_is_the_integral(self) -> None:
        nodes = _random_nodes(40, seed=6)
        rule = trapezoid_rule(nodes)
        values = jnp.sin(nodes) + 2.0
        total = integrate(rule, values).value
        from_top = cumulative(rule, values, reverse=True)
        assert_matches_scipy(from_top[0], total, rtol=1e-14)


class TestEdges:
    def test_initial_offsets_every_entry(self) -> None:
        rule = trapezoid_rule(_random_nodes(6))
        values = jnp.cos(rule.nodes)
        shifted = cumulative(rule, values, initial=2.5)
        assert_matches_scipy(shifted, cumulative(rule, values) + 2.5, rtol=1e-15)

    @pytest.mark.parametrize("reverse", [False, True])
    def test_a_single_node_gives_the_initial_value(self, reverse: bool) -> None:
        rule = StandInRule(nodes=jnp.array([1.0]), weights=jnp.array([0.0]))
        result = cumulative(
            rule, jnp.array([[4.0], [5.0]]), reverse=reverse, initial=1.0
        )
        assert_matches_scipy(result, np.ones((2, 1)), rtol=0.0)

    def test_integer_values_integrate_in_floating_point(self) -> None:
        rule = trapezoid_rule(jnp.array([0.0, 1.0, 2.0]))
        result = cumulative(rule, jnp.array([1, 1, 1]))
        assert jnp.issubdtype(result.dtype, jnp.floating)
        assert_matches_scipy(result, np.array([0.0, 1.0, 2.0]), rtol=1e-15)

    def test_complex_nodes_keep_their_imaginary_part(self) -> None:
        nodes = jnp.array([0.0, 1.0 + 1.0j, 3.0])
        values = jnp.array([1.0, 2.0, 3.0])
        reference = _forward_reference(values, nodes)
        actual = cumulative(StandInRule(nodes=nodes, weights=nodes), values)
        assert_matches_scipy(actual, reference, rtol=1e-15)

    @pytest.mark.parametrize("n_nodes", [1, 3])
    def test_dtype_does_not_depend_on_the_node_count(self, n_nodes: int) -> None:
        rule = trapezoid_rule(jnp.arange(float(n_nodes)))
        values = jnp.ones(n_nodes, dtype=jnp.float32)
        result = cumulative(rule, values, initial=float(jnp.float64(1.0)))
        assert result.dtype == jnp.float32

    def test_is_reexported_from_the_scan_module(self) -> None:
        assert ix.scan.cumulative is cumulative


class TestPrecision:
    def test_float32_values_keep_float32_with_float64_nodes(self) -> None:
        rule = trapezoid_rule(_random_nodes(5))
        values = jnp.ones(5, dtype=jnp.float32)
        assert cumulative(rule, values).dtype == jnp.float32

    @pytest.mark.parametrize("reverse", [False, True])
    def test_float32_drift_down_a_long_column(self, reverse: bool) -> None:
        altitude = jnp.linspace(0.0, 80.0, 100_000)
        rule = trapezoid_rule(altitude)
        absorption = jnp.exp(-altitude / 8.0)

        @jax.jit
        def optical_depth(values: jax.Array) -> jax.Array:
            return cumulative(rule, values, reverse=reverse)

        assert_float32_drift(optical_depth, absorption, rtol=1e-5)


class TestTransforms:
    @pytest.mark.parametrize("reverse", [False, True])
    def test_gradient_in_the_values(self, reverse: bool) -> None:
        rule = trapezoid_rule(_random_nodes(6))
        values = jnp.cos(rule.nodes)
        assert_grad_matches_fd(lambda v: cumulative(rule, v, reverse=reverse), values)

    def test_gradient_in_the_nodes(self) -> None:
        nodes = _random_nodes(6)
        values = jnp.cos(nodes)
        assert_grad_matches_fd(
            lambda x: cumulative(StandInRule(nodes=x, weights=x), values), nodes
        )

    @pytest.mark.parametrize("reverse", [False, True])
    def test_jit_and_vmap_agree_with_a_loop(self, reverse: bool) -> None:
        nodes = jnp.stack([_random_nodes(8, seed=seed) for seed in range(3)])
        values = jnp.sin(nodes)

        def running(node_row: jax.Array, value_row: jax.Array) -> jax.Array:
            return cumulative(trapezoid_rule(node_row), value_row, reverse=reverse)

        assert_traces(running, nodes, values)


class TestProperties:
    @pytest.mark.slow
    @settings(max_examples=15, deadline=None)
    @given(n_nodes=st.integers(2, 30), seed=st.integers(0, 2**16))
    def test_non_decreasing_for_a_positive_integrand(
        self, n_nodes: int, seed: int
    ) -> None:
        key_x, key_f = jax.random.split(jax.random.key(seed))
        nodes = jnp.sort(jax.random.uniform(key_x, (n_nodes,)))
        values = jax.random.uniform(key_f, (n_nodes,), minval=0.1, maxval=2.0)
        running = cumulative(trapezoid_rule(nodes), values)
        assert bool(jnp.all(jnp.diff(running) >= 0.0))

    @pytest.mark.slow
    @settings(max_examples=15, deadline=None)
    @given(n_nodes=st.integers(1, 30), seed=st.integers(0, 2**16))
    def test_forward_plus_reversed_is_the_total(self, n_nodes: int, seed: int) -> None:
        key_x, key_f = jax.random.split(jax.random.key(seed))
        nodes = jnp.sort(jax.random.uniform(key_x, (n_nodes,)))
        values = jax.random.normal(key_f, (n_nodes,))
        rule = trapezoid_rule(nodes)
        forward = cumulative(rule, values)
        backward = cumulative(rule, values, reverse=True)
        total = integrate(rule, values).value
        assert_matches_scipy(
            forward + backward, jnp.full(n_nodes, total), rtol=1e-12, atol=1e-13
        )


class TestValidation:
    def test_rejects_weights_without_nodes(self) -> None:
        # In the suite, jaxtyping's import hook rejects the annotation first;
        # without it, the boundary check raises ValidationError.
        with pytest.raises((ix.ValidationError, TypeCheckError)):
            cumulative(jnp.ones(3), jnp.ones(3))

    def test_rejects_a_rule_without_nodes(self) -> None:
        class WeightsOnly:
            weights = jnp.ones(3)

        with pytest.raises(ix.ValidationError, match="'nodes'"):
            cumulative(WeightsOnly(), jnp.ones(3))

    def test_names_weights_for_a_rule_with_only_nodes(self) -> None:
        class NodesOnly:
            nodes = jnp.arange(3.0)

        with pytest.raises((ix.ValidationError, TypeCheckError)):
            cumulative(NodesOnly(), jnp.ones(3))

    @pytest.mark.parametrize("nodes", [jnp.ones((2, 2)), jnp.ones(0)])
    def test_rejects_nodes_that_are_not_1d(self, nodes: jax.Array) -> None:
        rule = StandInRule(nodes=nodes, weights=nodes)
        with pytest.raises(ix.ValidationError, match="1-D nodes"):
            cumulative(rule, jnp.ones(4))

    def test_rejects_an_axis_out_of_range(self) -> None:
        with pytest.raises(ix.ValidationError, match="within the values' 1 axes"):
            cumulative(trapezoid_rule(jnp.arange(3.0)), jnp.ones(3), axis=1)

    def test_rejects_values_with_the_wrong_node_count(self) -> None:
        with pytest.raises(ix.ValidationError, match="one entry per node"):
            cumulative(trapezoid_rule(jnp.arange(3.0)), jnp.ones(4))
