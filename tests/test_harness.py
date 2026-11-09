"""Self-tests for the reference harness in `tests/_harness.py`.

Each helper must pass on a case that is right and fail, with the numbers
behind the failure, on a case that is wrong by a small perturbation, so a test
built on the harness that passes means something.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import (
    assert_exact_on_polynomials,
    assert_float32_drift,
    assert_grad_matches_fd,
    assert_matches_scipy,
    assert_order,
    assert_traces,
    assert_within_stderr,
)
from scipy.integrate import cumulative_trapezoid, trapezoid


# ε for the perturbed cases: far above float64 round-off (≈ 1e-16), far below
# anything a reader would call wrong by eye.
PERTURBATION = 1e-6


def _gauss_legendre(n_nodes: int) -> tuple[jax.Array, jax.Array]:
    """The n-point Gauss–Legendre rule on [−1, 1], from NumPy."""
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    return jnp.asarray(nodes), jnp.asarray(weights)


def _trapezoid_weights(nodes: jax.Array) -> jax.Array:
    """w₀ = (x₁ − x₀)/2, wₖ = (xₖ₊₁ − xₖ₋₁)/2, w_N = (x_N − x_{N−1})/2."""
    widths = jnp.diff(nodes)
    left = jnp.concatenate([widths, jnp.zeros(1)])
    right = jnp.concatenate([jnp.zeros(1), widths])
    return 0.5 * (left + right)


class TestMatchesScipy:
    def test_passes_against_trapezoid_on_non_uniform_nodes(self) -> None:
        rng = np.random.default_rng(0)
        nodes = np.sort(rng.uniform(0.0, 3.0, 50))
        values = np.sin(nodes)
        actual = _trapezoid_weights(jnp.asarray(nodes)) @ jnp.asarray(values)
        assert_matches_scipy(actual, trapezoid(values, nodes), rtol=1e-14)

    def test_passes_against_cumulative_trapezoid(self) -> None:
        nodes = np.array([0.0, 0.3, 1.0, 1.1])
        values = nodes**2
        increments = 0.5 * np.diff(nodes) * (values[:-1] + values[1:])
        assert_matches_scipy(
            np.cumsum(increments), cumulative_trapezoid(values, nodes), rtol=1e-14
        )

    def test_names_the_worst_index(self) -> None:
        reference = np.array([[1.0, 2.0], [3.0, 4.0]])
        actual = reference.copy()
        actual[1, 0] += 1e-3
        actual[0, 1] += 1e-9
        with pytest.raises(AssertionError, match=r"2 of 4 .* worst at index \(1, 0\)"):
            assert_matches_scipy(actual, reference, rtol=1e-12)

    def test_nan_matches_only_nan(self) -> None:
        assert_matches_scipy(np.array([np.nan, 1.0]), [np.nan, 1.0], rtol=1e-12)
        with pytest.raises(AssertionError, match="worst at index \\(0,\\)"):
            assert_matches_scipy(np.array([np.nan, 1.0]), [0.0, 1.0], rtol=1e-12)

    def test_rejects_a_shape_mismatch(self) -> None:
        with pytest.raises(AssertionError, match="shape mismatch"):
            assert_matches_scipy(np.zeros(3), np.zeros(4), rtol=1e-12)


class TestExactOnPolynomials:
    @pytest.mark.parametrize("n_nodes", [1, 2, 5, 8])
    def test_gauss_legendre_is_exact_to_degree_2n_minus_1(self, n_nodes: int) -> None:
        nodes, weights = _gauss_legendre(n_nodes)
        assert_exact_on_polynomials(
            lambda f: weights @ f(nodes), 2 * n_nodes - 1, interval=(-1.0, 1.0)
        )

    def test_fails_one_degree_past_exactness(self) -> None:
        nodes, weights = _gauss_legendre(3)
        with pytest.raises(AssertionError, match="not exact for degree 6"):
            assert_exact_on_polynomials(
                lambda f: weights @ f(nodes), 6, interval=(-1.0, 1.0)
            )

    def test_fails_on_perturbed_weights(self) -> None:
        nodes, weights = _gauss_legendre(4)
        perturbed = weights.at[0].add(PERTURBATION)
        with pytest.raises(AssertionError, match="not exact for degree 0"):
            assert_exact_on_polynomials(
                lambda f: perturbed @ f(nodes), 7, interval=(-1.0, 1.0)
            )

    def test_trapezoid_is_exact_for_lines_on_a_shifted_interval(self) -> None:
        nodes = jnp.array([2.0, 2.5, 4.0])
        weights = _trapezoid_weights(nodes)
        assert_exact_on_polynomials(lambda f: weights @ f(nodes), 1, interval=(2, 4))


class TestOrder:
    @staticmethod
    def _euler_errors(order: int) -> dict[float, float]:
        """Global error at T = 1 on y′ = λy, y(T) = y₀ e^{λT}, λ = −1."""
        errors = {}
        for n_steps in (20, 40, 80, 160):
            step = 1.0 / n_steps
            # Euler: yₙ₊₁ = (1 + λh) yₙ; Heun: yₙ₊₁ = (1 + λh + (λh)²/2) yₙ
            growth = 1.0 - step if order == 1 else 1.0 - step + 0.5 * step**2
            errors[step] = abs(growth**n_steps - math.exp(-1.0))
        return errors

    @pytest.mark.parametrize("order", [1, 2])
    def test_passes_on_euler_and_heun(self, order: int) -> None:
        assert_order(self._euler_errors(order), order)

    def test_fails_on_the_wrong_order(self) -> None:
        with pytest.raises(AssertionError, match=r"observed order 1\.0"):
            assert_order(self._euler_errors(1), 2)

    def test_trapezoid_converges_at_second_order(self) -> None:
        # ∫₀¹ eˣ dx = e − 1, with the trapezoid's O(h²) error
        errors = {}
        for n_nodes in (11, 21, 41, 81):
            nodes = jnp.linspace(0.0, 1.0, n_nodes)
            total = _trapezoid_weights(nodes) @ jnp.exp(nodes)
            errors[1.0 / (n_nodes - 1)] = abs(float(total) - (math.e - 1.0))
        assert_order(errors, 2)

    def test_rejects_too_few_steps(self) -> None:
        with pytest.raises(AssertionError, match="at least two"):
            assert_order({0.1: 1e-2}, 1)

    def test_rejects_a_zero_error(self) -> None:
        with pytest.raises(AssertionError, match="positive"):
            assert_order({0.1: 1e-2, 0.05: 0.0}, 1)


class TestGradMatchesFd:
    def test_passes_through_a_weighted_sum(self) -> None:
        # ∫₀^π sin x dx = 2; d/dwₖ ∑ wⱼ sin xⱼ = sin xₖ
        nodes = jnp.linspace(0.0, math.pi, 9)
        assert_grad_matches_fd(lambda w: w @ jnp.sin(nodes), _trapezoid_weights(nodes))

    def test_passes_on_a_pytree(self) -> None:
        x = {"a": jnp.array([0.3, 0.9]), "b": jnp.array(1.2)}
        assert_grad_matches_fd(lambda p: jnp.sin(p["a"]) * p["b"], x)

    def test_fails_on_a_wrong_custom_gradient(self) -> None:
        @jax.custom_jvp
        def scaled_sin(x: jax.Array) -> jax.Array:
            return jnp.sin(x)

        @scaled_sin.defjvp
        def _jvp(primals, tangents):
            (x,), (dx,) = primals, tangents
            return jnp.sin(x), (jnp.cos(x) + 1e-3) * dx

        with pytest.raises(AssertionError, match="worst at index"):
            assert_grad_matches_fd(scaled_sin, jnp.array([0.2, 0.5, 1.1]))


class TestWithinStderr:
    @staticmethod
    def _mean_estimate() -> tuple[float, float]:
        """Î and σ̂ for ∫₀¹ eˣ dx by plain Monte Carlo with 10,000 draws."""
        draws = jax.random.uniform(jax.random.key(0), (10_000,))
        samples = jnp.exp(draws)
        # σ̂ = s / √N
        standard_error = jnp.std(samples, ddof=1) / jnp.sqrt(samples.shape[0])
        return float(jnp.mean(samples)), float(standard_error)

    def test_passes_on_monte_carlo(self) -> None:
        estimate, standard_error = self._mean_estimate()
        assert_within_stderr(estimate, standard_error, math.e - 1.0)

    def test_fails_on_a_biased_estimate(self) -> None:
        estimate, standard_error = self._mean_estimate()
        with pytest.raises(AssertionError, match="standard errors"):
            assert_within_stderr(estimate + 10 * standard_error, standard_error, 1.0)

    def test_rejects_a_zero_standard_error(self) -> None:
        with pytest.raises(AssertionError, match="positive"):
            assert_within_stderr(1.0, 0.0, 1.0)

    def test_names_the_worst_entry(self) -> None:
        with pytest.raises(AssertionError, match=r"index \(1,\)"):
            assert_within_stderr([1.0, 2.0], [0.1, 0.1], [1.0, 1.0])


class TestTraces:
    def test_passes_on_a_batched_contraction(self) -> None:
        values = jax.random.normal(jax.random.key(3), (4, 6))
        weights = jnp.full((4, 6), 0.2)
        assert_traces(lambda w, v: w @ v, weights, values)

    def test_fails_when_jit_and_eager_differ(self) -> None:
        def leaks_tracing(x: jax.Array) -> jax.Array:
            # a Python-side branch: eager sees a concrete array, jit a tracer
            offset = 0.0 if isinstance(x, jax.core.Tracer) else PERTURBATION
            return x + offset

        with pytest.raises(AssertionError, match=r"jit\(fn\)"):
            assert_traces(leaks_tracing, jnp.ones((3, 2)))

    def test_rejects_mismatched_batches(self) -> None:
        with pytest.raises(AssertionError, match="leading batch axis"):
            assert_traces(lambda a, b: a + b[0], jnp.ones(3), jnp.ones(4))


class TestFloat32Drift:
    @staticmethod
    def _left_to_right_sum(values: jax.Array) -> jax.Array:
        """A running sum, the accumulation pairwise summation avoids."""
        total, _ = jax.lax.scan(
            lambda carry, value: (carry + value, None),
            jnp.zeros((), values.dtype),
            values,
        )
        return total

    @staticmethod
    def _pairwise_sum(values: jax.Array) -> jax.Array:
        """∑ by halving: O(log N) roundings per value."""
        while values.shape[0] > 1:
            if values.shape[0] % 2:
                values = jnp.concatenate([values, jnp.zeros(1, values.dtype)])
            values = values[0::2] + values[1::2]
        return values[0]

    def test_pairwise_sum_stays_close_over_a_long_spectrum(self) -> None:
        # jit traces the halving loop once per dtype instead of dispatching
        # every halving eagerly
        pairwise_sum = jax.jit(self._pairwise_sum)
        assert_float32_drift(pairwise_sum, jnp.full(100_000, 0.1), rtol=1e-5)

    def test_left_to_right_sum_drifts(self) -> None:
        with pytest.raises(AssertionError, match="float32 drifts"):
            assert_float32_drift(
                self._left_to_right_sum, jnp.full(100_000, 0.1), rtol=1e-5
            )

    def test_fails_when_float32_comes_back_promoted(self) -> None:
        with pytest.raises(AssertionError, match="came back as float64"):
            assert_float32_drift(
                lambda v: jnp.sum(v.astype(jnp.float64)), jnp.ones(10), rtol=1e-5
            )

    def test_rejects_a_float32_reference(self) -> None:
        with pytest.raises(AssertionError, match="must be float64"):
            assert_float32_drift(jnp.sum, jnp.ones(10, dtype=jnp.float32), rtol=1e-5)


def test_helpers_refuse_to_run_without_x64(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jax.config, "read", lambda name: False)
    with pytest.raises(AssertionError, match="needs float64"):
        assert_traces(jnp.sin, jnp.ones(2))
