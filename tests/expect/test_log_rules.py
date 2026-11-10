"""Tests for `integrix.expect.log_gauss_hermite` and `log_simpson` (#69).

Oracles: log 𝔼[Q] = μ + σ²/2 for a lognormal Q, and ∫₁^∞ 3q⁻⁴ dq = 1 (truncated
at 10⁶), both to rtol 1e-10 in float64; a heavy-tailed integrand that underflows
in linear q stays finite in log space; gradients in μ, σ and the parameters of
log f match central differences; jit and vmap over (μ, σ) agree with a loop.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_traces
from jaxtyping import TypeCheckError

import integrix as ix
from integrix.expect import log_gauss_hermite, log_simpson


@pytest.mark.parametrize(("mean", "std"), [(0.2, 0.5), (-1.0, 1.2), (3.0, 0.1)])
def test_gauss_hermite_lognormal_mean(mean: float, std: float) -> None:
    # log 𝔼[Q] = μ + σ²/2
    value = log_gauss_hermite(jnp.log, mean, std)
    np.testing.assert_allclose(value, mean + std**2 / 2.0, rtol=1e-10)


def test_gauss_hermite_moments_of_q() -> None:
    # log 𝔼[Qᵃ] = aμ + a²σ²/2
    for power in (2.0, -1.5, 3.0):
        value = log_gauss_hermite(lambda q, a=power: a * jnp.log(q), 0.4, 0.6)
        np.testing.assert_allclose(value, power * 0.4 + power**2 * 0.36 / 2, rtol=1e-10)


def test_gauss_hermite_is_exact_for_polynomials_in_log_q() -> None:
    # 𝔼[(log Q)²] = μ² + σ², exact with two nodes
    value = log_gauss_hermite(
        lambda q: 2.0 * jnp.log(jnp.abs(jnp.log(q))), 0.7, 0.3, n=2
    )
    np.testing.assert_allclose(value, math.log(0.7**2 + 0.3**2), rtol=1e-12)


def test_gauss_hermite_batches_over_mean_and_std() -> None:
    means = jnp.array([0.0, 1.0, 2.0])
    stds = jnp.array([0.5, 0.2, 0.1])
    values = log_gauss_hermite(jnp.log, means, stds)
    assert values.shape == (3,)
    np.testing.assert_allclose(values, means + stds**2 / 2.0, rtol=1e-10)


def test_simpson_power_law_tail() -> None:
    # ∫₁^10⁶ 3 q⁻⁴ dq = 1 − 10⁻¹⁸; Simpson's error falls like h⁴, and 4001
    # nodes reach 1e-10 (the default 2001 reach about 1e-9)
    value = log_simpson(lambda q: math.log(3.0) - 4.0 * jnp.log(q), 1.0, 1e6, n=4001)
    np.testing.assert_allclose(value, 0.0, atol=1e-10)


def test_simpson_stays_finite_where_linear_q_underflows() -> None:
    # ∫₁^e 10⁻⁴⁰⁰ dq/q: every value underflows in float64, its log does not
    value = log_simpson(lambda q: -400.0 * math.log(10.0) - jnp.log(q), 1.0, math.e)
    np.testing.assert_allclose(value, -400.0 * math.log(10.0), rtol=1e-12)


def test_simpson_makes_an_even_count_odd() -> None:
    odd = log_simpson(lambda q: -jnp.log(q), 1.0, 3.0, n=101)
    even = log_simpson(lambda q: -jnp.log(q), 1.0, 3.0, n=100)
    np.testing.assert_allclose(even, odd, rtol=1e-15)


class TestValidation:
    @pytest.mark.parametrize("n", [0, True])
    def test_gauss_hermite_rejects_a_bad_count(self, n: int) -> None:
        with pytest.raises((ix.ValidationError, TypeCheckError), match="n"):
            log_gauss_hermite(jnp.log, 0.0, 1.0, n=n)

    @pytest.mark.parametrize("n", [2, -1])
    def test_simpson_rejects_a_bad_count(self, n: int) -> None:
        with pytest.raises(ix.ValidationError, match="at least 3"):
            log_simpson(jnp.log, 1.0, 2.0, n=n)

    @pytest.mark.parametrize(
        ("lower", "upper"), [(0.0, 1.0), (2.0, 1.0), (1.0, np.inf)]
    )
    def test_simpson_rejects_bad_limits(self, lower: float, upper: float) -> None:
        with pytest.raises(ix.ValidationError, match="0 < lower < upper"):
            log_simpson(jnp.log, lower, upper)


@pytest.mark.slow
class TestTransforms:
    def test_gradient_in_mean_std_and_the_integrand(self) -> None:
        def log_moment(params: jax.Array) -> jax.Array:
            mean, std, power = params[0], params[1], params[2]
            return log_gauss_hermite(lambda q: power * jnp.log(q), mean, std)

        assert_grad_matches_fd(log_moment, jnp.array([0.3, 0.6, 1.5]))

    def test_simpson_gradient_in_the_integrand(self) -> None:
        def log_tail(exponent: jax.Array) -> jax.Array:
            return log_simpson(lambda q: -exponent * jnp.log(q), 1.0, 1e3, n=401)

        assert_grad_matches_fd(log_tail, jnp.asarray(3.0))

    def test_jit_and_vmap_over_mean_and_std_agree_with_a_loop(self) -> None:
        means = jnp.array([0.0, 0.5, 1.0])
        stds = jnp.array([0.2, 0.4, 0.6])
        assert_traces(lambda m, s: log_gauss_hermite(jnp.log, m, s), means, stds)
