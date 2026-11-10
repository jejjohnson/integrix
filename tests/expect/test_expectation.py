"""Tests for `integrix.expect` (#27).

Gate G4-bounded: every estimate lies within four of its own standard errors
of the exact value, on Gaussian moments and on ∫ over [0, 1]^D of ∏ cos x_d.
Randomised QMC's error falls faster than S^{−1/2} on a smooth integrand, and
gradients by reparametrisation match finite differences within their Monte
Carlo error. Every draw comes from a fixed key, so each bound is checked on a
reproducible sample.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np
import pytest
from _harness import assert_matches_scipy, assert_within_stderr
from jax.scipy.special import ndtri
from jaxtyping import TypeCheckError

import integrix as ix
from integrix.expect import Expectation, importance, monte_carlo, qmc


MEAN = 0.7
SCALE = 1.3


def _gaussian(key, n):
    """x ~ N(μ, s²), by reparametrisation."""
    return MEAN + SCALE * jax.random.normal(key, (n,))


def _cosine_product(point):
    """∏_d cos x_d, whose integral over [0, 1]^D is sin(1)^D."""
    return jnp.prod(jnp.cos(point))


class TestMonteCarlo:
    @pytest.mark.parametrize(
        ("fn", "exact"),
        [(lambda x: x, MEAN), (jnp.square, MEAN**2 + SCALE**2)],
        ids=["mean", "second_moment"],
    )
    def test_gaussian_moments_within_four_standard_errors(self, fn, exact) -> None:
        estimate = monte_carlo(fn, _gaussian, 20_000, key=jax.random.key(0))
        assert_within_stderr(estimate.value, estimate.standard_error, exact)

    def test_accepts_draws_already_made(self) -> None:
        # a NumPyro-style dict of posterior draws with a leading sample axis
        key_mu, key_sigma = jax.random.split(jax.random.key(1))
        draws = {
            "mu": MEAN + jax.random.normal(key_mu, (5000,)),
            "sigma": jnp.exp(0.1 * jax.random.normal(key_sigma, (5000,))),
        }
        estimate = monte_carlo(lambda d: d["mu"] * d["sigma"], draws)
        # 𝔼[μ σ] = 𝔼[μ] 𝔼[σ] = μ e^{0.005}
        assert_within_stderr(
            estimate.value, estimate.standard_error, MEAN * math.exp(0.005)
        )
        assert estimate.n_samples == 5000
        assert estimate.effective_sample_size is None

    def test_vector_valued_integrand(self) -> None:
        estimate = monte_carlo(
            lambda x: jnp.stack([x, x**2]), _gaussian, 20_000, key=jax.random.key(2)
        )
        assert estimate.value.shape == (2,)
        assert_within_stderr(
            estimate.value,
            estimate.standard_error,
            np.array([MEAN, MEAN**2 + SCALE**2]),
        )

    def test_standard_error_is_the_sample_formula(self) -> None:
        values = jnp.array([1.0, 2.0, 4.0, 7.0])
        estimate = monte_carlo(lambda x: x, values)
        assert_matches_scipy(
            estimate.standard_error,
            np.std(np.asarray(values), ddof=1) / 2.0,
            rtol=1e-14,
        )

    def test_runs_under_jit(self) -> None:
        traced = jax.jit(
            lambda key: monte_carlo(jnp.square, _gaussian, 1000, key=key).value
        )(jax.random.key(3))
        eager = monte_carlo(jnp.square, _gaussian, 1000, key=jax.random.key(3)).value
        assert_matches_scipy(traced, eager, rtol=1e-14)


class TestReparametrisedGradient:
    def test_matches_finite_differences_and_the_exact_gradient(self) -> None:
        # d/dμ 𝔼_{x~N(μ, s²)}[x²] = 2μ, with common random numbers
        noise = jax.random.normal(jax.random.key(4), (20_000,))

        def estimate(mean: jax.Array) -> jax.Array:
            return monte_carlo(jnp.square, mean + SCALE * noise).value

        gradient = jax.grad(estimate)(jnp.asarray(MEAN))
        step = 1e-5
        central = (estimate(MEAN + step) - estimate(MEAN - step)) / (2 * step)
        assert_matches_scipy(gradient, central, rtol=1e-6)
        # its own Monte Carlo error: the per-draw gradients 2 xₛ
        per_draw = monte_carlo(lambda x: 2.0 * x, MEAN + SCALE * noise)
        assert_within_stderr(gradient, per_draw.standard_error, 2.0 * MEAN)


class TestQMC:
    @pytest.mark.parametrize("sequence", ["sobol", "halton"])
    def test_cosine_product_within_four_standard_errors(self, sequence: str) -> None:
        estimate = qmc(
            _cosine_product, 5, 512, sequence=sequence, key=jax.random.key(5)
        )
        assert_within_stderr(
            estimate.value, estimate.standard_error, math.sin(1.0) ** 5
        )
        assert estimate.n_samples == 512 * 16

    def test_gaussian_moments_through_a_transform(self) -> None:
        estimate = qmc(
            lambda z: jnp.square(MEAN + SCALE * z[0]),
            1,
            1024,
            transform=ndtri,
            key=jax.random.key(6),
        )
        assert_within_stderr(
            estimate.value, estimate.standard_error, MEAN**2 + SCALE**2
        )

    @pytest.mark.slow
    def test_error_falls_faster_than_monte_carlo(self) -> None:
        # |Î − I| averaged over 8 keys, n = 64 … 4096 points per shift: plain
        # Monte Carlo would give a slope of −1/2
        exact = math.sin(1.0) ** 3
        sizes = [64, 256, 1024, 4096]
        errors = []
        for n in sizes:
            per_key = [
                abs(
                    float(
                        qmc(
                            _cosine_product, 3, n, n_shifts=2, key=jax.random.key(seed)
                        ).value
                    )
                    - exact
                )
                for seed in range(8)
            ]
            errors.append(np.mean(per_key))
        slope, _ = np.polyfit(np.log(sizes), np.log(errors), 1)
        # measured −0.83 here, against −0.50 for plain Monte Carlo on the same
        # integrand and sizes; the bound leaves room for the key-to-key spread
        assert slope < -0.65, slope

    def test_shifts_change_the_estimate(self) -> None:
        first = qmc(_cosine_product, 2, 64, key=jax.random.key(7)).value
        second = qmc(_cosine_product, 2, 64, key=jax.random.key(8)).value
        assert float(first) != float(second)


class TestImportance:
    def test_gaussian_moment_from_a_wider_proposal(self) -> None:
        estimate = importance(
            jnp.square,
            lambda key, n: 2.0 * jax.random.normal(key, (n,)),
            lambda x: stats.norm.logpdf(x, scale=2.0),
            lambda x: stats.norm.logpdf(x, loc=MEAN, scale=SCALE),
            20_000,
            key=jax.random.key(9),
        )
        assert_within_stderr(
            estimate.value, estimate.standard_error, MEAN**2 + SCALE**2
        )
        assert 0.0 < float(estimate.effective_sample_size) < 20_000

    def test_an_unnormalised_target_gives_the_same_estimate(self) -> None:
        def run(offset: float) -> Expectation:
            return importance(
                jnp.square,
                lambda key, n: 2.0 * jax.random.normal(key, (n,)),
                lambda x: stats.norm.logpdf(x, scale=2.0),
                lambda x: stats.norm.logpdf(x) + offset,
                2000,
                key=jax.random.key(10),
            )

        assert_matches_scipy(run(0.0).value, run(5.0).value, rtol=1e-12)

    def test_the_target_as_proposal_has_full_effective_sample_size(self) -> None:
        estimate = importance(
            jnp.square,
            lambda key, n: jax.random.normal(key, (n,)),
            stats.norm.logpdf,
            stats.norm.logpdf,
            1000,
            key=jax.random.key(11),
        )
        assert_matches_scipy(estimate.effective_sample_size, 1000.0, rtol=1e-12)


class TestIntegerAndBooleanIntegrands:
    def test_a_probability_from_a_boolean(self) -> None:
        estimate = monte_carlo(
            lambda x: x > MEAN, _gaussian, 20_000, key=jax.random.key(12)
        )
        assert estimate.value.dtype == jnp.result_type(float)
        assert_within_stderr(estimate.value, estimate.standard_error, 0.5)

    def test_qmc_of_an_indicator(self) -> None:
        estimate = qmc(lambda u: u[0] < 0.3, 1, 256, key=jax.random.key(13))
        assert_within_stderr(estimate.value, estimate.standard_error, 0.3)

    def test_importance_of_an_integer_integrand(self) -> None:
        estimate = importance(
            lambda x: (x > 0).astype(jnp.int32),
            lambda key, n: 2.0 * jax.random.normal(key, (n,)),
            lambda x: stats.norm.logpdf(x, scale=2.0),
            stats.norm.logpdf,
            20_000,
            key=jax.random.key(14),
        )
        assert_within_stderr(estimate.value, estimate.standard_error, 0.5)
        assert bool(jnp.isfinite(estimate.effective_sample_size))


class TestPointsStayInside:
    @pytest.mark.parametrize("sequence", ["sobol", "halton"])
    def test_no_point_reaches_zero_or_one(self, sequence: str) -> None:
        # the integrand counts points on the boundary; there must be none
        on_boundary = lambda u: jnp.any((u <= 0.0) | (u >= 1.0)).astype(float)
        estimate = qmc(
            on_boundary, 3, 4096, sequence=sequence, n_shifts=64, key=jax.random.key(32)
        )
        assert float(estimate.value) == 0.0

    def test_sobol_midpoints_in_float32_stay_below_one(self) -> None:
        from integrix._src.expect._sequences import SOBOL_BITS

        largest = jnp.uint32(2**SOBOL_BITS - 1)
        precision = jnp.finfo(jnp.float32).nmant
        leading = largest >> jnp.uint32(SOBOL_BITS - precision)
        point = (leading.astype(jnp.float32) + 0.5) / 2.0**precision
        assert float(point) < 1.0


class TestValidation:
    def test_importance_refuses_a_target_without_support(self) -> None:
        with pytest.raises(Exception, match="zero density"):
            importance(
                jnp.square,
                lambda key, n: jax.random.uniform(key, (n,)),
                lambda x: 0.0 * x,
                lambda x: jnp.where(x > 2.0, 0.0, -jnp.inf),
                100,
                key=jax.random.key(15),
            )

    def test_a_sampler_needs_a_key_and_a_count(self) -> None:
        with pytest.raises(ix.ValidationError, match="'key' and 'n'"):
            monte_carlo(jnp.square, _gaussian, 10)

    def test_a_standard_error_needs_two_draws(self) -> None:
        with pytest.raises(ix.ValidationError, match="at least two"):
            monte_carlo(jnp.square, jnp.ones(1))

    def test_the_count_must_match_the_draws(self) -> None:
        with pytest.raises(ix.ValidationError, match="'n' is 5"):
            monte_carlo(jnp.square, jnp.ones(4), 5)

    def test_draws_must_share_a_sample_axis(self) -> None:
        with pytest.raises(ix.ValidationError, match="one sample count"):
            monte_carlo(lambda d: d["a"], {"a": jnp.ones(3), "b": jnp.ones(4)})

    def test_qmc_rejects_a_single_shift(self) -> None:
        with pytest.raises(ix.ValidationError, match="'n_shifts'"):
            qmc(_cosine_product, 2, 16, n_shifts=1, key=jax.random.key(0))

    def test_qmc_rejects_an_unknown_sequence(self) -> None:
        # In the suite, jaxtyping's import hook rejects the Literal first.
        with pytest.raises((ix.ValidationError, TypeCheckError)):
            qmc(_cosine_product, 2, 16, sequence="lattice", key=jax.random.key(0))

    def test_qmc_rejects_too_many_sobol_dimensions(self) -> None:
        with pytest.raises(ix.ValidationError, match="halton"):
            qmc(_cosine_product, 65, 16, key=jax.random.key(0))

    def test_importance_needs_two_draws(self) -> None:
        with pytest.raises(ix.ValidationError, match="'n'"):
            importance(
                jnp.square,
                _gaussian,
                stats.norm.logpdf,
                stats.norm.logpdf,
                1,
                key=jax.random.key(0),
            )

    def test_is_reexported_from_the_expect_module(self) -> None:
        assert ix.expect.monte_carlo is monte_carlo
