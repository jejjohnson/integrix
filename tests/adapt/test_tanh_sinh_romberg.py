"""Tests for `TanhSinh`, `Romberg` and `cumulative_adaptive` (#25).

Tanh–sinh integrates ∫₀¹ log x dx = −1 and ∫₀¹ x^{−1/2} dx = 2 to 1e-12 and
handles infinite limits. Gate G3-scipy: each rule matches
`scipy.integrate.quad` to the requested tolerance on its battery. Gate
G3-grad: `jax.jacrev` matches central differences to 1e-6.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_matches_scipy
from jaxtyping import TypeCheckError
from scipy.integrate import quad

import integrix as ix
from integrix.adapt import GaussKronrod, Romberg, TanhSinh, cumulative_adaptive


def _scipy(fn, lower, upper, parameter) -> float:
    value, _ = quad(
        lambda x: float(fn(np.float64(x), parameter)),
        lower,
        upper,
        epsabs=0.0,
        epsrel=1e-12,
        limit=500,
    )
    return value


def _params(battery, fast):
    return [
        pytest.param(
            *case, id=case[0], marks=() if case[0] == fast else pytest.mark.slow
        )
        for case in battery
    ]


# (name, f(x, p), lower, upper, p)
TANH_SINH_BATTERY = [
    ("log_singular", lambda x, p: p * jnp.log(x), 0.0, 1.0, 1.5),
    ("inverse_sqrt", lambda x, p: x ** (-p), 0.0, 1.0, 0.5),
    ("smooth", lambda x, p: jnp.exp(p * x), -1.0, 2.0, 0.7),
    ("exp_decay_to_infinity", lambda x, p: jnp.exp(-p * x), 0.0, math.inf, 2.0),
    (
        "gaussian_over_the_line",
        lambda x, p: jnp.exp(-p * x**2),
        -math.inf,
        math.inf,
        0.8,
    ),
    ("cauchy_tail", lambda x, p: 1.0 / (p + x**2), 1.0, math.inf, 1.0),
]
ROMBERG_BATTERY = [
    ("exp", lambda x, p: jnp.exp(p * x), 0.0, 1.0, 1.3),
    ("oscillatory", lambda x, p: jnp.cos(p * x), 0.0, 1.0, 5.0),
    ("rational", lambda x, p: 1.0 / (p + x**2), 0.0, 1.0, 1.0),
    ("polynomial", lambda x, p: p * x**7 - x**3, -1.0, 2.0, 2.0),
]


class TestTanhSinh:
    @pytest.mark.parametrize(
        ("fn", "exact"),
        [(jnp.log, -1.0), (lambda x: x**-0.5, 2.0)],
        ids=["log", "inverse_sqrt"],
    )
    def test_endpoint_singularities_to_1e_12(self, fn, exact) -> None:
        result = TanhSinh(rtol=1e-13)(fn, 0.0, 1.0)
        assert bool(result.converged)
        assert abs(float(result.value) - exact) < 1e-12

    @pytest.mark.parametrize(
        ("fn", "lower", "upper", "exact"),
        [
            (lambda x: jnp.exp(-x), 0.0, math.inf, 1.0),
            (lambda x: jnp.exp(x), -math.inf, 0.0, 1.0),
            (lambda x: jnp.exp(-(x**2)), -math.inf, math.inf, math.sqrt(math.pi)),
            (lambda x: jnp.exp(-x), math.inf, 0.0, -1.0),
        ],
        ids=["zero_to_inf", "minus_inf_to_zero", "whole_line", "reversed"],
    )
    def test_infinite_limits(self, fn, lower, upper, exact) -> None:
        result = TanhSinh(rtol=1e-12)(fn, lower, upper)
        assert_matches_scipy(result.value, exact, rtol=1e-12)

    @pytest.mark.parametrize(
        ("name", "fn", "lower", "upper", "parameter"),
        _params(TANH_SINH_BATTERY, "log_singular"),
    )
    def test_matches_scipy(self, name, fn, lower, upper, parameter) -> None:
        rtol = 1e-10
        result = TanhSinh(rtol=rtol)(fn, lower, upper, args=parameter)
        assert bool(result.converged)
        assert_matches_scipy(
            result.value, _scipy(fn, lower, upper, parameter), rtol=rtol
        )

    @pytest.mark.parametrize(
        ("name", "fn", "lower", "upper", "parameter"),
        _params(TANH_SINH_BATTERY, "smooth"),
    )
    def test_gradient_in_the_parameter(self, name, fn, lower, upper, parameter) -> None:
        rule = TanhSinh(rtol=1e-12)
        assert_grad_matches_fd(
            lambda p: rule(fn, lower, upper, args=p).value,
            jnp.asarray(parameter),
            eps=1e-6,
            rtol=1e-6,
        )

    def test_gradient_in_the_finite_limits(self) -> None:
        gradient = jax.grad(
            lambda limits: TanhSinh(rtol=1e-12)(jnp.cos, limits[0], limits[1]).value
        )(jnp.array([0.2, 1.1]))
        assert_matches_scipy(
            gradient, np.array([-math.cos(0.2), math.cos(1.1)]), rtol=1e-10
        )

    def test_not_converged_with_too_few_levels(self) -> None:
        result = TanhSinh(rtol=1e-14, max_levels=1, min_levels=1)(
            lambda x: jnp.cos(30 * x), 0.0, 1.0
        )
        assert not bool(result.converged)

    def test_vector_valued_and_float32(self) -> None:
        result = TanhSinh(rtol=1e-10)(
            lambda x: jnp.stack([jnp.sin(x), x**2]), 0.0, math.pi
        )
        assert_matches_scipy(
            result.value, np.array([2.0, math.pi**3 / 3.0]), rtol=1e-10
        )
        single = TanhSinh(rtol=1e-5)(jnp.exp, jnp.float32(0.0), jnp.float32(1.0))
        assert single.value.dtype == jnp.float32

    def test_equal_infinite_limits_give_zero(self) -> None:
        for limit in (math.inf, -math.inf):
            result = TanhSinh()(lambda x: jnp.exp(-(x**2)), limit, limit)
            assert float(result.value) == 0.0
            assert bool(result.converged)

    def test_a_singular_finite_end_of_a_half_infinite_interval(self) -> None:
        # ∫₀^∞ x^{−1/2} e^{−x} dx = √π: the exp–sinh grid reaches close
        # enough to 0 that its truncation is below the tolerance
        result = TanhSinh(rtol=1e-10)(lambda x: x**-0.5 * jnp.exp(-x), 0.0, math.inf)
        assert bool(result.converged)
        assert_matches_scipy(result.value, math.sqrt(math.pi), rtol=1e-10)

    def test_never_returns_inf_as_converged(self) -> None:
        # the singularity at x = 1 ≠ 0 cannot be resolved below the float
        # spacing at 1, but no node may land on it and give inf
        result = TanhSinh()(lambda x: (x - 1.0) ** -0.5 * jnp.exp(-x), 1.0, math.inf)
        assert bool(jnp.isfinite(result.value))
        exact = math.sqrt(math.pi) * math.exp(-1.0)
        assert abs(float(result.value) - exact) < 1e-6 * exact

    def test_an_infinite_limit_as_a_jax_array(self) -> None:
        result = TanhSinh(rtol=1e-12)(lambda x: jnp.exp(-x), 0.0, jnp.inf)
        assert_matches_scipy(result.value, 1.0, rtol=1e-12)

    def test_a_traced_infinite_limit_raises(self) -> None:
        with pytest.raises(Exception, match="must be finite"):
            jax.jit(lambda b: TanhSinh()(jnp.exp, 0.0, b).value)(jnp.inf)

    def test_rejects_a_nan_limit(self) -> None:
        with pytest.raises(ix.ValidationError, match="NaN"):
            TanhSinh()(jnp.exp, math.nan, 1.0)


class TestRomberg:
    @pytest.mark.parametrize(
        ("name", "fn", "lower", "upper", "parameter"), _params(ROMBERG_BATTERY, "exp")
    )
    def test_matches_scipy(self, name, fn, lower, upper, parameter) -> None:
        rtol = 1e-10
        result = Romberg(rtol=rtol)(fn, lower, upper, args=parameter)
        assert bool(result.converged)
        assert_matches_scipy(
            result.value, _scipy(fn, lower, upper, parameter), rtol=rtol
        )

    @pytest.mark.parametrize(
        ("name", "fn", "lower", "upper", "parameter"),
        _params(ROMBERG_BATTERY, "oscillatory"),
    )
    def test_gradient_in_the_parameter(self, name, fn, lower, upper, parameter) -> None:
        rule = Romberg(rtol=1e-12)
        assert_grad_matches_fd(
            lambda p: rule(fn, lower, upper, args=p).value,
            jnp.asarray(parameter),
            eps=1e-6,
            rtol=1e-6,
        )

    def test_extrapolation_is_exact_on_low_degree_polynomials(self) -> None:
        # level ℓ's diagonal is exact for degree ≤ 2ℓ + 1, so x⁵ needs two levels
        result = Romberg(rtol=1e-14, max_levels=3, min_levels=1)(
            lambda x: x**5, 0.0, 1.0
        )
        assert_matches_scipy(result.value, 1.0 / 6.0, rtol=1e-14)

    def test_coarse_grids_on_the_integrand_s_zeros_are_not_accepted(self) -> None:
        # sin²(8πx) vanishes at every node of the 3-, 5- and 9-point grids
        result = Romberg(rtol=1e-10)(lambda x: jnp.sin(8 * jnp.pi * x) ** 2, 0.0, 1.0)
        assert_matches_scipy(result.value, 0.5, rtol=1e-10)

    def test_not_converged_with_too_few_levels(self) -> None:
        result = Romberg(rtol=1e-14, max_levels=2, min_levels=1)(
            lambda x: jnp.cos(30 * x), 0.0, 1.0
        )
        assert not bool(result.converged)
        assert int(result.n_evaluations) == 5

    def test_vmap_over_parameters_matches_a_loop(self) -> None:
        rule = Romberg(rtol=1e-10)
        rates = jnp.array([0.5, 1.0, 2.0])
        batched = jax.vmap(
            lambda p: rule(lambda x, p: jnp.exp(p * x), 0.0, 1.0, args=p).value
        )(rates)
        looped = jnp.stack(
            [rule(lambda x, p: jnp.exp(p * x), 0.0, 1.0, args=p).value for p in rates]
        )
        assert_matches_scipy(batched, looped, rtol=1e-14)

    def test_rejects_an_infinite_limit(self) -> None:
        with pytest.raises(ix.ValidationError, match="tanh–sinh"):
            Romberg()(jnp.exp, 0.0, math.inf)


class TestCumulativeAdaptive:
    def test_running_integral_of_exp(self) -> None:
        nodes = jnp.array([0.0, 0.3, 1.0, 1.7, 3.0])
        running = cumulative_adaptive(GaussKronrod(rtol=1e-12), jnp.exp, nodes)
        assert_matches_scipy(
            running.value, np.expm1(np.asarray(nodes)), rtol=1e-12, atol=1e-15
        )
        assert bool(running.converged)
        assert running.error.shape == (5,)
        assert float(running.error[0]) == 0.0

    def test_descending_nodes(self) -> None:
        nodes = jnp.array([1.0, 0.5, 0.0])
        running = cumulative_adaptive(GaussKronrod(rtol=1e-12), jnp.exp, nodes)
        assert_matches_scipy(
            running.value, np.exp(np.asarray(nodes)) - math.e, rtol=1e-12, atol=1e-15
        )

    def test_vector_valued_integrand(self) -> None:
        running = cumulative_adaptive(
            GaussKronrod(rtol=1e-12),
            lambda x: jnp.stack([x, x**2]),
            jnp.array([0.0, 1.0, 2.0]),
        )
        assert running.value.shape == (3, 2)
        assert_matches_scipy(running.value[-1], np.array([2.0, 8.0 / 3.0]), rtol=1e-12)

    def test_a_single_node_gives_zero(self) -> None:
        running = cumulative_adaptive(GaussKronrod(), jnp.exp, jnp.array([0.4]))
        assert running.value.shape == (1,)
        assert float(running.value[0]) == 0.0

    def test_gradient_in_the_parameter(self) -> None:
        nodes = jnp.linspace(0.0, 2.0, 5)
        assert_grad_matches_fd(
            lambda p: (
                cumulative_adaptive(
                    GaussKronrod(rtol=1e-12), lambda x, p: jnp.sin(p * x), nodes, args=p
                ).value
            ),
            jnp.asarray(1.3),
        )

    def test_last_entry_equals_one_integral(self) -> None:
        nodes = jnp.linspace(0.0, 2.0, 6)
        running = cumulative_adaptive(GaussKronrod(rtol=1e-12), jnp.cos, nodes)
        whole = GaussKronrod(rtol=1e-12)(jnp.cos, 0.0, 2.0).value
        assert_matches_scipy(running.value[-1], whole, rtol=1e-12)

    def test_rejects_a_rule_that_is_not_gauss_kronrod(self) -> None:
        with pytest.raises((ix.ValidationError, TypeCheckError)):
            cumulative_adaptive(Romberg(), jnp.exp, jnp.array([0.0, 1.0]))

    def test_rejects_nodes_that_are_not_a_vector(self) -> None:
        with pytest.raises(ix.ValidationError, match="'nodes'"):
            cumulative_adaptive(GaussKronrod(), jnp.exp, jnp.zeros((2, 2)))


class TestValidation:
    @pytest.mark.parametrize("rule", [TanhSinh, Romberg])
    def test_rejects_a_negative_tolerance(self, rule) -> None:
        with pytest.raises(ix.ValidationError, match="'rtol'"):
            rule(rtol=-1.0)

    @pytest.mark.parametrize("rule", [TanhSinh, Romberg])
    def test_rejects_no_levels(self, rule) -> None:
        with pytest.raises(ix.ValidationError, match="'max_levels'"):
            rule(max_levels=0)

    @pytest.mark.parametrize("rule", [TanhSinh, Romberg])
    def test_rejects_a_floor_above_the_ceiling(self, rule) -> None:
        with pytest.raises(ix.ValidationError, match="'min_levels'"):
            rule(max_levels=2, min_levels=3)

    def test_non_finite_levels_never_pass(self) -> None:
        from integrix._src.adapt._levels import first_converged

        *_, converged = first_converged(
            jnp.array([jnp.inf, jnp.inf]),
            jnp.array([jnp.inf, jnp.nan]),
            rtol=1e-8,
            atol=0.0,
        )
        assert not bool(converged)

    def test_are_reexported_from_the_adapt_module(self) -> None:
        assert ix.adapt.TanhSinh is TanhSinh
        assert ix.adapt.Romberg is Romberg
        assert ix.adapt.cumulative_adaptive is cumulative_adaptive
