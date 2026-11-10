"""Tests for `integrix.adapt.GaussKronrod` (#24).

Gate G3-scipy: the result matches `scipy.integrate.quad` to the requested
tolerance on smooth, oscillatory and peaked integrands, and the error estimate
bounds the actual error. Gate G3-grad: `jax.jacrev` matches central
differences to 1e-6, in the integrand's parameters and in the limits. The
nodes and weights equal QUADPACK's as quadax tabulates them.
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
from integrix._src.adapt._gauss_kronrod_tables import GAUSS, KRONROD, NODES
from integrix.adapt import GaussKronrod


def _monomial_integral(degree: int) -> float:
    """∫₋₁¹ xᵈ dx."""
    return (1.0 - (-1.0) ** (degree + 1)) / (degree + 1)


class TestTables:
    @pytest.mark.parametrize("order", sorted(NODES))
    def test_equal_quadpack_as_quadax_tabulates_it(self, order: int) -> None:
        from quadax._quad_weights import gk_weights

        reference = gk_weights[order]
        assert np.array_equal(np.array(NODES[order]), np.asarray(reference["xk"]))
        assert np.array_equal(np.array(KRONROD[order]), np.asarray(reference["wk"]))
        assert np.array_equal(np.array(GAUSS[order]), np.asarray(reference["wg"]))

    @pytest.mark.parametrize("order", sorted(NODES))
    def test_exactness_degrees(self, order: int) -> None:
        # K₂ₙ₊₁ is exact to degree 3n + 1, its embedded Gₙ to 2n − 1
        nodes = np.array(NODES[order])
        n_gauss = (order - 1) // 2
        for weights, degree in (
            (KRONROD[order], 3 * n_gauss + 1),
            (GAUSS[order], 2 * n_gauss - 1),
        ):
            for d in range(degree + 1):
                assert abs(np.array(weights) @ nodes**d - _monomial_integral(d)) < 1e-14


# (name, integrand f(x, p), limits, parameter p)
BATTERY = [
    ("exp", lambda x, p: jnp.exp(p * x), (0.0, 1.0), 1.3),
    # on [0, 1], where the integral sin(20)/20 is far from zero
    ("oscillatory", lambda x, p: jnp.cos(p * x), (0.0, 1.0), 20.0),
    ("peaked", lambda x, p: 1.0 / (p**2 + (x - 0.3) ** 2), (0.0, 1.0), 0.01),
    (
        "narrow_gaussian",
        lambda x, p: jnp.exp(-(((x - 0.5) / p) ** 2)),
        (0.0, 1.0),
        0.02,
    ),
    ("polynomial", lambda x, p: p * x**7 - x**3, (-1.0, 2.0), 2.0),
]
FAST_CASE = "peaked"


def _battery_params():
    return [
        pytest.param(
            *case, id=case[0], marks=() if case[0] == FAST_CASE else pytest.mark.slow
        )
        for case in BATTERY
    ]


def _scipy_reference(fn, limits, parameter) -> float:
    value, _ = quad(
        lambda x: float(fn(np.float64(x), parameter)),
        *limits,
        epsabs=0.0,
        epsrel=1e-12,
        limit=500,
    )
    return value


class TestMatchesScipy:
    @pytest.mark.parametrize(("name", "fn", "limits", "parameter"), _battery_params())
    @pytest.mark.parametrize("order", [15, 21])
    def test_to_the_requested_tolerance(
        self, name, fn, limits, parameter, order
    ) -> None:
        rtol = 1e-10
        result = GaussKronrod(order=order, rtol=rtol, max_intervals=128)(
            fn, *limits, args=parameter
        )
        reference = _scipy_reference(fn, limits, parameter)
        assert bool(result.converged)
        assert_matches_scipy(result.value, reference, rtol=rtol)
        # the estimate bounds the actual error
        assert abs(float(result.value) - reference) <= float(
            result.error
        ) + 1e-15 * abs(reference)

    @pytest.mark.slow
    def test_matches_quadax(self) -> None:
        import quadax

        fn = lambda x, p: 1.0 / (p**2 + (x - 0.3) ** 2)
        ours = GaussKronrod(order=21, rtol=1e-12, max_intervals=128)(
            fn, 0.0, 1.0, args=0.01
        )
        theirs, _ = quadax.quadgk(
            fn,
            jnp.array([0.0, 1.0]),
            args=(0.01,),
            epsabs=0.0,
            epsrel=1e-12,
            max_ninter=128,
        )
        assert_matches_scipy(ours.value, theirs, rtol=1e-11)


class TestResult:
    def test_converged_is_false_when_the_buffer_fills(self) -> None:
        peak = lambda x: 1.0 / (1e-4 + (x - 0.3) ** 2)
        result = GaussKronrod(rtol=1e-12, max_intervals=2)(peak, 0.0, 1.0)
        assert not bool(result.converged)
        assert float(result.error) > 1e-12 * abs(float(result.value))
        assert int(result.n_evaluations) == 15 * 3

    def test_a_smooth_integrand_needs_one_interval(self) -> None:
        result = GaussKronrod(rtol=1e-10)(jnp.exp, 0.0, 1.0)
        assert bool(result.converged)
        assert int(result.n_evaluations) == 15
        assert_matches_scipy(result.value, math.e - 1.0, rtol=1e-14)

    def test_vector_valued_integrand(self) -> None:
        result = GaussKronrod(rtol=1e-10)(
            lambda x: jnp.stack([jnp.sin(x), x**2]), 0.0, math.pi
        )
        assert_matches_scipy(
            result.value, np.array([2.0, math.pi**3 / 3.0]), rtol=1e-10
        )

    def test_reversed_limits_flip_the_sign(self) -> None:
        forward = GaussKronrod()(jnp.cos, 0.0, 1.0).value
        backward = GaussKronrod()(jnp.cos, 1.0, 0.0).value
        assert_matches_scipy(backward, -forward, rtol=1e-15)

    def test_equal_limits_give_zero(self) -> None:
        result = GaussKronrod()(jnp.cos, 0.5, 0.5)
        assert float(result.value) == 0.0
        assert bool(result.converged)

    def test_float32_limits_give_a_float32_integral(self) -> None:
        result = GaussKronrod(rtol=1e-5)(jnp.exp, jnp.float32(0.0), jnp.float32(1.0))
        assert result.value.dtype == jnp.float32

    def test_is_reexported_from_the_adapt_module(self) -> None:
        assert ix.adapt.GaussKronrod is GaussKronrod


class TestGradients:
    @pytest.mark.parametrize(("name", "fn", "limits", "parameter"), _battery_params())
    def test_in_the_parameter(self, name, fn, limits, parameter) -> None:
        rule = GaussKronrod(rtol=1e-12, max_intervals=128)
        assert_grad_matches_fd(
            lambda p: rule(fn, *limits, args=p).value,
            jnp.asarray(parameter),
            eps=1e-6 * max(1.0, abs(parameter)),
            rtol=1e-6,
        )

    def test_in_the_limits(self) -> None:
        # dI/db = f(b), dI/da = −f(a)
        rule = GaussKronrod(rtol=1e-12)
        gradient = jax.grad(lambda limits: rule(jnp.cos, limits[0], limits[1]).value)(
            jnp.array([0.2, 1.1])
        )
        assert_matches_scipy(
            gradient, np.array([-math.cos(0.2), math.cos(1.1)]), rtol=1e-12
        )

    @pytest.mark.slow
    def test_in_a_closed_over_array(self) -> None:
        rule = GaussKronrod(rtol=1e-12)

        def integral(width: jax.Array) -> jax.Array:
            return rule(lambda x: 1.0 / (width**2 + (x - 0.3) ** 2), 0.0, 1.0).value

        assert_grad_matches_fd(integral, jnp.asarray(0.05), eps=1e-8, rtol=1e-6)

    def test_forward_mode_matches_reverse_mode(self) -> None:
        rule = GaussKronrod(rtol=1e-12)
        peak = lambda x, p: 1.0 / (p**2 + (x - 0.3) ** 2)
        reverse = jax.grad(lambda p: rule(peak, 0.0, 1.0, args=p).value)(0.05)
        forward = jax.jvp(
            lambda p: rule(peak, 0.0, 1.0, args=p).value, (0.05,), (1.0,)
        )[1]
        assert_matches_scipy(forward, reverse, rtol=1e-12)


class TestMemory:
    def test_does_not_keep_closed_over_arrays_alive(self) -> None:
        # jax.closure_convert caches globally on the function; a fresh
        # integrand per call must not stay alive (with its arrays) after it
        import gc
        import weakref

        class Holder:
            def __init__(self) -> None:
                self.field = jnp.ones(100)

            def __call__(self, x):
                return x * self.field[0]

        rule = GaussKronrod(rtol=1e-8)
        references = []
        for _ in range(3):
            holder = Holder()
            references.append(weakref.ref(holder))
            rule(holder, 0.0, 1.0)
            del holder
        gc.collect()
        assert all(reference() is None for reference in references)


class TestTransforms:
    def test_vmap_over_parameters_matches_a_loop(self) -> None:
        rule = GaussKronrod(rtol=1e-10)
        widths = jnp.array([0.02, 0.05, 0.2])
        peak = lambda x, p: 1.0 / (p**2 + (x - 0.3) ** 2)
        batched = jax.vmap(lambda p: rule(peak, 0.0, 1.0, args=p).value)(widths)
        looped = jnp.stack([rule(peak, 0.0, 1.0, args=p).value for p in widths])
        assert_matches_scipy(batched, looped, rtol=1e-12)

    def test_jit_matches_eager(self) -> None:
        rule = GaussKronrod(rtol=1e-10)
        eager = rule(jnp.sin, 0.0, 3.0).value
        traced = jax.jit(lambda b: rule(jnp.sin, 0.0, b).value)(3.0)
        assert_matches_scipy(traced, eager, rtol=1e-14)


class TestValidation:
    def test_rejects_an_order_without_a_table(self) -> None:
        with pytest.raises(ix.ValidationError, match="'order'"):
            GaussKronrod(order=17)

    @pytest.mark.parametrize("field", ["rtol", "atol"])
    def test_rejects_a_negative_tolerance(self, field: str) -> None:
        with pytest.raises(ix.ValidationError, match=f"'{field}'"):
            GaussKronrod(**{field: -1e-3})

    @pytest.mark.parametrize("max_intervals", [0, True])
    def test_rejects_an_empty_buffer(self, max_intervals) -> None:
        with pytest.raises((ix.ValidationError, TypeCheckError), match="max_intervals"):
            GaussKronrod(max_intervals=max_intervals)

    def test_rejects_an_infinite_limit(self) -> None:
        with pytest.raises(ix.ValidationError, match="tanh–sinh"):
            GaussKronrod()(jnp.exp, 0.0, math.inf)

    def test_rejects_an_array_limit(self) -> None:
        # In the suite, jaxtyping's import hook rejects the shape first.
        with pytest.raises((ix.ValidationError, TypeCheckError)):
            GaussKronrod()(jnp.exp, jnp.zeros(2), 1.0)
