"""Tests for `SampledPath` and `imex_stepsize_controller(step_ts=…)`."""

from __future__ import annotations

import inspect

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_traces
from functax.exceptions import GeometryError
from functax.interp import (
    Akima,
    BSpline,
    Conservative,
    Cubic,
    CubicSpline,
    Linear,
    Nearest,
    Pchip,
    Policies,
    Steffen,
)

import integrix.march as im
from integrix.exceptions import ValidationError


EVEN = jnp.linspace(0.0, 2.0, 11)
UNEVEN = jnp.array([0.0, 0.1, 0.4, 1.0, 1.1, 1.7, 2.0])
CONTINUOUS = [
    Linear(),
    Cubic(),
    BSpline(3),
    CubicSpline(),
    Pchip(),
    Akima(),
    Steffen(),
]
IDS = ["linear", "cubic", "bspline3", "spline", "pchip", "akima", "steffen"]


def _samples(ts):
    return jnp.stack([jnp.sin(ts), jnp.cos(2.0 * ts)], axis=-1)


class TestRead:
    @pytest.mark.parametrize("ts", [EVEN, UNEVEN], ids=["even", "uneven"])
    def test_linear_equals_diffrax(self, ts):
        # (N_t, 2, 3): two channel dims, read inside the samples and past them
        us = jnp.asarray(np.random.default_rng(0).normal(size=(ts.size, 2, 3)))
        policies = Policies(outside="extrapolate")
        path = im.SampledPath.fit(Linear(), ts, us, policies=policies)
        reference = dfx.LinearInterpolation(ts=ts, ys=us)
        times = jnp.linspace(-0.5, 2.5, 61)
        read = jax.vmap(path.evaluate)(times)
        assert read.shape == (61, 2, 3)
        assert jnp.allclose(read, jax.vmap(reference.evaluate)(times), atol=1e-12)
        increment = path.evaluate(0.3, 1.9)
        assert jnp.allclose(increment, reference.evaluate(0.3, 1.9), atol=1e-12)
        assert path.t0 == reference.t0 and path.t1 == reference.t1

    @pytest.mark.parametrize("left", [True, False], ids=["left", "right"])
    @pytest.mark.parametrize("ts", [EVEN, UNEVEN], ids=["even", "uneven"])
    def test_one_sided_derivatives_equal_diffrax(self, ts, left):
        us = _samples(ts)
        path = im.SampledPath.fit(Linear(), ts, us)
        reference = dfx.LinearInterpolation(ts=ts, ys=us)
        # every sample time, the ends included, and the cell middles
        times = jnp.concatenate([ts, 0.5 * (ts[1:] + ts[:-1])])
        expected = jax.vmap(lambda t: reference.derivative(t, left=left))(times)
        read = jax.vmap(lambda t: path.derivative(t, left=left))(times)
        assert jnp.allclose(read, expected, rtol=1e-11, atol=1e-11)
        # with the path traced, as diffeqsolve sees a path in its terms
        traced = eqx.filter_jit(
            lambda p, t: jax.vmap(lambda s: p.derivative(s, left=left))(t)
        )(path, times)
        assert jnp.allclose(traced, expected, rtol=1e-11, atol=1e-11)

    @pytest.mark.parametrize("scheme", CONTINUOUS, ids=IDS)
    def test_passes_through_the_samples(self, scheme):
        path = im.SampledPath.fit(scheme, EVEN, _samples(EVEN))
        read = jax.vmap(path.evaluate)(EVEN)
        assert jnp.allclose(read, _samples(EVEN), atol=1e-12)
        assert jnp.array_equal(path.ts, EVEN)

    @pytest.mark.parametrize("scheme", CONTINUOUS, ids=IDS)
    def test_derivative_is_that_of_evaluate(self, scheme):
        path = im.SampledPath.fit(scheme, EVEN, _samples(EVEN))
        for t in (0.07, 0.93, 1.71):
            _, slope = jax.jvp(path.evaluate, (t,), (1.0,))
            assert jnp.allclose(path.derivative(t), slope, rtol=1e-12, atol=1e-12)

    def test_a_rounding_error_past_the_end_is_held(self):
        path = im.SampledPath.fit(Pchip(), UNEVEN, _samples(UNEVEN))
        end = path.evaluate(path.t1 + 1e-12)
        assert jnp.allclose(end, _samples(UNEVEN)[-1], atol=1e-14)
        policies = Policies()
        nan = im.SampledPath.fit(Pchip(), UNEVEN, _samples(UNEVEN), policies=policies)
        assert bool(jnp.all(jnp.isnan(nan.evaluate(path.t1 + 1e-12))))

    def test_pchip_stays_between_its_samples(self):
        discharge = jnp.array([0.0, 0.0, 3.0, 0.1, 0.0, 0.0, 5.0])
        path = im.SampledPath.fit(Pchip(), UNEVEN, discharge)
        read = jax.vmap(path.evaluate)(jnp.linspace(0.0, 2.0, 1000))
        assert float(read.min()) >= 0.0 and float(read.max()) <= 5.0

    def test_scalar_samples_integer_times_and_float32(self):
        ts = jnp.arange(4.0, dtype=jnp.float32)
        path = im.SampledPath.fit(Linear(), ts, ts**2)
        assert path.evaluate(2) == 4.0
        assert path.evaluate(jnp.asarray(1.5, jnp.float64)).dtype == jnp.float32
        assert path.derivative(jnp.asarray(1.5, jnp.float64)).dtype == jnp.float32


class TestValidation:
    def test_rejects_bad_schemes_times_and_samples(self):
        us = _samples(EVEN)
        with pytest.raises(ValidationError, match="'scheme' must be"):
            im.SampledPath.fit("linear", EVEN, us)
        with pytest.raises(ValidationError, match="got Conservative"):
            im.SampledPath.fit(Conservative(), EVEN, us)
        with pytest.raises(ValidationError, match="at least two times, got 1"):
            im.SampledPath.fit(Linear(), EVEN[:1], us[:1])
        with pytest.raises(ValidationError, match=r"one sample per time.*\(5, 2\)"):
            # unwrapped: the type hook of the suite would refuse the shapes first
            inspect.unwrap(im.SampledPath.fit.__func__)(
                im.SampledPath, Linear(), EVEN, us[:5]
            )
        with pytest.raises(ValidationError, match="'ts' must increase"):
            im.SampledPath.fit(Linear(), EVEN[::-1], us)
        with pytest.raises(GeometryError, match="strictly monotone"):
            im.SampledPath.fit(Linear(), EVEN.at[3].set(5.0), us)

    def test_traced_decreasing_times_are_caught(self):
        fit = eqx.filter_jit(lambda ts: im.SampledPath.fit(Linear(), ts, ts))
        with pytest.raises(Exception, match="'ts' must increase"):
            fit(EVEN[::-1])

    @pytest.mark.parametrize("scheme", [Cubic(), BSpline(3)], ids=["cubic", "b3"])
    def test_uneven_times_are_refused_by_index_space_schemes(self, scheme):
        us = _samples(UNEVEN)
        with pytest.raises(GeometryError, match="CubicSpline or Pchip"):
            im.SampledPath.fit(scheme, UNEVEN, us)
        policies = Policies(spacing="index")
        im.SampledPath.fit(scheme, UNEVEN, us, policies=policies)

    def test_concrete_times_are_checked_under_jit(self):
        with pytest.raises(GeometryError, match="CubicSpline or Pchip"):
            jax.jit(lambda us: im.SampledPath.fit(Cubic(), UNEVEN, us).evaluate(0.5))(
                _samples(UNEVEN)
            )

    def test_nearest_is_accepted(self):
        path = im.SampledPath.fit(Nearest(), UNEVEN, _samples(UNEVEN))
        assert jnp.array_equal(path.evaluate(0.11), _samples(UNEVEN)[1])


class TestInASolve:
    @pytest.mark.parametrize("scheme", [Linear(), Pchip()], ids=["linear", "pchip"])
    def test_a_cde_returns_the_increment_of_its_control(self, scheme):
        # dx = I dX(t) ⇒ x(t₁) − x(t₀) = X̂(t₁) − X̂(t₀)
        control = im.SampledPath.fit(scheme, UNEVEN, _samples(UNEVEN))
        term = dfx.ControlTerm(lambda t, x, args: jnp.eye(2), control)
        solution = dfx.diffeqsolve(term, im.RK3SSP(), 0.0, 2.0, 0.05, jnp.zeros(2))
        expected = control.evaluate(0.0, 2.0)
        assert jnp.allclose(solution.ys[-1], expected, atol=1e-12)

    @pytest.mark.parametrize("scheme", [Linear(), Pchip()], ids=["linear", "pchip"])
    def test_gradient_in_the_samples_through_a_solve(self, scheme):
        def final(us):
            forcing = im.SampledPath.fit(scheme, EVEN, us)
            term = dfx.ODETerm(lambda t, x, args: -x + forcing.evaluate(t))
            solution = dfx.diffeqsolve(term, im.RK3SSP(), 0.0, 2.0, 0.1, jnp.zeros(2))
            return solution.ys[-1]

        assert_grad_matches_fd(final, _samples(EVEN))

    def test_traces_in_the_time(self):
        path = im.SampledPath.fit(CubicSpline(), UNEVEN, _samples(UNEVEN))
        assert_traces(lambda t: path.evaluate(t), jnp.linspace(0.05, 1.9, 5))

    def test_rk4_is_exact_on_the_knots_and_second_order_across_them(self):
        # ẋ = û(t), with û linear between samples of sin t every ½: RK4 integrates
        # each linear piece exactly, and loses that when a step straddles a knot
        ts = jnp.linspace(0.0, 4.0, 9)
        forcing = im.SampledPath.fit(Linear(), ts, jnp.sin(ts))
        exact = jnp.trapezoid(jnp.sin(ts), ts)

        def error(n_steps):
            solution = dfx.diffeqsolve(
                dfx.ODETerm(lambda t, x, args: forcing.evaluate(t)),
                im.RK4Classic(),
                0.0,
                4.0,
                4.0 / n_steps,
                jnp.zeros(()),
            )
            return abs(float(solution.ys[-1] - exact))

        assert max(error(n) for n in (8, 16, 32, 64)) <= 1e-12
        errors = [error(n) for n in (15, 31, 63)]
        steps = [4.0 / n for n in (15, 31, 63)]
        slope = np.polyfit(np.log(steps), np.log(errors), 1)[0]
        assert 1.9 <= slope <= 2.1


class TestStepControl:
    def test_step_ts_lands_on_the_samples(self):
        forcing = im.SampledPath.fit(Linear(), UNEVEN, _samples(UNEVEN))
        controller = im.imex_stepsize_controller(
            rtol=1e-3, atol=1e-5, step_ts=forcing.ts
        )
        assert isinstance(controller, dfx.ClipStepSizeController)
        assert controller.rtol == 1e-3 and controller.atol == 1e-5
        solution = dfx.diffeqsolve(
            dfx.ODETerm(lambda t, x, args: forcing.evaluate(t)),
            im.RK2Heun(),
            0.0,
            2.0,
            0.3,
            jnp.zeros(2),
            stepsize_controller=controller,
            saveat=dfx.SaveAt(steps=True),
        )
        times = solution.ts[jnp.isfinite(solution.ts)]
        nearest = jnp.min(jnp.abs(times[:, None] - UNEVEN[None, 1:]), axis=0)
        assert jnp.array_equal(nearest, jnp.zeros(UNEVEN.size - 1))

    def test_without_step_ts_it_is_the_plain_controller(self):
        controller = im.imex_stepsize_controller(rtol=1e-3, atol=1e-5)
        assert type(controller) is dfx.PIDController
