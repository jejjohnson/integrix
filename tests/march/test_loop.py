"""Tests for `AbstractStepper`, `run` and the history-carrying steppers (#20).

The AB2, leapfrog-RAF and split-explicit steppers are checked against hand
loops over finitevolX's pure step functions (ported unchanged), so the new
`rhs(t, y, args)` interface with history in the loop carry computes what
finitevolX's module-held history did. Gate G1-order for AB2, the leapfrog-RAF
oscillator bound over 10⁴ steps, `run` under jit and vmap, and gradients
through it against finite differences complete #20's definition of done.
"""

from __future__ import annotations

import math

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_matches_scipy, assert_order
from jaxtyping import TypeCheckError
from scipy.linalg import expm

import integrix as ix
from integrix.march import (
    IMEX_SSP2,
    RK3SSP,
    AB2Solver,
    AbstractStepper,
    LeapfrogRAFSolver,
    SplitExplicitRKSolver,
    SplitExplicitTerms,
    ab2_step,
    heun_step,
    leapfrog_raf_step,
    run,
    split_explicit_step,
)


def _decay(t, y, args):
    """y′ = −y."""
    return -y


def _oscillator(t, y, args):
    """(q, p)′ = (p, −ω² q), with ω = args."""
    position, momentum = y
    return momentum, -(args**2) * position


class TestRunDispatchesToDiffrax:
    def test_final_state_equals_diffeqsolve(self) -> None:
        y = run(RK3SSP(), _decay, 0.0, jnp.array(1.0), 0.1, 10)
        solution = dfx.diffeqsolve(
            dfx.ODETerm(_decay), RK3SSP(), 0.0, 1.0, 0.1, jnp.array(1.0)
        )
        assert_matches_scipy(y, solution.ys[0], rtol=1e-14)

    def test_every_step_is_saved(self) -> None:
        every = run(
            RK3SSP(), _decay, 0.0, jnp.ones(2), 0.1, 7, saveat=dfx.SaveAt(steps=True)
        )
        assert every.shape == (7, 2)
        final = run(RK3SSP(), _decay, 0.0, jnp.ones(2), 0.1, 7)
        assert_matches_scipy(every[-1], final, rtol=1e-15)

    def test_takes_a_diffrax_term_for_imex(self) -> None:
        terms = dfx.MultiTerm(
            dfx.ODETerm(lambda t, y, args: -y), dfx.ODETerm(lambda t, y, args: -10 * y)
        )
        solver = IMEX_SSP2(root_finder=dfx.VeryChord(rtol=1e-12, atol=1e-12))
        y = run(solver, terms, 0.0, jnp.array(1.0), 0.01, 100)
        solution = dfx.diffeqsolve(terms, solver, 0.0, 1.0, 0.01, jnp.array(1.0))
        assert_matches_scipy(y, solution.ys[0], rtol=1e-12)

    def test_passes_a_diffrax_adjoint_through(self) -> None:
        def final(y0: jax.Array) -> jax.Array:
            return run(
                RK3SSP(),
                _decay,
                0.0,
                y0,
                0.1,
                10,
                adjoint=dfx.RecursiveCheckpointAdjoint(),
            )

        assert_grad_matches_fd(final, jnp.array([1.0, 2.0]))


class TestAB2:
    def test_matches_heun_then_ab2_steps(self) -> None:
        dt, n_steps = 0.05, 12
        y = run(AB2Solver(), _decay, 0.0, jnp.array(1.0), dt, n_steps)
        f = lambda state: -state
        reference = heun_step(jnp.array(1.0), f, dt)
        history = f(jnp.array(1.0))
        for _ in range(n_steps - 1):
            reference, history, _ = ab2_step(reference, f, dt, history)
        assert_matches_scipy(y, reference, rtol=1e-14)

    @pytest.mark.slow
    def test_order_on_decay_and_advection(self) -> None:
        steps = (0.04, 0.02, 0.01, 0.005)
        decay = {
            dt: abs(
                float(run(AB2Solver(), _decay, 0.0, jnp.array(1.0), dt, round(1 / dt)))
                - math.exp(-1.0)
            )
            for dt in steps
        }
        assert_order(decay, 2)
        n_points = 32
        spacing = 1.0 / n_points
        shift = np.roll(np.eye(n_points), 1, axis=1) - np.roll(
            np.eye(n_points), -1, axis=1
        )
        # A = −D, the periodic centred difference for unit speed
        operator = -shift / (2.0 * spacing)
        initial = np.sin(2.0 * math.pi * np.arange(n_points) * spacing)
        exact = expm(operator) @ initial
        matrix = jnp.asarray(operator)
        advection = {
            dt: float(
                np.max(
                    np.abs(
                        np.asarray(
                            run(
                                AB2Solver(),
                                lambda t, y, a: matrix @ y,
                                0.0,
                                jnp.asarray(initial),
                                dt,
                                round(1 / dt),
                            )
                        )
                        - exact
                    )
                )
            )
            for dt in steps
        }
        assert_order(advection, 2)

    def test_float32_state_stays_float32_with_a_time_dependent_field(self) -> None:
        y = run(
            AB2Solver(),
            lambda t, y, args: -y * jnp.cos(t),
            0.0,
            jnp.ones(3, dtype=jnp.float32),
            0.1,
            5,
        )
        assert y.dtype == jnp.float32


class TestLeapfrogRAF:
    def test_matches_heun_then_leapfrog_steps(self) -> None:
        dt, n_steps, alpha = 0.05, 12, 0.07
        y = run(
            LeapfrogRAFSolver(alpha=alpha), _decay, 0.0, jnp.array(1.0), dt, n_steps
        )
        f = lambda state: -state
        previous = jnp.array(1.0)
        current = heun_step(previous, f, dt)
        for _ in range(n_steps - 1):
            current, previous = leapfrog_raf_step(current, previous, f, dt, alpha=alpha)
        assert_matches_scipy(y, current, rtol=1e-14)

    @pytest.mark.slow
    @pytest.mark.parametrize(("alpha", "order"), [(0.0, 2), (0.05, 1)])
    def test_order(self, alpha: float, order: int) -> None:
        # y′ = −1.5y: second order unfiltered, first order with the filter
        stepper = LeapfrogRAFSolver(alpha=alpha)
        steps = (0.004, 0.002, 0.001, 0.0005) if alpha else (0.02, 0.01, 0.005, 0.0025)
        errors = {
            dt: abs(
                float(
                    run(
                        stepper,
                        lambda t, y, a: -1.5 * y,
                        0.0,
                        jnp.array(1.0),
                        dt,
                        round(1 / dt),
                    )
                )
                - math.exp(-1.5)
            )
            for dt in steps
        }
        assert_order(errors, order, slack=0.1)
        assert stepper.order == order

    def test_computational_mode_stays_bounded_over_ten_thousand_steps(self) -> None:
        # ωh = 0.5 on the oscillator: the amplitude must neither grow nor leave
        # the filter's slow damping
        positions, momenta = run(
            LeapfrogRAFSolver(alpha=0.05),
            _oscillator,
            0.0,
            (jnp.array(1.0), jnp.array(0.0)),
            0.05,
            10_000,
            args=10.0,
            saveat=dfx.SaveAt(steps=True),
        )
        # E = (p² + ω² q²) / 2ω², 1/2 at the start
        energy = 0.5 * (momenta**2 / 100.0 + positions**2)
        assert bool(jnp.all(jnp.isfinite(energy)))
        assert float(jnp.max(energy)) < 0.5 * 1.05
        assert float(energy[-1]) > 0.0

    @pytest.mark.parametrize("alpha", [-0.1, float("nan")])
    def test_rejects_a_negative_or_nan_alpha(self, alpha: float) -> None:
        with pytest.raises(ix.ValidationError, match="'alpha'"):
            LeapfrogRAFSolver(alpha=alpha)

    def test_alpha_can_be_traced_and_differentiated(self) -> None:
        def final(alpha: jax.Array) -> jax.Array:
            return run(
                LeapfrogRAFSolver(alpha=alpha), _decay, 0.0, jnp.array(1.0), 0.05, 20
            )

        traced = jax.jit(final)(jnp.array(0.05))
        assert_matches_scipy(traced, final(0.05), rtol=1e-14)
        assert_grad_matches_fd(final, jnp.array(0.05))
        assert jax.vmap(final)(jnp.array([0.0, 0.05, 0.1])).shape == (3,)


class TestSplitExplicit:
    @staticmethod
    def _terms() -> SplitExplicitTerms:
        return SplitExplicitTerms(
            slow=lambda t, y3, mean_2d, args: -args * y3 + mean_2d,
            fast=lambda t, y2, y3, args: -20.0 * y2 * jnp.cos(t) + y3,
            slow_forcing=lambda t, y3, args: 0.1 * y3,
            couple=lambda y3, mean_2d, args: y3 + 0.01 * mean_2d,
        )

    def test_matches_split_explicit_steps(self) -> None:
        terms = self._terms()
        dt, n_steps, rate = 0.1, 6, 0.5
        y3, y2 = run(
            SplitExplicitRKSolver(n_substeps=8),
            terms,
            0.3,
            (jnp.array(1.0), jnp.array(0.5)),
            dt,
            n_steps,
            args=rate,
        )
        ref3, ref2 = jnp.array(1.0), jnp.array(0.5)
        for n in range(n_steps):
            t = 0.3 + n * dt
            ref3, ref2 = split_explicit_step(
                ref3,
                ref2,
                lambda a, m, t=t: terms.slow(t, a, m, rate),
                lambda s, b, a: terms.fast(s, b, a, rate),
                lambda a, m: terms.couple(a, m, rate),
                dt,
                8,
                slow_forcing=lambda a, t=t: terms.slow_forcing(t, a, rate),
                t0=t,
            )
        assert_matches_scipy(y3, ref3, rtol=1e-13)
        assert_matches_scipy(y2, ref2, rtol=1e-13)

    def test_float32_states_stay_float32(self) -> None:
        y3, y2 = run(
            SplitExplicitRKSolver(n_substeps=5),
            self._terms(),
            0.0,
            (jnp.ones(3, jnp.float32), jnp.ones(3, jnp.float32)),
            0.1,
            3,
            args=0.5,
        )
        assert y3.dtype == jnp.float32
        assert y2.dtype == jnp.float32

    def test_rejects_a_plain_vector_field(self) -> None:
        with pytest.raises(ix.ValidationError, match="SplitExplicitTerms"):
            run(SplitExplicitRKSolver(), _decay, 0.0, (1.0, 1.0), 0.1, 2)

    @pytest.mark.parametrize("n_substeps", [0, 2.5, True])
    def test_rejects_a_substep_count_that_is_not_a_positive_integer(
        self, n_substeps
    ) -> None:
        # In the suite, jaxtyping's import hook rejects 2.5 against `int` first;
        # without it, __check_init__ raises ValidationError.
        with pytest.raises((ix.ValidationError, TypeCheckError)):
            SplitExplicitRKSolver(n_substeps=n_substeps)


class TestTime:
    def test_complex_states_get_real_time(self) -> None:
        seen = []

        def field(t, y, args):
            seen.append(t.dtype)
            return -y * jnp.cos(jnp.floor(t))

        run(AB2Solver(), field, 0.0, jnp.ones(2, dtype=jnp.complex64), 0.1, 3)
        assert all(dtype == jnp.float32 for dtype in seen)

    def test_the_diffrax_path_keeps_float32_with_a_time_dependent_field(self) -> None:
        y = run(
            RK3SSP(),
            lambda t, y, args: -y + jnp.cos(t),
            0.0,
            jnp.ones(2, dtype=jnp.float32),
            0.1,
            5,
        )
        assert y.dtype == jnp.float32


class TestTransforms:
    @pytest.mark.parametrize(
        "stepper",
        [AB2Solver(), LeapfrogRAFSolver(alpha=0.05), RK3SSP()],
        ids=["ab2", "leapfrog_raf", "rk3_ssp"],
    )
    def test_jit_and_vmap_agree_with_a_loop(self, stepper) -> None:
        initial = jnp.linspace(0.5, 2.0, 4)

        def final(y0: jax.Array) -> jax.Array:
            return run(stepper, _oscillator, 0.0, (y0, 0.0 * y0), 0.05, 20, args=2.0)[0]

        eager = jnp.stack([final(y0) for y0 in initial])
        assert_matches_scipy(jax.jit(jax.vmap(final))(initial), eager, rtol=1e-12)
        assert_matches_scipy(jax.vmap(final)(initial), eager, rtol=1e-12)

    @pytest.mark.parametrize(
        "stepper",
        [AB2Solver(), LeapfrogRAFSolver(alpha=0.05)],
        ids=["ab2", "leapfrog_raf"],
    )
    def test_gradient_through_run_matches_finite_differences(self, stepper) -> None:
        def final(y0: jax.Array) -> jax.Array:
            return run(
                stepper, lambda t, y, a: -y * (1.0 + jnp.sin(t)), 0.0, y0, 0.05, 30
            )

        assert_grad_matches_fd(final, jnp.array([0.7, 1.3]))

    def test_gradient_through_split_explicit(self) -> None:
        terms = TestSplitExplicit._terms()

        def final(y0: jax.Array) -> jax.Array:
            y3, y2 = run(
                SplitExplicitRKSolver(n_substeps=4),
                terms,
                0.0,
                (y0, y0),
                0.1,
                5,
                args=0.5,
            )
            return y3 + y2

        assert_grad_matches_fd(final, jnp.array([0.7, 1.3]))


class TestProtocol:
    def test_a_custom_stepper_runs(self) -> None:
        class Midpoint(AbstractStepper):
            @property
            def order(self) -> int:
                return 2

            def init(self, rhs, t0, y0, dt, args):
                return None, self.step(rhs, t0, y0, dt, None, args)[0]

            def step(self, rhs, t, y, dt, state, args):
                half = y + 0.5 * dt * rhs(t, y, args)
                return y + dt * rhs(t + 0.5 * dt, half, args), state

        y = run(Midpoint(), _decay, 0.0, jnp.array(1.0), 0.01, 100)
        assert_matches_scipy(y, math.exp(-1.0), rtol=1e-4)

    def test_a_stepper_without_step_cannot_be_built(self) -> None:
        class Incomplete(AbstractStepper):
            @property
            def order(self) -> int:
                return 1

        with pytest.raises(TypeError):
            Incomplete()

    def test_a_single_step_is_the_init_step(self) -> None:
        every = run(
            AB2Solver(),
            _decay,
            0.0,
            jnp.array(1.0),
            0.1,
            1,
            saveat=dfx.SaveAt(steps=True),
        )
        assert every.shape == (1,)
        assert_matches_scipy(
            every[0], heun_step(jnp.array(1.0), lambda y: -y, 0.1), rtol=1e-15
        )


class TestValidation:
    @pytest.mark.parametrize("n_steps", [0, -3])
    def test_rejects_no_steps(self, n_steps: int) -> None:
        with pytest.raises(ix.ValidationError, match="'n_steps' must be at least 1"):
            run(AB2Solver(), _decay, 0.0, jnp.array(1.0), 0.1, n_steps)

    def test_rejects_a_traced_step_count(self) -> None:
        with pytest.raises((ix.ValidationError, TypeError)):
            run(AB2Solver(), _decay, 0.0, jnp.array(1.0), 0.1, 2.5)

    @pytest.mark.parametrize(
        "saveat",
        [
            dfx.SaveAt(steps=True, fn=lambda t, y, args: 100 * y),
            dfx.SaveAt(t1=True, dense=True),
            dfx.SaveAt(t1=True, solver_state=True),
        ],
        ids=["fn", "dense", "solver_state"],
    )
    @pytest.mark.parametrize("stepper", [AB2Solver(), RK3SSP()], ids=["ab2", "rk3"])
    def test_rejects_saveat_options_it_would_ignore(self, saveat, stepper) -> None:
        with pytest.raises(ix.ValidationError, match="'saveat'"):
            run(stepper, _decay, 0.0, jnp.array(1.0), 0.1, 3, saveat=saveat)

    def test_rejects_saving_at_times(self) -> None:
        with pytest.raises(ix.ValidationError, match="'saveat'"):
            run(
                AB2Solver(),
                _decay,
                0.0,
                jnp.array(1.0),
                0.1,
                3,
                saveat=dfx.SaveAt(ts=jnp.array([0.1])),
            )

    def test_rejects_an_adjoint_for_an_abstract_stepper(self) -> None:
        with pytest.raises(ix.ValidationError, match="'adjoint'"):
            run(
                AB2Solver(),
                _decay,
                0.0,
                jnp.array(1.0),
                0.1,
                3,
                adjoint=dfx.RecursiveCheckpointAdjoint(),
            )

    def test_rejects_something_that_is_not_a_stepper(self) -> None:
        with pytest.raises((ix.ValidationError, Exception), match="stepper"):
            run(object(), _decay, 0.0, jnp.array(1.0), 0.1, 3)
