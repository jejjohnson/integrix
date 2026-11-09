"""Tests for `Checkpointed`, `checkpointed_scan` and the adjoints of `run` (#23).

Gate G2-values: forward values are bit-identical to an undifferentiated run.
Gradients equal plain reverse mode to 1e-12, and the states reverse mode
stores fall from T to ⌈T/k⌉ (√T at k = √T), counted with JAX's
``saved_residuals``, not by wall-clock.
"""

from __future__ import annotations

import math

import diffrax as dfx
import jax
import jax.numpy as jnp
import pytest
from _harness import assert_grad_matches_fd, assert_matches_scipy
from _rules import trapezoid_rule
from jax._src.ad_checkpoint import saved_residuals
from jaxtyping import TypeCheckError

import integrix as ix
from integrix.adjoint import Checkpointed, Truncated, checkpointed_scan
from integrix.march import RK3SSP, AB2Solver, LeapfrogRAFSolver, run
from integrix.scan import cumulative


def _body(carry, x):
    """A pytree carry with a nonlinear, input-driven map."""
    position, momentum = carry
    new_momentum = jnp.tanh(momentum) - 0.3 * jnp.sin(position) + x
    new_position = position + 0.1 * new_momentum
    return (new_position, new_momentum), new_position


def _initial() -> tuple[jax.Array, jax.Array]:
    return jnp.array([0.3, -0.2]), jnp.array([0.1, 0.4])


def _inputs(n_steps: int) -> jax.Array:
    return jnp.linspace(-1.0, 1.0, n_steps)[:, None] * jnp.ones((n_steps, 2))


def _assert_bit_identical(actual, expected) -> None:
    for got, want in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        assert bool(jnp.array_equal(got, want))


class TestCheckpointedScan:
    @pytest.mark.parametrize("every", [1, 3, 5, 15, 40])
    def test_forward_values_are_bit_identical(self, every: int) -> None:
        # 15 steps: whole blocks, a remainder, one block, and every > T
        xs = _inputs(15)
        _assert_bit_identical(
            checkpointed_scan(_body, _initial(), xs, every=every),
            jax.lax.scan(_body, _initial(), xs),
        )

    @pytest.mark.parametrize("every", [1, 4, 15])
    def test_gradients_equal_plain_reverse_mode(self, every: int) -> None:
        xs = _inputs(15)

        def loss(scan):
            def value(initial):
                _, ys = scan(_body, initial, xs)
                return jnp.sum(ys**2)

            return value

        plain = jax.grad(loss(jax.lax.scan))(_initial())
        checkpointed = jax.grad(
            loss(lambda f, c, x: checkpointed_scan(f, c, x, every=every))
        )(_initial())
        for got, want in zip(
            jax.tree.leaves(checkpointed), jax.tree.leaves(plain), strict=True
        ):
            assert_matches_scipy(got, want, rtol=1e-12, atol=1e-14)

    def test_runs_without_stacked_inputs(self) -> None:
        carry, ys = checkpointed_scan(
            lambda c, _: (0.5 * c, None), 1.0, None, every=3, length=7
        )
        assert_matches_scipy(carry, 0.5**7, rtol=1e-15)
        assert ys is None

    @pytest.mark.parametrize("n_steps", [100, 400])
    def test_stores_one_state_per_block(self, n_steps: int) -> None:
        every = math.isqrt(n_steps)
        state = jnp.ones(8)

        def body(carry, _):
            return jnp.sin(carry) * 1.01, None

        def stored_states(scan) -> float:
            def value(initial):
                return jnp.sum(scan(body, initial, None)[0])

            residuals = saved_residuals(value, state)
            return sum(math.prod(aval.shape) for aval, _ in residuals) / state.size

        plain = stored_states(lambda f, c, x: jax.lax.scan(f, c, x, length=n_steps))
        checkpointed = stored_states(
            lambda f, c, x: checkpointed_scan(f, c, x, every=every, length=n_steps)
        )
        # a plain scan keeps all T states; checkpointing keeps ⌈T/k⌉ = √T, and
        # recomputes the k states of one block at a time on the way back
        assert plain == n_steps
        assert checkpointed == math.ceil(n_steps / every) == every

    def test_needs_inputs_or_a_length(self) -> None:
        with pytest.raises(ix.ValidationError, match="length"):
            checkpointed_scan(lambda c, _: (c, c), 1.0, None, every=2)

    def test_zero_steps_is_the_empty_scan(self) -> None:
        carry, ys = checkpointed_scan(_body, _initial(), _inputs(0), every=3)
        _assert_bit_identical(carry, _initial())
        assert ys.shape == (0, 2)

    def test_rejects_a_length_that_disagrees_with_the_inputs(self) -> None:
        with pytest.raises(ix.ValidationError, match="length=5"):
            checkpointed_scan(_body, _initial(), _inputs(10), every=3, length=5)

    def test_rejects_an_input_without_a_step_axis(self) -> None:
        with pytest.raises(ix.ValidationError, match="leading"):
            checkpointed_scan(lambda c, x: (c, c), 0.0, jnp.array(1.0), every=1)


class TestCheckpointed:
    @pytest.mark.parametrize("every", [0, -1, True])
    def test_rejects_an_every_that_is_not_a_positive_integer(self, every) -> None:
        with pytest.raises(ix.ValidationError, match="'every'"):
            Checkpointed(every=every)

    def test_rejects_a_fractional_every(self) -> None:
        # In the suite, jaxtyping's import hook rejects 2.5 against `int` first.
        with pytest.raises((ix.ValidationError, TypeCheckError)):
            Checkpointed(every=2.5)


def _decay(t, y, args):
    """y′ = −y (1 + sin t)."""
    return -y * (1.0 + jnp.sin(t))


N_STEPS = 24


class TestRunWithAnAdjoint:
    @pytest.mark.parametrize(
        "stepper",
        [
            AB2Solver(),
            LeapfrogRAFSolver(alpha=0.05),
            # diffrax compiles for 1–4 s here; the fast tier keeps the integrix
            # loop and test_checkpointed_maps_to_diffrax_checkpoints
            pytest.param(RK3SSP(), marks=pytest.mark.slow),
        ],
        ids=["ab2", "leapfrog", "rk3"],
    )
    @pytest.mark.parametrize(
        "adjoint",
        [Checkpointed(every=5), Checkpointed(every=1)],
        ids=["every5", "every1"],
    )
    def test_forward_values_are_bit_identical(self, stepper, adjoint) -> None:
        y0 = jnp.array([0.7, 1.3])
        _assert_bit_identical(
            run(stepper, _decay, 0.0, y0, 0.05, N_STEPS, adjoint=adjoint),
            run(stepper, _decay, 0.0, y0, 0.05, N_STEPS),
        )

    @pytest.mark.parametrize(
        "stepper",
        [
            AB2Solver(),
            LeapfrogRAFSolver(alpha=0.05),
            # diffrax compiles for 1–4 s here; the fast tier keeps the integrix
            # loop and test_checkpointed_maps_to_diffrax_checkpoints
            pytest.param(RK3SSP(), marks=pytest.mark.slow),
        ],
        ids=["ab2", "leapfrog", "rk3"],
    )
    def test_checkpointed_gradient_equals_plain_reverse_mode(self, stepper) -> None:
        def final(adjoint):
            return lambda y0: jnp.sum(
                run(stepper, _decay, 0.0, y0, 0.05, N_STEPS, adjoint=adjoint) ** 2
            )

        y0 = jnp.array([0.7, 1.3])
        plain = jax.grad(final(None))(y0)
        checkpointed = jax.grad(final(Checkpointed(every=5)))(y0)
        assert_matches_scipy(checkpointed, plain, rtol=1e-12)

    @pytest.mark.parametrize(
        "stepper", [AB2Solver(), LeapfrogRAFSolver(alpha=0.05)], ids=["ab2", "leapfrog"]
    )
    def test_truncated_forward_values_are_bit_identical(self, stepper) -> None:
        y0 = jnp.array([0.7, 1.3])
        _assert_bit_identical(
            run(stepper, _decay, 0.0, y0, 0.05, N_STEPS, adjoint=Truncated(k=3)),
            run(stepper, _decay, 0.0, y0, 0.05, N_STEPS),
        )

    @pytest.mark.parametrize("k", [N_STEPS, N_STEPS + 3])
    def test_truncated_with_k_at_least_n_is_the_exact_gradient(self, k: int) -> None:
        def final(adjoint):
            return lambda y0: jnp.sum(
                run(AB2Solver(), _decay, 0.0, y0, 0.05, N_STEPS, adjoint=adjoint)
            )

        y0 = jnp.array([0.7, 1.3])
        assert_matches_scipy(
            jax.grad(final(Truncated(k=k)))(y0), jax.grad(final(None))(y0), rtol=1e-14
        )

    @pytest.mark.parametrize("k", [1, 5, N_STEPS - 1])
    def test_truncated_gradient_flows_through_the_last_k_steps_only(
        self, k: int
    ) -> None:
        # A hand-written march that cuts the history before the last k steps,
        # gradient taken in the vector field's rate
        stepper = AB2Solver()
        dt = 0.05
        y0 = jnp.array([0.7, 1.3])

        def field(rate):
            return lambda t, y, args: -rate * y * (1.0 + jnp.sin(t))

        def manual(rate):
            rhs = field(rate)
            state, y = stepper.init(rhs, 0.0, y0, dt, None)
            if N_STEPS - k >= 1:
                y, state = jax.lax.stop_gradient((y, state))
            for n in range(1, N_STEPS):
                y, state = stepper.step(rhs, n * dt, y, dt, state, None)
                if n < N_STEPS - k:
                    y, state = jax.lax.stop_gradient((y, state))
            return jnp.sum(y)

        def truncated(rate):
            return jnp.sum(
                run(stepper, field(rate), 0.0, y0, dt, N_STEPS, adjoint=Truncated(k=k))
            )

        rate = jnp.asarray(1.2)
        assert_matches_scipy(
            jax.grad(truncated)(rate), jax.grad(manual)(rate), rtol=1e-12
        )

    def test_checkpointed_maps_to_diffrax_checkpoints(self) -> None:
        def final(adjoint):
            return lambda y0: jnp.sum(
                run(RK3SSP(), _decay, 0.0, y0, 0.05, N_STEPS, adjoint=adjoint)
            )

        y0 = jnp.array([0.7, 1.3])
        via_checkpointed = jax.grad(final(Checkpointed(every=5)))(y0)
        # ⌈24 / 5⌉ = 5 checkpoints
        via_diffrax = jax.grad(final(dfx.RecursiveCheckpointAdjoint(checkpoints=5)))(y0)
        assert_matches_scipy(via_checkpointed, via_diffrax, rtol=1e-14)

    def test_a_pipekit_spec_reaches_a_diffrax_solver(self) -> None:
        class RecursiveCheckpointAdjoint:
            checkpoints = 4

        y0 = jnp.array([0.7, 1.3])
        y = run(
            RK3SSP(),
            _decay,
            0.0,
            y0,
            0.05,
            N_STEPS,
            adjoint=RecursiveCheckpointAdjoint(),
        )
        _assert_bit_identical(y, run(RK3SSP(), _decay, 0.0, y0, 0.05, N_STEPS))

    def test_a_single_step_runs_under_every_adjoint(self) -> None:
        for adjoint in (Checkpointed(every=3), Truncated(k=1), None):
            every = run(
                AB2Solver(),
                _decay,
                0.0,
                jnp.ones(2),
                0.1,
                1,
                adjoint=adjoint,
                saveat=dfx.SaveAt(steps=True),
            )
            assert every.shape == (1, 2)

    def test_truncation_keeps_each_saved_state_s_own_gradient(self) -> None:
        # forward Euler as a stepper: dyₙ/da through its own step is h yₙ₋₁ even
        # when the carry into it is cut, so y₁'s entry is h y₀, not zero
        from integrix.march import AbstractStepper

        class Euler(AbstractStepper):
            @property
            def order(self) -> int:
                return 1

            def init(self, rhs, t0, y0, dt, args):
                return None, y0 + dt * rhs(t0, y0, args)

            def step(self, rhs, t, y, dt, state, args):
                return y + dt * rhs(t, y, args), state

        def saved(rate):
            return run(
                Euler(),
                lambda t, y, a: rate * y,
                0.0,
                jnp.array(1.0),
                0.1,
                6,
                adjoint=Truncated(k=2),
                saveat=dfx.SaveAt(steps=True),
            )

        jacobian = jax.jacrev(saved)(jnp.asarray(1.3))
        assert_matches_scipy(jacobian[0], 0.1, rtol=1e-14)
        assert bool(jnp.all(jacobian > 0.0))

    def test_rejects_truncated_for_a_diffrax_solver(self) -> None:
        with pytest.raises(ix.ValidationError, match="needs an AbstractStepper"):
            run(RK3SSP(), _decay, 0.0, jnp.ones(2), 0.05, 4, adjoint=Truncated(k=2))

    def test_rejects_a_diffrax_adjoint_for_an_abstract_stepper(self) -> None:
        with pytest.raises(ix.ValidationError, match="diffrax solver only"):
            run(
                AB2Solver(),
                _decay,
                0.0,
                jnp.ones(2),
                0.05,
                4,
                adjoint=dfx.DirectAdjoint(),
            )

    @pytest.mark.slow
    def test_gradient_through_a_checkpointed_run_matches_finite_differences(
        self,
    ) -> None:
        assert_grad_matches_fd(
            lambda y0: run(
                LeapfrogRAFSolver(alpha=0.05),
                _decay,
                0.0,
                y0,
                0.05,
                N_STEPS,
                adjoint=Checkpointed(every=5),
            ),
            jnp.array([0.7, 1.3]),
        )


class TestReverseScan:
    @pytest.mark.slow
    def test_the_adjoint_of_a_running_integral_is_its_transpose(self) -> None:
        # ⟨A f, c⟩ = ⟨f, Aᵀ c⟩, with Aᵀ from jax.linear_transpose and from vjp
        rule = trapezoid_rule(jnp.linspace(0.0, 2.0, 9) ** 1.5)
        values = jnp.cos(jnp.arange(9.0))
        cotangent = jnp.sin(jnp.arange(9.0))
        forward = cumulative(rule, values)
        (transpose,) = jax.linear_transpose(lambda f: cumulative(rule, f), values)(
            cotangent
        )
        _, vjp = jax.vjp(lambda f: cumulative(rule, f), values)
        (pulled_back,) = vjp(cotangent)
        assert_matches_scipy(
            jnp.vdot(forward, cotangent), jnp.vdot(values, transpose), rtol=1e-13
        )
        assert_matches_scipy(pulled_back, transpose, rtol=1e-13, atol=1e-15)
