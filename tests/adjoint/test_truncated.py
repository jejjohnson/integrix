"""Tests for `Truncated` and `truncated_scan` beyond pipekit-jax's own (#22).

Gate G2-values: forward values are bit-identical to an undifferentiated
`jax.lax.scan`. Gate G2-truncated: with k ≥ T the gradient equals the exact
one. Both are checked on a pytree carry, under jit and vmap.
"""

from __future__ import annotations

import diffrax as dfx
import jax
import jax.numpy as jnp
import pytest
from _harness import assert_matches_scipy
from jaxtyping import TypeCheckError

import integrix as ix
from integrix.adjoint import Truncated, to_diffrax_adjoint, truncated_scan


N_STEPS = 12


def _step(theta: jax.Array):
    """A pytree carry (position, momentum) with a mildly expanding map."""

    def f(carry, x):
        position, momentum = carry
        new_momentum = jnp.tanh(theta) * 1.4 * momentum - 0.3 * position + x
        new_position = position + 0.1 * new_momentum
        return (new_position, new_momentum), new_position

    return f


def _initial() -> tuple[jax.Array, jax.Array]:
    return jnp.array([0.3, -0.2]), jnp.array([0.1, 0.4])


def _forcing() -> jax.Array:
    return jnp.linspace(-1.0, 1.0, N_STEPS)[:, None] * jnp.ones((N_STEPS, 2))


class TestForwardValues:
    @pytest.mark.parametrize("k", [1, 3, N_STEPS, 2 * N_STEPS])
    def test_bit_identical_to_a_plain_scan(self, k: int) -> None:
        theta = jnp.asarray(0.7)
        plain = jax.lax.scan(_step(theta), _initial(), _forcing())
        truncated = truncated_scan(_step(theta), _initial(), _forcing(), k=k)
        for got, want in zip(
            jax.tree.leaves(truncated), jax.tree.leaves(plain), strict=True
        ):
            assert bool(jnp.array_equal(got, want))

    def test_bit_identical_under_jit(self) -> None:
        theta = jnp.asarray(0.7)
        plain = jax.jit(lambda t: jax.lax.scan(_step(t), _initial(), _forcing()))(theta)
        truncated = jax.jit(
            lambda t: truncated_scan(_step(t), _initial(), _forcing(), k=4)
        )(theta)
        for got, want in zip(
            jax.tree.leaves(truncated), jax.tree.leaves(plain), strict=True
        ):
            assert bool(jnp.array_equal(got, want))


class TestGradients:
    @staticmethod
    def _loss(scan):
        def loss(theta: jax.Array) -> jax.Array:
            _, ys = scan(_step(theta), _initial(), _forcing())
            return jnp.sum(ys[-1] ** 2)

        return loss

    @pytest.mark.parametrize("k", [N_STEPS, N_STEPS + 5])
    def test_k_at_least_t_is_the_exact_gradient(self, k: int) -> None:
        exact = jax.grad(self._loss(jax.lax.scan))(jnp.asarray(0.7))
        truncated = jax.grad(
            self._loss(lambda f, c, xs: truncated_scan(f, c, xs, k=k))
        )(jnp.asarray(0.7))
        assert_matches_scipy(truncated, exact, rtol=1e-14)

    def test_k_below_t_cuts_the_history(self) -> None:
        k = 3

        def manual(theta: jax.Array) -> jax.Array:
            carry = _initial()
            for x in _forcing()[: N_STEPS - k]:
                carry, _ = _step(theta)(carry, x)
            carry = jax.lax.stop_gradient(carry)
            for x in _forcing()[N_STEPS - k :]:
                carry, y = _step(theta)(carry, x)
            return jnp.sum(y**2)

        truncated = jax.grad(
            self._loss(lambda f, c, xs: truncated_scan(f, c, xs, k=k))
        )(jnp.asarray(0.7))
        assert_matches_scipy(truncated, jax.grad(manual)(jnp.asarray(0.7)), rtol=1e-12)

    def test_vmap_over_parameters(self) -> None:
        thetas = jnp.array([0.2, 0.5, 0.9])
        loss = self._loss(lambda f, c, xs: truncated_scan(f, c, xs, k=2))
        batched = jax.vmap(jax.grad(loss))(thetas)
        looped = jnp.stack([jax.grad(loss)(theta) for theta in thetas])
        assert_matches_scipy(batched, looped, rtol=1e-12)


class TestLength:
    def test_runs_without_stacked_inputs(self) -> None:
        carry, ys = truncated_scan(lambda c, _: (0.5 * c, c), 1.0, None, k=2, length=4)
        assert_matches_scipy(carry, 0.5**4, rtol=1e-15)
        assert ys.shape == (4,)

    def test_needs_inputs_or_a_length(self) -> None:
        with pytest.raises(ValueError, match="length"):
            truncated_scan(lambda c, _: (c, c), 1.0, None, k=2)


class TestTruncated:
    def test_defaults_to_one_step(self) -> None:
        assert Truncated().k == 1

    @pytest.mark.parametrize("k", [0, -2, True])
    def test_rejects_a_k_that_is_not_a_positive_integer(self, k) -> None:
        with pytest.raises(ix.ValidationError, match="'k'"):
            Truncated(k=k)

    def test_rejects_a_fractional_k(self) -> None:
        # In the suite, jaxtyping's import hook rejects 1.5 against `int` first.
        with pytest.raises((ix.ValidationError, TypeCheckError)):
            Truncated(k=1.5)

    def test_is_rejected_for_a_single_diffrax_solve(self) -> None:
        with pytest.raises(ix.ValidationError, match="outer rollout"):
            to_diffrax_adjoint(Truncated(k=2))


class TestToDiffraxAdjoint:
    def test_checkpoints_default_to_diffrax_s_choice(self) -> None:
        class RecursiveCheckpointAdjoint:
            pass

        adjoint = to_diffrax_adjoint(RecursiveCheckpointAdjoint())
        assert isinstance(adjoint, dfx.RecursiveCheckpointAdjoint)
        assert adjoint.checkpoints is None

    def test_is_reexported_from_the_adjoint_module(self) -> None:
        assert ix.adjoint.to_diffrax_adjoint is to_diffrax_adjoint
