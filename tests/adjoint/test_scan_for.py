"""Tests for `scan_for`, the scan of an adjoint for a loop of one's own."""

from __future__ import annotations

import diffrax as dfx
import jax
import jax.numpy as jnp
import pytest

import integrix as ix
from integrix.adjoint import (
    Checkpointed,
    Truncated,
    checkpointed_scan,
    scan_for,
    truncated_scan,
)


N_STEPS = 10


def _step(theta: jax.Array):
    """A pytree carry (position, momentum) with a mildly expanding map."""

    def f(carry, x):
        position, momentum = carry
        new_momentum = jnp.tanh(theta) * 1.4 * momentum - 0.3 * position + x
        new_position = position + 0.1 * new_momentum
        return (new_position, new_momentum), new_position

    return f


INITIAL = (jnp.array([0.3, -0.2]), jnp.array([0.1, 0.4]))
FORCING = jnp.linspace(-1.0, 1.0, N_STEPS)[:, None] * jnp.ones((N_STEPS, 2))
ADJOINTS = [None, Checkpointed(every=3), Truncated(k=2)]


def _loss(theta: jax.Array, scan) -> jax.Array:
    (position, _), ys = scan(_step(theta), INITIAL, FORCING)
    return jnp.sum(position**2) + jnp.sum(ys)


@pytest.mark.parametrize("adjoint", ADJOINTS, ids=str)
def test_forward_values_are_those_of_a_plain_scan(adjoint) -> None:
    carry, ys = scan_for(adjoint)(_step(0.7), INITIAL, FORCING)
    plain_carry, plain_ys = jax.lax.scan(_step(0.7), INITIAL, FORCING)
    assert jnp.array_equal(ys, plain_ys)
    for leaf, plain in zip(carry, plain_carry, strict=True):
        assert jnp.array_equal(leaf, plain)


def test_none_is_jax_lax_scan() -> None:
    assert scan_for(None) is jax.lax.scan


def test_checkpointed_is_the_plain_gradient() -> None:
    plain = jax.grad(_loss)(0.7, jax.lax.scan)
    found = jax.grad(_loss)(0.7, scan_for(Checkpointed(every=3)))
    direct = jax.grad(_loss)(0.7, lambda f, c, x: checkpointed_scan(f, c, x, every=3))
    assert jnp.allclose(found, plain, rtol=1e-12)
    assert jnp.array_equal(found, direct)


@pytest.mark.parametrize("k", [1, 4, N_STEPS])
def test_truncated_is_truncated_scan(k: int) -> None:
    found = jax.grad(_loss)(0.7, scan_for(Truncated(k=k)))
    direct = jax.grad(_loss)(0.7, lambda f, c, x: truncated_scan(f, c, x, k=k))
    assert jnp.array_equal(found, direct)


def test_truncated_cuts_the_gradient_of_the_initial_carry() -> None:
    def last(start: jax.Array) -> jax.Array:
        carry, _ = scan_for(Truncated(k=3))(_step(0.7), (start, INITIAL[1]), FORCING)
        return jnp.sum(carry[0])

    assert jnp.array_equal(jax.grad(last)(INITIAL[0]), jnp.zeros(2))


@pytest.mark.parametrize(
    "adjoint",
    [dfx.RecursiveCheckpointAdjoint(), dfx.ForwardMode(), Truncated, "checkpointed"],
    ids=["diffrax adjoint", "forward mode", "the class", "a string"],
)
def test_anything_else_is_refused(adjoint) -> None:
    with pytest.raises(ix.ValidationError, match="'adjoint' of a scan must be"):
        scan_for(adjoint)


def test_run_still_refuses_a_diffrax_adjoint_for_a_stepper() -> None:
    from integrix.march import AB2Solver, run

    with pytest.raises(ix.ValidationError, match="AbstractStepper"):
        run(
            AB2Solver(),
            lambda t, y, args: -y,
            0.0,
            jnp.array(1.0),
            0.1,
            3,
            adjoint=dfx.RecursiveCheckpointAdjoint(),
        )
