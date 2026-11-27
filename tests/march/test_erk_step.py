"""Tests for `erk_step`, one explicit Runge–Kutta step from any tableau."""

from __future__ import annotations

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import integrix as ix
from integrix.march import (
    RK3SSP,
    SSP_RK104,
    ForwardEulerDfx,
    RK2Heun,
    RK4Classic,
    erk_step,
    heun_step,
    rk3_ssp_step,
    rk4_step,
)


PAIRS = {
    "heun": (RK2Heun(), heun_step),
    "rk3": (RK3SSP(), rk3_ssp_step),
    "rk4": (RK4Classic(), rk4_step),
}
ORDERS = {"heun": 2, "rk3": 3, "rk4": 4, "tsit5": 5, "ssp104": 4}
SOLVERS = {
    "heun": RK2Heun(),
    "rk3": RK3SSP(),
    "rk4": RK4Classic(),
    "tsit5": dfx.Tsit5(),
    "ssp104": SSP_RK104(),
}


def _decay(t, y):
    return -y


@pytest.mark.parametrize("name", PAIRS)
def test_it_is_the_hand_written_step(name: str) -> None:
    solver, step = PAIRS[name]
    y = {"a": jnp.array([1.0, -2.0]), "b": jnp.array(0.5)}
    found = erk_step(solver, lambda t, y: jax.tree.map(lambda v: -v, y), y, 0.0, 0.1)
    expected = step(y, lambda y: jax.tree.map(lambda v: -v, y), 0.1)
    for leaf, reference in zip(
        jax.tree.leaves(found), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_allclose(leaf, reference, rtol=1e-15)


@pytest.mark.parametrize("name", SOLVERS)
def test_one_step_has_the_solvers_order(name: str) -> None:
    """The local error of y′ = −y at h is O(h^{p+1})."""
    errors = []
    for h in (0.1, 0.05):
        found = erk_step(SOLVERS[name], _decay, jnp.array(1.0), 0.0, h)
        errors.append(abs(float(found) - np.exp(-h)))
    slope = np.log2(errors[0] / errors[1])
    assert slope == pytest.approx(ORDERS[name] + 1, abs=0.3)


@pytest.mark.parametrize("name", SOLVERS)
def test_a_step_per_element_is_separate_steps(name: str) -> None:
    solver = SOLVERS[name]

    def rhs(t, y):
        return jnp.sin(t)[:, None] * y + jnp.cos(y)

    y = jnp.array([[0.3, -0.2], [1.0, 0.5], [0.0, 2.0]])
    t = jnp.array([0.0, 1.0, -0.5])
    dt = jnp.array([0.1, 0.2, -0.3])
    found = erk_step(solver, rhs, y, t, dt)
    for p in range(3):
        alone = erk_step(solver, rhs, y[p : p + 1], t[p : p + 1], dt[p : p + 1])
        np.testing.assert_allclose(found[p], alone[0], rtol=1e-14, atol=1e-15)


def test_a_time_dependent_rhs_is_read_at_the_stage_times() -> None:
    """RK4 integrates f(t) = t³ exactly over one step."""
    found = erk_step(RK4Classic(), lambda t, y: t**3, jnp.array(0.0), 1.0, 2.0)
    # ∫₁³ t³ dt = (81 − 1)/4
    np.testing.assert_allclose(found, 20.0, rtol=1e-14)


def test_a_zero_step_is_bit_for_bit() -> None:
    y = jnp.array([1.0, 2.0, 3.0])
    found = erk_step(RK4Classic(), _decay, y, 0.0, jnp.zeros(3))
    assert jnp.array_equal(found, y)


def test_a_backward_step_undoes_a_forward_one() -> None:
    y = jnp.array([1.0, -0.5])
    forward = erk_step(RK4Classic(), lambda t, y: jnp.sin(y + t), y, 0.0, 1e-3)
    back = erk_step(RK4Classic(), lambda t, y: jnp.sin(y + t), forward, 1e-3, -1e-3)
    np.testing.assert_allclose(back, y, atol=1e-14)


def test_forward_and_reverse_mode_agree_inside_a_scan() -> None:
    def rollout(y0):
        def body(y, k):
            return erk_step(
                RK3SSP(), lambda t, y: jnp.tanh(y) * t, y, 0.1 * k, 0.1
            ), None

        final, _ = jax.lax.scan(body, y0, jnp.arange(50.0))
        return final

    y0 = jnp.array([0.3, -0.7])
    forward = jax.jacfwd(rollout)(y0)
    reverse = jax.jacrev(rollout)(y0)
    np.testing.assert_allclose(forward, reverse, rtol=1e-12)


def test_the_states_dtypes_are_kept() -> None:
    y = jnp.ones(2, jnp.float32)
    found = erk_step(RK4Classic(), lambda t, y: jnp.ones(2, jnp.float64), y, 0.0, 0.1)
    assert found.dtype == jnp.float32


def test_jit_and_vmap_agree() -> None:
    y = jnp.array([[1.0, 2.0], [0.5, -1.0]])

    def one(y, h):
        return erk_step(dfx.Tsit5(), lambda t, y: -y * t, y, 1.0, h)

    eager = jnp.stack([one(row, 0.2) for row in y])
    np.testing.assert_allclose(jax.jit(jax.vmap(one, (0, None)))(y, 0.2), eager)


@pytest.mark.parametrize(
    "solver", [dfx.ImplicitEuler(), ForwardEulerDfx(), "rk4"], ids=str
)
def test_a_solver_without_an_explicit_tableau_is_refused(solver) -> None:
    with pytest.raises((ix.ValidationError, TypeError), match="solver"):
        erk_step(solver, _decay, jnp.array(1.0), 0.0, 0.1)
