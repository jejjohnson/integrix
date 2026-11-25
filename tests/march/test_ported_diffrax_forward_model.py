"""pipekit-jax's tests of DiffraxForwardModel, run against integrix (pipekit#283).

Ported from pipekit ``packages/pipekit-jax/tests/test_diffrax_model.py`` at
``5fb9a12``: the bodies of `TestODECorrectness` and `TestSDE` are unchanged.
``pipekit_jax`` becomes ``integrix.march``, and pipekit-cycle's adjoint specs
become stand-ins, since integrix (Core tier 1) never imports pipekit-cycle
(tier 2). So the protocols are checked by their members, and the cycle is a
loop of steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import diffrax
import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest
from _harness import assert_traces

from integrix.exceptions import ValidationError
from integrix.march import RK3SSP, DiffraxForwardModel


# A parity suite ported from another library: the integration tier.
pytestmark = pytest.mark.integration


# Stand-ins for pipekit_cycle.adjoints at 5fb9a12, matched by class name.
@dataclass(frozen=True)
class RecursiveCheckpointAdjoint:
    checkpoints: int | None = None


@dataclass(frozen=True)
class BacksolveAdjoint:
    pass


def _decay(t, y, args):
    return -y


class _RateDecay(eqx.Module):
    """Parametrised drift; the rate is a pytree leaf of the terms."""

    rate: jax.Array

    def __call__(self, t, y, args):
        return -self.rate * y


@pytest.fixture
def model():
    return DiffraxForwardModel(vector_field=_decay, dt0=0.01, dt=1.0)


class TestProtocols:
    def test_the_members_of_a_forward_model(self, model):
        assert callable(model.step)
        assert model.dt == 1.0
        assert model.state_signature is None

    def test_the_members_of_an_adjoint_holder(self, model):
        assert isinstance(model.adjoint, dfx.RecursiveCheckpointAdjoint)
        swapped = model.with_adjoint(BacksolveAdjoint())
        assert swapped.adjoint == BacksolveAdjoint()
        assert isinstance(model.adjoint, dfx.RecursiveCheckpointAdjoint)  # unchanged


class TestODECorrectness:
    def test_matches_closed_form(self, model):
        y0 = jnp.array([1.0, 2.0])
        out = model.step(y0, 1.0)
        assert jnp.allclose(out, y0 * jnp.exp(-1.0), atol=1e-4)

    def test_as_dynamics_explicit_span(self, model):
        dyn = model.as_dynamics()
        y0 = jnp.array([1.0])
        assert jnp.allclose(dyn(y0, 0.0, 2.0), y0 * jnp.exp(-2.0), atol=1e-4)

    def test_gradient_through_step(self):
        def loss(rate):
            m = DiffraxForwardModel(vector_field=_RateDecay(rate), dt0=0.01)
            return m.step(jnp.array([1.0]), 1.0)[0]

        # d/dr exp(-r) at r=1 is -exp(-1)
        g = jax.grad(loss)(jnp.asarray(1.0))
        assert jnp.allclose(g, -jnp.exp(-1.0), atol=1e-3)

    def test_gradient_with_backsolve(self):
        # BacksolveAdjoint differentiates w.r.t. inputs of the solve, so
        # the parameter rides the terms pytree (an eqx.Module field), not
        # a Python closure.
        def loss(rate):
            m = DiffraxForwardModel(
                vector_field=_RateDecay(rate),
                dt0=0.01,
                adjoint=BacksolveAdjoint(),
            )
            return m.step(jnp.array([1.0]), 1.0)[0]

        g = jax.grad(loss)(jnp.asarray(1.0))
        assert jnp.allclose(g, -jnp.exp(-1.0), atol=1e-3)


class TestSDE:
    def test_sde_smoke_and_shapes(self):
        m = DiffraxForwardModel(
            vector_field=_decay,
            # diffrax >= 0.7: diagonal noise is a lineax operator mapping
            # the Brownian increment into state space.
            diffusion=lambda t, y, args: lx.DiagonalLinearOperator(
                0.1 * jnp.ones_like(y)
            ),
            sde_key=jax.random.key(0),
            solver=diffrax.Heun(),
            dt0=0.01,
        )
        out = m.step(jnp.ones(3), 1.0)
        assert out.shape == (3,)
        assert jnp.all(jnp.isfinite(out))

    def test_sde_nonsquare_noise(self):
        # Two Brownian channels driving a three-dimensional state: the
        # diffusion is a (3, 2) matrix and brownian_shape must be (2,).
        B = jnp.array([[0.1, 0.0], [0.0, 0.1], [0.05, 0.05]])
        m = DiffraxForwardModel(
            vector_field=_decay,
            diffusion=lambda t, y, args: B,
            brownian_shape=(2,),
            sde_key=jax.random.key(1),
            solver=diffrax.Heun(),
            dt0=0.01,
        )
        out = m.step(jnp.ones(3), 1.0)
        assert out.shape == (3,)
        assert jnp.all(jnp.isfinite(out))

    def test_sde_requires_key(self):
        with pytest.raises(ValueError, match="sde_key"):
            DiffraxForwardModel(
                vector_field=_decay,
                diffusion=lambda t, y, args: lx.DiagonalLinearOperator(
                    jnp.ones_like(y)
                ),
            )


class TestInIntegrix:
    def test_an_sde_without_a_key_names_it(self):
        with pytest.raises(ValidationError, match="'sde_key' is needed"):
            DiffraxForwardModel(_decay, diffusion=lambda t, y, args: jnp.ones(1))

    def test_a_cycle_of_windows(self, model):
        state = jnp.array([1.0])
        for _ in range(2):
            state = model.step(state, model.dt)
        assert jnp.allclose(state, jnp.exp(-2.0), atol=1e-3)

    def test_a_pytree_state(self):
        model = DiffraxForwardModel(lambda t, y, args: jax.tree.map(jnp.negative, y))
        out = model.step({"u": jnp.ones(2), "v": jnp.ones(3)}, 1.0)
        assert jnp.allclose(out["u"], jnp.exp(-1.0), atol=1e-4)
        assert out["v"].shape == (3,)

    @pytest.mark.parametrize(
        "spec",
        [
            dfx.DirectAdjoint(),
            dfx.RecursiveCheckpointAdjoint(checkpoints=4),
            RecursiveCheckpointAdjoint(checkpoints=4),
        ],
    )
    def test_an_adjoint_of_diffrax_or_a_spec(self, model, spec):
        fn = model.with_adjoint(spec).step
        gradient = jax.grad(lambda y: fn(y, 1.0).sum())(jnp.ones(2))
        assert jnp.allclose(gradient, jnp.exp(-1.0), atol=1e-4)

    def test_an_own_scheme_under_the_transforms(self):
        model = DiffraxForwardModel(_decay, solver=RK3SSP(), dt0=0.1)
        assert_traces(lambda y: model.step(y, 1.0), jnp.ones((2, 3)))

    def test_a_model_is_a_jit_argument(self, model):
        step = eqx.filter_jit(lambda model, y: model.step(y, 1.0))
        assert jnp.allclose(step(model, jnp.ones(2)), jnp.exp(-1.0), atol=1e-4)
