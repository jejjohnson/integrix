"""Tests for the traceable `stable_step` and the matrix-free `imex_solver` (#21).

sesmax-core's own tests run unchanged in ``test_ported_imex_control.py``; these
cover what moved with the port: `stable_step` now takes JAX arrays, traces
under jit and vmap, and differentiates, and `imex_solver` solves a stiff
problem matrix-free to the accuracy its tolerances ask for.
"""

from __future__ import annotations

import math

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_matches_scipy
from scipy.linalg import expm

import integrix as ix
from integrix.march import imex_solver, imex_stepsize_controller, run, stable_step


class TestStableStepTraces:
    def test_returns_a_jax_scalar(self) -> None:
        bound = stable_step((1.0,), speed=(2.0,))
        assert isinstance(bound, jax.Array)
        assert bound.shape == ()

    def test_jit_matches_eager_on_arrays(self) -> None:
        spacing = jnp.array([50.0, 50.0, 25.0])
        speed = jnp.array([3.0, 0.5, 0.0])
        diffusivity = jnp.array([10.0, 10.0, 2.0])
        eager = stable_step(spacing, speed=speed, diffusivity=diffusivity)
        traced = jax.jit(lambda s, u, k: stable_step(s, speed=u, diffusivity=k))(
            spacing, speed, diffusivity
        )
        assert_matches_scipy(traced, eager, rtol=1e-15)
        assert_matches_scipy(eager, 1.0 / 0.0924, rtol=1e-12)

    def test_vmap_over_speeds(self) -> None:
        speeds = jnp.array([[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
        bounds = jax.vmap(lambda u: stable_step(jnp.ones(2), speed=u))(speeds)
        assert_matches_scipy(bounds, np.array([1.0, 0.5, 0.5]), rtol=1e-15)

    def test_infinite_bound_under_jit_has_a_finite_gradient(self) -> None:
        bound, gradient = jax.value_and_grad(
            lambda u: stable_step(jnp.ones(1), speed=u)
        )(jnp.zeros(1))
        assert bool(jnp.isinf(bound))
        assert bool(jnp.all(jnp.isfinite(gradient)))

    def test_gradient_in_the_speeds_and_diffusivities(self) -> None:
        def bound(fields: jax.Array) -> jax.Array:
            speed, diffusivity = fields
            return stable_step(
                jnp.array([0.5, 2.0]), speed=speed, diffusivity=diffusivity
            )

        assert_grad_matches_fd(bound, jnp.array([[1.0, 0.3], [0.2, 0.7]]))

    def test_float32_inputs_give_a_float32_bound(self) -> None:
        bound = stable_step(
            jnp.ones(2, dtype=jnp.float32), speed=jnp.ones(2, dtype=jnp.float32)
        )
        assert bound.dtype == jnp.float32

    def test_a_traced_negative_spacing_raises_at_run_time(self) -> None:
        with pytest.raises(Exception, match="stable_step: 'spacing'"):
            jax.jit(lambda s: stable_step(s, speed=jnp.ones(1)))(jnp.array([-1.0]))

    def test_a_traced_nan_speed_raises_at_run_time(self) -> None:
        with pytest.raises(Exception, match=r"stable_step: '\|speed\|'"):
            jax.jit(lambda u: stable_step(jnp.ones(1), speed=u))(jnp.array([jnp.nan]))

    def test_traced_checks_fire_when_nothing_moves(self) -> None:
        # with no speed or diffusivity the bound is inf whatever the spacing,
        # so the check must ride on the result, not on the unused input
        with pytest.raises(Exception, match="stable_step: 'spacing'"):
            jax.jit(lambda s: stable_step(s))(jnp.array([-1.0]))
        with pytest.raises(Exception, match="stable_step: 'cfl'"):
            jax.jit(lambda c: stable_step((1.0,), speed=(0.0,), cfl=c))(
                jnp.array(jnp.nan)
            )

    def test_host_tuples_do_not_promote_float32(self) -> None:
        bound = stable_step(
            jnp.ones(2, dtype=jnp.float32), speed=(1.0, 0.0), cfl=np.float64(0.5)
        )
        assert bound.dtype == jnp.float32

    def test_rejects_a_vector_cfl(self) -> None:
        with pytest.raises(ix.ValidationError, match="'cfl' must be a scalar"):
            stable_step((1.0, 1.0), speed=(1.0, 1.0), cfl=np.array([0.5, 0.25]))

    def test_a_complex_speed_counts_by_its_magnitude(self) -> None:
        assert_matches_scipy(stable_step((1.0,), speed=(1j,)), 1.0, rtol=1e-15)

    def test_rejects_a_spacing_that_is_not_a_vector(self) -> None:
        with pytest.raises(ix.ValidationError, match="one value per axis"):
            stable_step(np.ones((2, 2)))


class TestImexSolver:
    @pytest.mark.slow
    def test_stiff_diffusion_matches_the_matrix_exponential(self) -> None:
        # y′ = −c D y + κ D₂ y on a periodic grid of 32, with κ D₂ stiff
        n_points = 32
        spacing = 1.0 / n_points
        forward = np.roll(np.eye(n_points), 1, axis=1)
        backward = np.roll(np.eye(n_points), -1, axis=1)
        advection = -(forward - backward) / (2.0 * spacing)
        diffusion = 0.05 * (forward - 2.0 * np.eye(n_points) + backward) / spacing**2
        initial = np.sin(2.0 * math.pi * np.arange(n_points) * spacing)
        exact = expm(0.2 * (advection + diffusion)) @ initial
        explicit = jnp.asarray(advection)
        implicit = jnp.asarray(diffusion)
        terms = dfx.MultiTerm(
            dfx.ODETerm(lambda t, y, args: explicit @ y),
            dfx.ODETerm(lambda t, y, args: implicit @ y),
        )
        solution = dfx.diffeqsolve(
            terms,
            imex_solver(rtol=1e-8, atol=1e-10),
            0.0,
            0.2,
            0.01,
            jnp.asarray(initial),
            stepsize_controller=imex_stepsize_controller(rtol=1e-8, atol=1e-10),
            max_steps=10_000,
        )
        # local error control at 1e-8: a global error well under 1e-5
        assert float(np.max(np.abs(np.asarray(solution.ys[0]) - exact))) < 1e-5

    def test_runs_through_run_with_a_multi_term(self) -> None:
        terms = dfx.MultiTerm(
            dfx.ODETerm(lambda t, y, args: -y), dfx.ODETerm(lambda t, y, args: -5 * y)
        )
        y = run(imex_solver(rtol=1e-10, atol=1e-12), terms, 0.0, jnp.ones(3), 0.01, 100)
        # KenCarp3 at h = 0.01 on y′ = −6y: third order, error well under 1e-5
        assert_matches_scipy(y, np.full(3, math.exp(-6.0)), rtol=1e-4)
