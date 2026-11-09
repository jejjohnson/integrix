"""Gate G1-order for the diffrax-solver steppers.

Each stepper's global error at a fixed end time, on y′ = λy and on linear
advection, has a log–log slope against h within 0.1 of its order. Advection
is semi-discretised with second-order centred differences on a periodic grid,
and the reference is the exact solution of that semi-discrete system,
exp(T A) y₀ (`scipy.linalg.expm`), so the error measured is the stepper's
alone, not the spatial truncation.
"""

from __future__ import annotations

import math

import diffrax as dfx
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_order
from scipy.linalg import expm

from integrix.march import (
    IMEX_SSP2,
    RK3SSP,
    SSP_RK2,
    SSP_RK104,
    AB3Solver,
    ForwardEulerDfx,
    RK2Heun,
    RK4Classic,
)


# λ for the decay problem, and T for both problems
DECAY_RATE = -1.0
END_TIME = 1.0

# (solver, order, step sizes): each sequence sits between the pre-asymptotic
# regime and the float64 round-off floor for its order.
EXPLICIT = [
    pytest.param(ForwardEulerDfx(), 1, (0.02, 0.01, 0.005, 0.0025), id="euler"),
    pytest.param(RK2Heun(), 2, (0.04, 0.02, 0.01, 0.005), id="heun"),
    pytest.param(SSP_RK2(), 2, (0.04, 0.02, 0.01, 0.005), id="ssp_rk2"),
    pytest.param(RK3SSP(), 3, (0.04, 0.02, 0.01, 0.005), id="rk3_ssp"),
    pytest.param(RK4Classic(), 4, (0.1, 0.05, 0.025, 0.0125), id="rk4"),
    pytest.param(SSP_RK104(), 4, (0.25, 0.125, 0.0625, 0.03125), id="ssp_rk104"),
    pytest.param(AB3Solver(), 3, (0.04, 0.02, 0.01, 0.005), id="ab3"),
]


def _imex_solver() -> IMEX_SSP2:
    """IMEX-SSP2 with a root find tight enough to leave only the time error."""
    return IMEX_SSP2(root_finder=dfx.VeryChord(rtol=1e-13, atol=1e-13))


def _final(terms, solver, y0, step: float) -> np.ndarray:
    """y(T) from ``diffeqsolve`` with constant steps h."""
    n_steps = round(END_TIME / step)
    solution = dfx.diffeqsolve(
        terms,
        solver,
        t0=0.0,
        t1=END_TIME,
        dt0=step,
        y0=y0,
        max_steps=n_steps + 1,
    )
    return np.asarray(solution.ys[0])


def _advection_operator(n_points: int = 32, speed: float = 1.0) -> np.ndarray:
    """A = −c D, with D the periodic centred difference on [0, 1)."""
    spacing = 1.0 / n_points
    shift_forward = np.roll(np.eye(n_points), 1, axis=1)
    shift_backward = np.roll(np.eye(n_points), -1, axis=1)
    # (D u)ⱼ = (uⱼ₊₁ − uⱼ₋₁) / 2Δx
    difference = (shift_forward - shift_backward) / (2.0 * spacing)
    return -speed * difference


def _diffusion_operator(n_points: int = 32, diffusivity: float = 1e-3) -> np.ndarray:
    """B = κ D₂, with D₂ the periodic second difference on [0, 1)."""
    spacing = 1.0 / n_points
    shift_forward = np.roll(np.eye(n_points), 1, axis=1)
    shift_backward = np.roll(np.eye(n_points), -1, axis=1)
    # (D₂ u)ⱼ = (uⱼ₊₁ − 2uⱼ + uⱼ₋₁) / Δx²
    second = (shift_forward - 2.0 * np.eye(n_points) + shift_backward) / spacing**2
    return diffusivity * second


def _initial_wave(n_points: int = 32) -> np.ndarray:
    """u₀(x) = sin 2πx on the grid."""
    return np.sin(2.0 * math.pi * np.arange(n_points) / n_points)


@pytest.mark.slow
class TestDecay:
    @pytest.mark.parametrize(("solver", "order", "steps"), EXPLICIT)
    def test_order_on_exponential_decay(self, solver, order: int, steps) -> None:
        # y(T) = y₀ e^{λT}
        exact = math.exp(DECAY_RATE * END_TIME)
        terms = dfx.ODETerm(lambda t, y, args: DECAY_RATE * y)
        errors = {
            step: abs(float(_final(terms, solver, jnp.array(1.0), step)) - exact)
            for step in steps
        }
        assert_order(errors, order)

    def test_imex_order_on_split_decay(self) -> None:
        # y′ = −y (explicit) − 2y (implicit), y(T) = e^{−3T}
        exact = math.exp(-3.0 * END_TIME)
        terms = dfx.MultiTerm(
            dfx.ODETerm(lambda t, y, args: -y),
            dfx.ODETerm(lambda t, y, args: -2.0 * y),
        )
        errors = {
            step: abs(
                float(_final(terms, _imex_solver(), jnp.array(1.0), step)) - exact
            )
            for step in (0.04, 0.02, 0.01, 0.005)
        }
        assert_order(errors, 2)


@pytest.mark.slow
class TestAdvection:
    @pytest.mark.parametrize(("solver", "order", "steps"), EXPLICIT)
    def test_order_on_linear_advection(self, solver, order: int, steps) -> None:
        operator = _advection_operator()
        initial = _initial_wave()
        # y(T) = exp(T A) y₀, the semi-discrete solution
        exact = expm(END_TIME * operator) @ initial
        matrix = jnp.asarray(operator)
        terms = dfx.ODETerm(lambda t, y, args: matrix @ y)
        errors = {
            step: float(
                np.max(
                    np.abs(_final(terms, solver, jnp.asarray(initial), step) - exact)
                )
            )
            for step in steps
        }
        assert_order(errors, order)

    def test_imex_order_on_advection_diffusion(self) -> None:
        advection = _advection_operator()
        diffusion = _diffusion_operator()
        initial = _initial_wave()
        exact = expm(END_TIME * (advection + diffusion)) @ initial
        explicit = jnp.asarray(advection)
        implicit = jnp.asarray(diffusion)
        terms = dfx.MultiTerm(
            dfx.ODETerm(lambda t, y, args: explicit @ y),
            dfx.ODETerm(lambda t, y, args: implicit @ y),
        )
        errors = {
            step: float(
                np.max(
                    np.abs(
                        _final(terms, _imex_solver(), jnp.asarray(initial), step)
                        - exact
                    )
                )
            )
            for step in (0.02, 0.01, 0.005, 0.0025)
        }
        assert_order(errors, 2)
