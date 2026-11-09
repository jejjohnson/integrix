"""sesmax-core's tests for the IMEX solver, step control and CFL bound.

Ported from sesmax ``packages/sesmax-core/tests/test_solvers.py`` at
``f08f59d`` (#21): every test body is unchanged; only the import path moves
from ``sesmax_core.solvers`` to ``integrix.march``. The tests that integrate
sesmax-core's Burgers model, and the check that sesmax-core exports
``stable_step`` from its root, stay in sesmax (sesmax#116 runs them against
integrix).
"""

from __future__ import annotations

import diffrax as dfx
import optimistix as optx
import pytest

from integrix.march import imex_solver, imex_stepsize_controller, stable_step


class TestImexSolverConstruction:
    def test_solver_is_kencarp_with_matrix_free_newton(self) -> None:
        solver = imex_solver()
        assert isinstance(solver, dfx.KenCarp3)
        # The implicit stage must use a Krylov (GMRES) linear solver, not the
        # default dense AutoLinearSolver — that's the whole point of #55.
        rf = solver.root_finder
        assert isinstance(rf, optx.Newton)
        assert type(rf.linear_solver).__name__ == "GMRES"

    def test_controller_is_adaptive(self) -> None:
        assert isinstance(imex_stepsize_controller(), dfx.PIDController)

    def test_tolerances_threaded(self) -> None:
        solver = imex_solver(rtol=1e-3, atol=1e-7)
        assert solver.root_finder.rtol == 1e-3
        assert solver.root_finder.atol == 1e-7


def test_stable_step_matches_the_hand_computed_bound():
    """Δt_max = 1 / (Σ |u| / Δx + 2 Σ K / Δx²), worked by hand.

    With Δx = (50, 50, 25), u = (3, 0.5, 0) and K = (10, 10, 2): the advective
    rate is 0.06 + 0.01 = 0.07, the diffusive rate is 2 (0.004 + 0.004 + 0.0032)
    = 0.0224, and the bound is 1 / 0.0924.
    """
    bound = stable_step(
        (50.0, 50.0, 25.0), speed=(3.0, 0.5, 0.0), diffusivity=(10.0, 10.0, 2.0)
    )
    assert bound == pytest.approx(1.0 / 0.0924, rel=1e-12)


def test_stable_step_scales_with_cfl_and_ignores_the_sign_of_a_speed():
    base = stable_step((1.0, 1.0), speed=(1.0, 2.0))
    assert stable_step((1.0, 1.0), speed=(-1.0, -2.0)) == base
    assert stable_step((1.0, 1.0), speed=(1.0, 2.0), cfl=0.5) == pytest.approx(
        0.5 * base
    )


def test_stable_step_scales_with_spacing_for_advection_and_its_square_for_diffusion():
    coarse_advection = stable_step((1.0,), speed=(1.0,))
    fine_advection = stable_step((0.5,), speed=(1.0,))
    assert fine_advection == pytest.approx(0.5 * coarse_advection)
    coarse_diffusion = stable_step((1.0,), diffusivity=(1.0,))
    fine_diffusion = stable_step((0.5,), diffusivity=(1.0,))
    assert fine_diffusion == pytest.approx(0.25 * coarse_diffusion)


def test_stable_step_is_infinite_when_nothing_moves_or_diffuses():
    assert stable_step((1.0, 1.0)) == float("inf")
    assert stable_step((1.0,), speed=(0.0,), diffusivity=(0.0,)) == float("inf")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spacing": (0.0,)},
        {"spacing": (-1.0,)},
        {"spacing": (1.0,), "cfl": 0.0},
        {"spacing": (1.0,), "diffusivity": (-1.0,)},
        {"spacing": (1.0,), "speed": (float("nan"),)},
        {"spacing": (1.0,), "speed": (float("inf"),)},
        {"spacing": (1.0, 1.0), "speed": (1.0,)},
        {"spacing": (1.0,), "diffusivity": (1.0, 1.0)},
    ],
)
def test_stable_step_rejects_bad_inputs(kwargs):
    with pytest.raises(ValueError, match="stable_step"):
        stable_step(**kwargs)
