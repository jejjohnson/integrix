"""Marches: time steppers, yₙ₊₁ = Φ(yₙ, tₙ, h).

The single-step schemes are diffrax solvers (design decision 8, #18), so
`diffrax.diffeqsolve`, its step-size controllers and its adjoints run them:
forward Euler, Heun, SSP-RK2, SSP-RK3, classical RK4, Ketcheson's SSP-RK(10,4),
IMEX-SSP2 and AB3 (whose history lives in the solver state). The schemes that
carry history between steps (AB2, leapfrog with a Robert–Asselin filter, and
split-explicit) are `AbstractStepper`s, run by `run` with the history in the
loop carry; `run` also takes the diffrax solvers. Each scheme comes with its
pure step function for an autonomous right-hand side f(y). `imex_solver`,
`imex_stepsize_controller` and `stable_step` (moved from sesmax-core) give a
matrix-free IMEX solve, its step control and the explicit CFL bound.
"""

from __future__ import annotations

from integrix._src.march._erk import (
    RK3SSP,
    SSP_RK2,
    SSP_RK104,
    ForwardEulerDfx,
    RK2Heun,
    RK4Classic,
    euler_step,
    heun_step,
    rk3_ssp_step,
    rk4_step,
)
from integrix._src.march._imex import IMEX_SSP2, imex_ssp2_step
from integrix._src.march._imex_control import (
    imex_solver,
    imex_stepsize_controller,
    stable_step,
)
from integrix._src.march._loop import AbstractStepper, run
from integrix._src.march._multistep import (
    AB2Solver,
    AB3Solver,
    LeapfrogRAFSolver,
    ab2_step,
    ab3_step,
    leapfrog_raf_step,
)
from integrix._src.march._split_explicit import (
    SplitExplicitRKSolver,
    SplitExplicitTerms,
    split_explicit_step,
)


# Plain sorted() order, which tests/test_public_api.py checks; ruff's RUF022
# would put the SCREAMING_CASE names (IMEX_SSP2, SSP_RK2) first.
__all__ = [  # noqa: RUF022
    "AB2Solver",
    "AB3Solver",
    "AbstractStepper",
    "ForwardEulerDfx",
    "IMEX_SSP2",
    "LeapfrogRAFSolver",
    "RK2Heun",
    "RK3SSP",
    "RK4Classic",
    "SSP_RK104",
    "SSP_RK2",
    "SplitExplicitRKSolver",
    "SplitExplicitTerms",
    "ab2_step",
    "ab3_step",
    "euler_step",
    "heun_step",
    "imex_solver",
    "imex_ssp2_step",
    "imex_stepsize_controller",
    "leapfrog_raf_step",
    "rk3_ssp_step",
    "rk4_step",
    "run",
    "split_explicit_step",
    "stable_step",
]
