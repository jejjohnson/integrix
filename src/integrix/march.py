"""Marches: time steppers, yₙ₊₁ = Φ(yₙ, tₙ, h).

The single-step schemes are diffrax solvers (design decision 8, #18), so
`diffrax.diffeqsolve`, its step-size controllers and its adjoints run them:
forward Euler, Heun, SSP-RK2, SSP-RK3, classical RK4, Ketcheson's SSP-RK(10,4),
IMEX-SSP2 and AB3 (whose history lives in the solver state). Each comes with
its pure step function for an autonomous right-hand side f(y).
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
from integrix._src.march._multistep import AB3Solver, ab3_step


# Plain sorted() order, which tests/test_public_api.py checks; ruff's RUF022
# would put the SCREAMING_CASE names (IMEX_SSP2, SSP_RK2) first.
__all__ = [  # noqa: RUF022
    "AB3Solver",
    "ForwardEulerDfx",
    "IMEX_SSP2",
    "RK2Heun",
    "RK3SSP",
    "RK4Classic",
    "SSP_RK104",
    "SSP_RK2",
    "ab3_step",
    "euler_step",
    "heun_step",
    "imex_ssp2_step",
    "rk3_ssp_step",
    "rk4_step",
]
