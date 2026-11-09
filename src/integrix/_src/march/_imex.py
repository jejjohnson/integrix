"""The IMEX-SSP2(2,2,2) stepper: a diffrax solver and its pure step function.

Moved from finitevolX (``diffrax_solvers.py`` and ``imex.py`` at ``82090a3``)
with its interface unchanged. The ODE splits as y′ = F_E(t, y) + F_I(t, y),
with F_E explicit (advection) and F_I implicit (vertical diffusion, say).

References:
    Pareschi, L. and Russo, G. (2005). Implicit–explicit Runge–Kutta schemes
    and applications to hyperbolic systems with relaxation. J. Sci. Comput.
    25, 129–155.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, ClassVar

import diffrax as dfx
import equinox as eqx
import jax
import numpy as np
from jaxtyping import PyTree, ScalarLike


__all__ = [
    "IMEX_SSP2",
    "imex_ssp2_step",
]


# γ = 1 − 1/√2, the SDIRK diagonal of IMEX-SSP2
_GAMMA = 1.0 - 1.0 / math.sqrt(2.0)


class IMEX_SSP2(dfx.AbstractRungeKutta, dfx.AbstractImplicitSolver):
    """IMEX-SSP2(2,2,2) for a `diffrax.MultiTerm` of an explicit and an implicit term.

    The explicit part is SSP with C = 1; the implicit part is A-stable SDIRK
    with γ = 1 − 1/√2. Order 2. The embedded first-order solution
    y + h (F_E(Y₁) + F_I(Y₁)) (b̂ = [1, 0] in both tableaux) gives the error
    estimate for adaptive step-size control. Dense output is linear between
    steps.

    Explicit tableau: c = (0, 1), a₂₁ = 1, b = (½, ½). Implicit tableau:
    c = (γ, 1 − γ), a₁₁ = a₂₂ = γ, a₂₁ = 1 − 2γ, b = (½, ½).

    With fixed steps, give the implicit root finder its own tolerances
    (``IMEX_SSP2(root_finder=dfx.VeryChord(rtol=1e-8, atol=1e-8))``); with a
    `diffrax.PIDController`, the default root finder takes the controller's.

    Examples:
        >>> import diffrax as dfx
        >>> import jax.numpy as jnp
        >>> terms = dfx.MultiTerm(
        ...     dfx.ODETerm(lambda t, y, args: -y),
        ...     dfx.ODETerm(lambda t, y, args: -10.0 * y),
        ... )
        >>> solver = IMEX_SSP2(root_finder=dfx.VeryChord(rtol=1e-10, atol=1e-10))
        >>> solution = dfx.diffeqsolve(
        ...     terms, solver, 0.0, 1.0, 0.01, jnp.array(1.0)
        ... )
        >>> bool(jnp.allclose(solution.ys[0], jnp.exp(-11.0), rtol=1e-2))
        True
    """

    tableau: ClassVar[dfx.MultiButcherTableau] = dfx.MultiButcherTableau(
        # Explicit part
        dfx.ButcherTableau(
            c=np.array([1.0]),
            b_sol=np.array([0.5, 0.5]),
            # b_error = b_hat − b_sol, b_hat = [1, 0]
            b_error=np.array([0.5, -0.5]),
            a_lower=(np.array([1.0]),),
        ),
        # Implicit part (SDIRK): c₁ = γ, c₂ = (1 − 2γ) + γ = 1 − γ
        dfx.ButcherTableau(
            c=np.array([1.0 - _GAMMA]),
            b_sol=np.array([0.5, 0.5]),
            # b_error = b_hat − b_sol, b_hat = [1, 0] (same embedded weights)
            b_error=np.array([0.5, -0.5]),
            a_lower=(np.array([1.0 - 2.0 * _GAMMA]),),
            a_diagonal=np.array([_GAMMA, _GAMMA]),
            a_predictor=(np.array([1.0]),),
            c1=_GAMMA,
        ),
    )

    @staticmethod
    def interpolation_cls(t0, t1, y0, y1, k):
        # linear dense output; k (one stage tuple per term) is not needed
        del k
        return dfx.LocalLinearInterpolation(t0=t0, t1=t1, y0=y0, y1=y1)

    term_structure: ClassVar[Any] = dfx.MultiTerm[
        tuple[dfx.AbstractTerm, dfx.AbstractTerm]
    ]
    calculate_jacobian: ClassVar[Any] = dfx.CalculateJacobian.first_stage

    # diffrax declares these on AbstractImplicitSolver for subclasses to fill,
    # as its own KenCarp3 does; ty reads the declaration as a class variable.
    root_finder: Any = dfx.with_stepsize_controller_tols(dfx.VeryChord)()  # ty: ignore[invalid-attribute-override]
    root_find_max_steps: int = eqx.field(static=True, default=10)  # ty: ignore[invalid-attribute-override]

    def order(self, terms):
        return 2


def imex_ssp2_step(
    state: PyTree,
    rhs_explicit: Callable[[PyTree], PyTree],
    rhs_implicit: Callable[[PyTree], PyTree],
    implicit_solve: Callable[[PyTree, ScalarLike], PyTree],
    dt: ScalarLike,
) -> PyTree:
    """One IMEX-SSP2(2,2,2) step, with the caller's implicit solve.

    Second order in the explicit part, the implicit part and the coupled
    split (the implicit row (1 − 2γ, γ) gives ∑ᵢ bᵢ cᵢ = ½).

    Algorithm:
        1. Y₁ = yₙ + γh f_I(Y₁): ``implicit_solve(yₙ, γh)``.
        2. Y₂* = yₙ + h f_E(Y₁) + (1 − 2γ) h f_I(Y₁), then
           Y₂ = Y₂* + γh f_I(Y₂): ``implicit_solve(Y₂*, γh)``.
        3. yₙ₊₁ = yₙ + (h/2)(f_E(Y₁) + f_E(Y₂)) + (h/2)(f_I(Y₁) + f_I(Y₂)).

    Args:
        state: The state yₙ, any pytree.
        rhs_explicit: The non-stiff right-hand side f_E(y).
        rhs_implicit: The stiff right-hand side f_I(y).
        implicit_solve: Solves Y − γh f_I(Y) = r for Y, given (r, γh); for
            vertical diffusion, a tridiagonal solve along columns.
        dt: The step h.

    Returns:
        The state yₙ₊₁.

    Examples:
        >>> import jax.numpy as jnp
        >>> y = imex_ssp2_step(
        ...     jnp.array(1.0),
        ...     lambda y: -y,
        ...     lambda y: -10.0 * y,
        ...     lambda r, gamma_dt: r / (1.0 + 10.0 * gamma_dt),
        ...     0.01,
        ... )
        >>> bool(jnp.allclose(y, jnp.exp(-0.11), rtol=1e-4))
        True
    """
    gamma_dt = _GAMMA * dt

    # Stage 1: Y₁ = yₙ + γh f_I(Y₁)
    y1 = implicit_solve(state, gamma_dt)
    fe_1 = rhs_explicit(y1)
    fi_1 = rhs_implicit(y1)

    # Stage 2: Y₂* = yₙ + h f_E(Y₁) + (1 − 2γ) h f_I(Y₁); Y₂ = Y₂* + γh f_I(Y₂)
    y2_star = jax.tree.map(
        lambda y, fe, fi: y + dt * fe + (1.0 - 2.0 * _GAMMA) * dt * fi,
        state,
        fe_1,
        fi_1,
    )
    y2 = implicit_solve(y2_star, gamma_dt)
    fe_2 = rhs_explicit(y2)
    fi_2 = rhs_implicit(y2)

    # yₙ₊₁ = yₙ + (h/2)(f_E(Y₁) + f_E(Y₂)) + (h/2)(f_I(Y₁) + f_I(Y₂))
    return jax.tree.map(
        lambda y, fe1, fe2, fi1, fi2: (
            y + 0.5 * dt * (fe1 + fe2) + 0.5 * dt * (fi1 + fi2)
        ),
        state,
        fe_1,
        fe_2,
        fi_1,
        fi_2,
    )
