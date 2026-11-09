"""Explicit Runge–Kutta steppers: diffrax solvers and their pure step functions.

Moved from finitevolX (``finitevolx/_src/timestepping/diffrax_solvers.py`` and
``explicit_rk.py`` at ``82090a3``) with their interfaces unchanged. The classes
are diffrax solvers, so `diffrax.diffeqsolve`, its step-size controllers and
its adjoints run them; the step functions take an autonomous right-hand side
``f(y)`` and any pytree state.

References:
    Shu, C.-W. and Osher, S. (1988). Efficient implementation of essentially
    non-oscillatory shock-capturing schemes. J. Comput. Phys. 77, 439–471.

    Gottlieb, S., Shu, C.-W. and Tadmor, E. (2001). Strong stability-preserving
    high-order time discretization methods. SIAM Review 43, 89–112.

    Ketcheson, D. I. (2008). Highly efficient strong stability-preserving
    Runge–Kutta methods with low-storage implementations. SIAM J. Sci.
    Comput. 30, 2113–2136.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import diffrax as dfx
import jax
import numpy as np
from diffrax import RESULTS
from jaxtyping import PyTree, ScalarLike


__all__ = [
    "RK3SSP",
    "SSP_RK2",
    "SSP_RK104",
    "ForwardEulerDfx",
    "RK2Heun",
    "RK4Classic",
    "euler_step",
    "heun_step",
    "rk3_ssp_step",
    "rk4_step",
]


# ---------------------------------------------------------------------------
# diffrax solvers
# ---------------------------------------------------------------------------


class ForwardEulerDfx(dfx.AbstractSolver):
    """Forward Euler as a diffrax solver: yₙ₊₁ = yₙ + h F(tₙ, yₙ).

    Order 1, one stage, no error estimate (fixed steps only). Included for
    completeness and parity with finitevolX; `diffrax.Euler` is the same
    scheme.

    Examples:
        >>> import diffrax as dfx
        >>> import jax.numpy as jnp
        >>> solution = dfx.diffeqsolve(
        ...     dfx.ODETerm(lambda t, y, args: -y),
        ...     ForwardEulerDfx(),
        ...     t0=0.0,
        ...     t1=1.0,
        ...     dt0=0.1,
        ...     y0=jnp.array(1.0),
        ... )
        >>> bool(jnp.allclose(solution.ys[0], 0.9**10))
        True
    """

    term_structure: ClassVar[Any] = dfx.AbstractTerm
    interpolation_cls: ClassVar[Any] = dfx.LocalLinearInterpolation

    def order(self, terms):
        return 1

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        dt = t1 - t0
        f0 = terms.vf(t0, y0, args)
        # yₙ₊₁ = yₙ + h F(tₙ, yₙ)
        y1 = jax.tree.map(lambda y, f: y + dt * f, y0, f0)
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)


class _FixedStepERK(dfx.AbstractERK):
    """An explicit Runge–Kutta solver with no embedded error estimate.

    ``step`` returns ``y_error = None``, so an adaptive step-size controller
    raises (as for `ForwardEulerDfx`) instead of reading a zero error
    estimate and taking unbounded steps.
    """

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        y1, _, dense_info, solver_state, result = super().step(
            terms, t0, t1, y0, args, solver_state, made_jump
        )
        return y1, None, dense_info, solver_state, result


class RK2Heun(dfx.AbstractERK):
    """Heun's method (RK2), with an embedded forward-Euler error estimate.

    Order 2, two stages, SSP with coefficient C = 1. The error weights are
    b̂ − b = [½, −½] (as `diffrax.Heun`), so adaptive controllers work.

    Butcher tableau: c = (0, 1), a₂₁ = 1, b = (½, ½).

    Examples:
        >>> RK2Heun().order(None)
        2
    """

    tableau: ClassVar[dfx.ButcherTableau] = dfx.ButcherTableau(
        c=np.array([1.0]),
        b_sol=np.array([0.5, 0.5]),
        # b_error = b_euler − b_sol, b_euler = [1, 0]
        b_error=np.array([0.5, -0.5]),
        a_lower=(np.array([1.0]),),
    )
    interpolation_cls: ClassVar[Any] = (
        dfx.ThirdOrderHermitePolynomialInterpolation.from_k
    )

    def order(self, terms):
        return 2


class RK3SSP(dfx.AbstractERK):
    """Third-order strong-stability-preserving Runge–Kutta (Shu–Osher).

    Order 3, three stages, SSP coefficient C = 1 (optimal): it keeps the
    monotonicity, positivity and TVD properties of forward Euler within the
    same CFL limit. The embedded second-order solution is Heun on the first
    two stages (b̂ = [½, ½, 0]), so adaptive controllers work.

    Butcher tableau: c = (0, 1, ½), a₂₁ = 1, a₃₁ = a₃₂ = ¼, b = (⅙, ⅙, ⅔).

    Examples:
        >>> RK3SSP().order(None)
        3
    """

    tableau: ClassVar[dfx.ButcherTableau] = dfx.ButcherTableau(
        c=np.array([1.0, 0.5]),
        b_sol=np.array([1.0 / 6, 1.0 / 6, 2.0 / 3]),
        # b_error = b_hat − b_sol, b_hat = [1/2, 1/2, 0] (Heun, 2nd order)
        b_error=np.array([1.0 / 3, 1.0 / 3, -2.0 / 3]),
        a_lower=(
            np.array([1.0]),
            np.array([0.25, 0.25]),
        ),
    )
    interpolation_cls: ClassVar[Any] = (
        dfx.ThirdOrderHermitePolynomialInterpolation.from_k
    )

    def order(self, terms):
        return 3


class RK4Classic(_FixedStepERK):
    """The classical fourth-order Runge–Kutta method, fixed steps only.

    Order 4, four stages, not SSP. The tableau has no embedded lower-order
    solution, so an adaptive controller raises; `diffrax.Tsit5` or
    `diffrax.Dopri5` are the adaptive fourth- and fifth-order choices.

    Butcher tableau: c = (0, ½, ½, 1), a₂₁ = a₃₂ = ½, a₄₃ = 1,
    b = (⅙, ⅓, ⅓, ⅙).

    Examples:
        >>> RK4Classic().order(None)
        4
    """

    tableau: ClassVar[dfx.ButcherTableau] = dfx.ButcherTableau(
        c=np.array([0.5, 0.5, 1.0]),
        b_sol=np.array([1.0 / 6, 1.0 / 3, 1.0 / 3, 1.0 / 6]),
        b_error=np.zeros(4),
        a_lower=(
            np.array([0.5]),
            np.array([0.0, 0.5]),
            np.array([0.0, 0.0, 1.0]),
        ),
    )
    interpolation_cls: ClassVar[Any] = (
        dfx.ThirdOrderHermitePolynomialInterpolation.from_k
    )

    def order(self, terms):
        return 4


class SSP_RK2(dfx.AbstractERK):
    """Second-order SSP Runge–Kutta, the same scheme as `RK2Heun`.

    Order 2, two stages, SSP coefficient C = 1, with an embedded forward-Euler
    error estimate.

    Examples:
        >>> SSP_RK2().order(None)
        2
    """

    tableau: ClassVar[dfx.ButcherTableau] = dfx.ButcherTableau(
        c=np.array([1.0]),
        b_sol=np.array([0.5, 0.5]),
        # b_error = b_euler − b_sol, b_euler = [1, 0]
        b_error=np.array([0.5, -0.5]),
        a_lower=(np.array([1.0]),),
    )
    interpolation_cls: ClassVar[Any] = (
        dfx.ThirdOrderHermitePolynomialInterpolation.from_k
    )

    def order(self, terms):
        return 2


class SSP_RK104(_FixedStepERK):
    """Ketcheson's ten-stage, fourth-order SSP Runge–Kutta, fixed steps only.

    Order 4, ten stages, SSP coefficient C = 6, the largest achievable at
    fourth order. There is no embedded error estimate, so an adaptive
    controller raises.

    Examples:
        >>> SSP_RK104().order(None)
        4
    """

    tableau: ClassVar[dfx.ButcherTableau] = dfx.ButcherTableau(
        # c[i] = sum(a_lower[i]) for i = 0..8 (excludes the first stage, c = 0)
        c=np.array(
            [
                1.0 / 6,
                1.0 / 3,
                1.0 / 2,
                2.0 / 3,
                1.0 / 3,
                1.0 / 2,
                2.0 / 3,
                5.0 / 6,
                1.0,
            ]
        ),
        b_sol=np.array([1.0 / 10] * 10),
        b_error=np.zeros(10),
        a_lower=(
            # Rows 1-4: all entries 1/6
            np.array([1.0 / 6]),
            np.array([1.0 / 6, 1.0 / 6]),
            np.array([1.0 / 6, 1.0 / 6, 1.0 / 6]),
            np.array([1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6]),
            # Rows 5-9: 1/15 for cols 0-4, then 1/6 for later cols
            # (convex combination at stage 6: 3/5*y0 + 2/5*y5 maps 1/6 -> 1/15)
            np.array([1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15]),
            np.array([1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 6]),
            np.array(
                [1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 6, 1.0 / 6]
            ),
            np.array([*[1.0 / 15] * 5, *[1.0 / 6] * 3]),
            np.array([*[1.0 / 15] * 5, *[1.0 / 6] * 4]),
        ),
    )
    interpolation_cls: ClassVar[Any] = (
        dfx.ThirdOrderHermitePolynomialInterpolation.from_k
    )

    def order(self, terms):
        return 4


# ---------------------------------------------------------------------------
# Pure step functions, f(y) autonomous
# ---------------------------------------------------------------------------


def euler_step(
    state: PyTree, rhs_fn: Callable[[PyTree], PyTree], dt: ScalarLike
) -> PyTree:
    """One forward-Euler step: yₙ₊₁ = yₙ + h f(yₙ).

    Args:
        state: The state yₙ, any pytree.
        rhs_fn: The autonomous right-hand side f(y).
        dt: The step h.

    Returns:
        The state yₙ₊₁.

    Examples:
        >>> import jax.numpy as jnp
        >>> bool(euler_step(jnp.array(1.0), lambda y: -y, 0.1) == 0.9)
        True
    """
    k1 = rhs_fn(state)
    return jax.tree.map(lambda y, f: y + dt * f, state, k1)


def heun_step(
    state: PyTree, rhs_fn: Callable[[PyTree], PyTree], dt: ScalarLike
) -> PyTree:
    """One Heun (RK2) step, order 2, SSP with C = 1.

    k₁ = f(yₙ), k₂ = f(yₙ + h k₁), yₙ₊₁ = yₙ + (h/2)(k₁ + k₂).

    Args:
        state: The state yₙ, any pytree.
        rhs_fn: The autonomous right-hand side f(y).
        dt: The step h.

    Returns:
        The state yₙ₊₁.

    Examples:
        >>> import jax.numpy as jnp
        >>> bool(jnp.allclose(heun_step(jnp.array(1.0), lambda y: -y, 0.1), 0.905))
        True
    """
    k1 = rhs_fn(state)
    state_star = jax.tree.map(lambda y, f: y + dt * f, state, k1)
    k2 = rhs_fn(state_star)
    return jax.tree.map(lambda y, f1, f2: y + 0.5 * dt * (f1 + f2), state, k1, k2)


def rk3_ssp_step(
    state: PyTree, rhs_fn: Callable[[PyTree], PyTree], dt: ScalarLike
) -> PyTree:
    """One third-order SSP Runge–Kutta step in Shu–Osher form, C = 1.

    y⁽¹⁾ = yₙ + h f(yₙ); y⁽²⁾ = ¾ yₙ + ¼ y⁽¹⁾ + ¼ h f(y⁽¹⁾);
    yₙ₊₁ = ⅓ yₙ + ⅔ y⁽²⁾ + ⅔ h f(y⁽²⁾).

    Args:
        state: The state yₙ, any pytree.
        rhs_fn: The autonomous right-hand side f(y).
        dt: The step h.

    Returns:
        The state yₙ₊₁.

    Examples:
        >>> import jax.numpy as jnp
        >>> y = rk3_ssp_step(jnp.array(1.0), lambda y: -y, 0.1)
        >>> bool(jnp.allclose(y, 1.0 - 0.1 + 0.1**2 / 2 - 0.1**3 / 6))
        True
    """
    k1 = rhs_fn(state)
    y1 = jax.tree.map(lambda y, f: y + dt * f, state, k1)

    k2 = rhs_fn(y1)
    y2 = jax.tree.map(
        lambda y, y1_, f: 0.75 * y + 0.25 * y1_ + 0.25 * dt * f,
        state,
        y1,
        k2,
    )

    k3 = rhs_fn(y2)
    return jax.tree.map(
        lambda y, y2_, f: (1.0 / 3.0) * y + (2.0 / 3.0) * y2_ + (2.0 / 3.0) * dt * f,
        state,
        y2,
        k3,
    )


def rk4_step(
    state: PyTree, rhs_fn: Callable[[PyTree], PyTree], dt: ScalarLike
) -> PyTree:
    """One classical fourth-order Runge–Kutta step.

    k₁ = f(yₙ), k₂ = f(yₙ + (h/2) k₁), k₃ = f(yₙ + (h/2) k₂), k₄ = f(yₙ + h k₃),
    yₙ₊₁ = yₙ + (h/6)(k₁ + 2k₂ + 2k₃ + k₄).

    Args:
        state: The state yₙ, any pytree.
        rhs_fn: The autonomous right-hand side f(y).
        dt: The step h.

    Returns:
        The state yₙ₊₁.

    Examples:
        >>> import jax.numpy as jnp
        >>> y = rk4_step({"a": jnp.array(1.0)}, lambda y: {"a": -y["a"]}, 0.1)
        >>> bool(jnp.allclose(y["a"], jnp.exp(-0.1), rtol=1e-6))
        True
    """
    k1 = rhs_fn(state)
    y1 = jax.tree.map(lambda y, f: y + 0.5 * dt * f, state, k1)

    k2 = rhs_fn(y1)
    y2 = jax.tree.map(lambda y, f: y + 0.5 * dt * f, state, k2)

    k3 = rhs_fn(y2)
    y3 = jax.tree.map(lambda y, f: y + dt * f, state, k3)

    k4 = rhs_fn(y3)
    return jax.tree.map(
        lambda y, f1, f2, f3, f4: y + (dt / 6.0) * (f1 + 2.0 * f2 + 2.0 * f3 + f4),
        state,
        k1,
        k2,
        k3,
        k4,
    )
