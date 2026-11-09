"""Multistep steppers: Adams–Bashforth 2 and 3, and leapfrog with an RA filter.

Moved from finitevolX (``diffrax_solvers.py`` and ``multistep.py`` at
``82090a3``). AB3 keeps its history in the diffrax solver state, so
`diffrax.diffeqsolve` runs it; AB2 and leapfrog are `AbstractStepper`s run by
`integrix.march.run` (design decision 8, #18), taking the diffrax vector field
``rhs(t, y, args)`` and carrying their history in the loop. The pure step
functions keep finitevolX's interfaces, an autonomous f(y).

References:
    Durran, D. R. (2010). Numerical Methods for Fluid Dynamics, 2nd ed.
    Springer.

    Robert, A. J. (1966). The integration of a low order spectral form of the
    primitive meteorological equations. J. Meteor. Soc. Japan 44, 237–245.

    Asselin, R. (1972). Frequency filter for time integrations. Mon. Wea. Rev.
    100, 487–490.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, ClassVar

import diffrax as dfx
import jax
import jax.numpy as jnp
from diffrax import RESULTS
from jaxtyping import Array, Float, PyTree, ScalarLike

from integrix._src.march._loop import AbstractStepper
from integrix.exceptions import ValidationError


__all__ = [
    "AB2Solver",
    "AB3Solver",
    "LeapfrogRAFSolver",
    "ab2_step",
    "ab3_step",
    "leapfrog_raf_step",
]


def ab2_step(
    state: PyTree,
    rhs_fn: Callable[[PyTree], PyTree],
    dt: ScalarLike,
    rhs_nm1: PyTree,
) -> tuple[PyTree, PyTree, PyTree]:
    """One second-order Adams–Bashforth step.

    yₙ₊₁ = yₙ + (h/2)(3 fₙ − fₙ₋₁), one evaluation of f per step.

    Args:
        state: The state yₙ, any pytree.
        rhs_fn: The autonomous right-hand side f(y).
        dt: The step h.
        rhs_nm1: fₙ₋₁, the right-hand side one step back.

    Returns:
        ``(yₙ₊₁, fₙ, fₙ₋₁)``: thread fₙ as the next ``rhs_nm1``.

    Examples:
        >>> import jax.numpy as jnp
        >>> y, f_n, _ = ab2_step(jnp.array(1.0), lambda y: -y, 0.1, -1.0)
        >>> bool(jnp.allclose(y, 0.9)), bool(f_n == -1.0)
        (True, True)
    """
    rhs_n = rhs_fn(state)
    # yₙ₊₁ = yₙ + (h/2)(3 fₙ − fₙ₋₁)
    new_state = jax.tree.map(
        lambda y, fn, fnm1: y + (dt / 2.0) * (3.0 * fn - fnm1),
        state,
        rhs_n,
        rhs_nm1,
    )
    return new_state, rhs_n, rhs_nm1


def ab3_step(
    state: PyTree,
    rhs_fn: Callable[[PyTree], PyTree],
    dt: ScalarLike,
    rhs_nm1: PyTree,
    rhs_nm2: PyTree,
) -> tuple[PyTree, PyTree, PyTree]:
    """One third-order Adams–Bashforth step.

    yₙ₊₁ = yₙ + (h/12)(23 fₙ − 16 fₙ₋₁ + 5 fₙ₋₂), one evaluation of f per
    step. The start must be at least second-order accurate to keep the
    global order 3: bootstrap y₁ and y₂ with `rk3_ssp_step`, as `AB3Solver`
    does; the crude start fₙ₋₂ = fₙ₋₁ = f(y₀) degrades AB3 to second order.

    Args:
        state: The state yₙ, any pytree.
        rhs_fn: The autonomous right-hand side f(y).
        dt: The step h.
        rhs_nm1: fₙ₋₁, the right-hand side one step back.
        rhs_nm2: fₙ₋₂, two steps back.

    Returns:
        ``(yₙ₊₁, fₙ, fₙ₋₁)``: thread fₙ as the next ``rhs_nm1`` and fₙ₋₁ as
        the next ``rhs_nm2``.

    Examples:
        >>> import jax.numpy as jnp
        >>> f = lambda y: -y
        >>> y, f_n, f_nm1 = ab3_step(jnp.array(1.0), f, 0.1, f(1.0), f(1.0))
        >>> bool(jnp.allclose(y, 0.9)), bool(f_n == -1.0)
        (True, True)
    """
    rhs_n = rhs_fn(state)
    # yₙ₊₁ = yₙ + (h/12)(23 fₙ − 16 fₙ₋₁ + 5 fₙ₋₂)
    new_state = jax.tree.map(
        lambda y, fn, fnm1, fnm2: (
            y + (dt / 12.0) * (23.0 * fn - 16.0 * fnm1 + 5.0 * fnm2)
        ),
        state,
        rhs_n,
        rhs_nm1,
        rhs_nm2,
    )
    return new_state, rhs_n, rhs_nm1


class AB3Solver(dfx.AbstractSolver):
    """Third-order Adams–Bashforth as a diffrax solver, with an SSP-RK3 start.

    yₙ₊₁ = yₙ + (h/12)(23 fₙ − 16 fₙ₋₁ + 5 fₙ₋₂), one evaluation of F per
    step. The two-level history lives in the diffrax solver state, and the
    first two steps are SSP-RK3 steps (a third-order bootstrap).

    Fixed, uniform steps only (``dt0=``, or ``StepTo`` with equal spacing):
    the weights assume a constant step and there is no error estimate, so an
    adaptive controller raises. Order 3; dense output is linear.

    Examples:
        >>> import diffrax as dfx
        >>> import jax.numpy as jnp
        >>> solution = dfx.diffeqsolve(
        ...     dfx.ODETerm(lambda t, y, args: -y),
        ...     AB3Solver(),
        ...     t0=0.0,
        ...     t1=1.0,
        ...     dt0=0.01,
        ...     y0=jnp.array(1.0),
        ... )
        >>> bool(jnp.allclose(solution.ys[0], jnp.exp(-1.0), rtol=1e-5))
        True
    """

    term_structure: ClassVar[Any] = dfx.AbstractTerm
    interpolation_cls: ClassVar[Any] = dfx.LocalLinearInterpolation

    def order(self, terms):
        return 3

    def init(self, terms, t0, t1, y0, args):
        # (fₙ₋₁, fₙ₋₂, number of steps taken)
        f0 = terms.vf(t0, y0, args)
        zeros = jax.tree.map(jnp.zeros_like, f0)
        return zeros, zeros, jnp.asarray(0, dtype=jnp.int32)

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del made_jump
        f_nm1, f_nm2, n_taken = solver_state
        dt = t1 - t0
        f_n = terms.vf(t0, y0, args)

        def _rk3(_):
            # SSP-RK3 bootstrap (reuses fₙ as its first stage)
            y1 = jax.tree.map(lambda y, f: y + dt * f, y0, f_n)
            k2 = terms.vf(t1, y1, args)
            y2 = jax.tree.map(
                lambda y, a, b: 0.75 * y + 0.25 * (a + dt * b), y0, y1, k2
            )
            k3 = terms.vf(t0 + 0.5 * dt, y2, args)
            return jax.tree.map(
                lambda y, a, b: y / 3.0 + (2.0 / 3.0) * (a + dt * b), y0, y2, k3
            )

        def _ab3(_):
            # fₙ is already evaluated, so hand ab3_step a constant right-hand side
            y_next, _, _ = ab3_step(y0, lambda _y: f_n, dt, f_nm1, f_nm2)
            return y_next

        y1 = jax.lax.cond(n_taken < 2, _rk3, _ab3, None)
        dense_info = dict(y0=y0, y1=y1)
        new_state = (f_n, f_nm1, n_taken + 1)
        return y1, None, dense_info, new_state, RESULTS.successful

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)


def leapfrog_raf_step(
    state: PyTree,
    state_nm1: PyTree,
    rhs_fn: Callable[[PyTree], PyTree],
    dt: ScalarLike,
    alpha: float = 0.05,
) -> tuple[PyTree, PyTree]:
    """One leapfrog step with a Robert–Asselin filter on the middle level.

    ỹₙ₊₁ = yₙ₋₁ + 2h f(yₙ) and ȳₙ = yₙ + α (yₙ₋₁ − 2yₙ + ỹₙ₊₁). The filtered
    ȳₙ damps the computational mode of the three-level scheme. Second order
    for α = 0 and first order for α > 0 (the filter's O(αh) damping; the
    measured rate is about 1.1 at the default 0.05).

    Args:
        state: The state yₙ, any pytree.
        state_nm1: The (filtered) state yₙ₋₁.
        rhs_fn: The autonomous right-hand side f(y).
        dt: The step h.
        alpha: The filter coefficient α, typically 0.01 to 0.1.

    Returns:
        ``(yₙ₊₁, ȳₙ)``: yₙ₊₁ is the next state and ȳₙ the next ``state_nm1``.

    Examples:
        >>> import jax.numpy as jnp
        >>> y_next, filtered = leapfrog_raf_step(1.0, 1.0, lambda y: 0.0 * y, 0.1)
        >>> bool(y_next == 1.0), bool(filtered == 1.0)
        (True, True)
    """
    rhs_n = rhs_fn(state)

    # ỹₙ₊₁ = yₙ₋₁ + 2h f(yₙ)
    y_next = jax.tree.map(
        lambda ynm1, fn: ynm1 + 2.0 * dt * fn,
        state_nm1,
        rhs_n,
    )

    # ȳₙ = yₙ + α (yₙ₋₁ − 2yₙ + ỹₙ₊₁)
    state_filtered = jax.tree.map(
        lambda ynm1, yn, ynp1: yn + alpha * (ynm1 - 2.0 * yn + ynp1),
        state_nm1,
        state,
        y_next,
    )

    return y_next, state_filtered


def _heun_bootstrap(
    rhs: Callable[..., PyTree],
    t0: ScalarLike,
    y0: PyTree,
    dt: ScalarLike,
    args: Any,
) -> tuple[PyTree, PyTree]:
    """One Heun (RK2) step for the multistep start: returns ``(f(t₀, y₀), y₁)``."""
    k1 = rhs(t0, y0, args)
    predictor = jax.tree.map(lambda y, f: y + dt * f, y0, k1)
    k2 = rhs(t0 + dt, predictor, args)
    # y₁ = y₀ + (h/2)(k₁ + k₂)
    y1 = jax.tree.map(lambda y, f1, f2: y + 0.5 * dt * (f1 + f2), y0, k1, k2)
    return k1, y1


class AB2Solver(AbstractStepper):
    """Second-order Adams–Bashforth, run by `integrix.march.run`.

    yₙ₊₁ = yₙ + (h/2)(3 fₙ − fₙ₋₁), one evaluation of the vector field per
    step. ``init`` bootstraps with one Heun step and keeps f(t₀, y₀) as the
    history; fixed, uniform steps only (variable steps break the weights).

    Examples:
        >>> import jax.numpy as jnp
        >>> from integrix.march import run
        >>> y = run(
        ...     AB2Solver(), lambda t, y, args: -y, 0.0, jnp.array(1.0), 0.01, 100
        ... )
        >>> bool(jnp.allclose(y, jnp.exp(-1.0), rtol=1e-4))
        True
    """

    @property
    def order(self) -> int:
        """Second order."""
        return 2

    def init(self, rhs, t0, y0, dt, args):
        """Heun to y₁; the history is fₙ₋₁ = f(t₀, y₀)."""
        f0, y1 = _heun_bootstrap(rhs, t0, y0, dt, args)
        return f0, y1

    def step(self, rhs, t, y, dt, state, args):
        """yₙ₊₁ = yₙ + (h/2)(3 fₙ − fₙ₋₁); the history becomes fₙ."""
        f_curr = rhs(t, y, args)
        y_next = jax.tree.map(
            lambda yi, fi, fi_1: yi + (dt / 2.0) * (3.0 * fi - fi_1),
            y,
            f_curr,
            state,
        )
        return y_next, f_curr


class LeapfrogRAFSolver(AbstractStepper):
    """Leapfrog with a Robert–Asselin filter, run by `integrix.march.run`.

    ỹₙ₊₁ = ȳₙ₋₁ + 2h F(tₙ, yₙ) and ȳₙ = yₙ + α (ȳₙ₋₁ − 2yₙ + ỹₙ₊₁): the filter
    damps the computational mode, which grows without it. Second order for
    α = 0 and first order for α > 0 (the Robert–Asselin degradation).
    ``init`` bootstraps with one Heun step and keeps y₀ as the history; fixed,
    uniform steps only.

    Attributes:
        alpha: The filter coefficient α ≥ 0, typically 0.01 to 0.1. It is an
            array leaf, so it can be traced, vmapped over or differentiated;
            a Python or NumPy value is checked when the stepper is built.

    Raises:
        ValidationError: If a concrete ``alpha`` is negative or not finite.

    Examples:
        >>> import jax.numpy as jnp
        >>> from integrix.march import run
        >>> stepper = LeapfrogRAFSolver(alpha=0.05)
        >>> y = run(
        ...     stepper, lambda t, y, args: -y, 0.0, jnp.array(1.0), 0.001, 1000
        ... )
        >>> bool(jnp.allclose(y, jnp.exp(-1.0), rtol=1e-3)), stepper.order
        (True, 1)
    """

    alpha: float | Float[Array, ""] = 0.05

    def __check_init__(self) -> None:
        if isinstance(self.alpha, jax.Array):
            return
        if not (math.isfinite(self.alpha) and self.alpha >= 0.0):
            raise ValidationError(
                f"'alpha' must be finite and non-negative, got {self.alpha}"
            )

    @property
    def order(self) -> int:
        """Second order without the filter, first order with it (or when traced)."""
        try:
            unfiltered = bool(self.alpha == 0.0)
        except jax.errors.ConcretizationTypeError:
            unfiltered = False
        return 2 if unfiltered else 1

    def init(self, rhs, t0, y0, dt, args):
        """Heun to y₁; the history is ȳ₀ = y₀."""
        _, y1 = _heun_bootstrap(rhs, t0, y0, dt, args)
        return y0, y1

    def step(self, rhs, t, y, dt, state, args):
        """Leapfrog from the filtered ȳₙ₋₁, then filter yₙ; the history becomes ȳₙ."""
        f_curr = rhs(t, y, args)
        # ỹₙ₊₁ = ȳₙ₋₁ + 2h F(tₙ, yₙ)
        y_next = jax.tree.map(lambda yp, fc: yp + 2.0 * dt * fc, state, f_curr)
        # ȳₙ = yₙ + α (ȳₙ₋₁ − 2yₙ + ỹₙ₊₁)
        y_filtered = jax.tree.map(
            lambda yp, yc, yn: yc + self.alpha * (yp - 2.0 * yc + yn),
            state,
            y,
            y_next,
        )
        return y_next, y_filtered
