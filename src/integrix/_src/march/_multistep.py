"""Adams–Bashforth steppers: AB3 as a diffrax solver, and the pure AB3 step.

Moved from finitevolX (``diffrax_solvers.py`` and ``multistep.py`` at
``82090a3``) with their interfaces unchanged. AB3 keeps its history in the
diffrax solver state, so `diffrax.diffeqsolve` runs it.

References:
    Durran, D. R. (2010). Numerical Methods for Fluid Dynamics, 2nd ed.
    Springer.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import diffrax as dfx
import jax
import jax.numpy as jnp
from diffrax import RESULTS
from jaxtyping import PyTree, ScalarLike


__all__ = [
    "AB3Solver",
    "ab3_step",
]


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
