"""Semi-Lagrangian advection: departure points by integration, then a read-out.

A tracer q(t, x) carried by a velocity u(t, x) is constant along the
characteristics Ẋ = u(t, X), so q(t + Δt, x) = q(t, x_d), where x_d is the
point that reaches the node x after Δt. integrix integrates the
characteristic backward to x_d (the trajectory is the integral); the caller's
read-out, any functax interpolation plan, evaluates the old field there. The
scheme has no grid: the nodes, the velocity and the read-out come in as
arguments.

References:
    Staniforth, A. and Côté, J. (1991). Semi-Lagrangian integration schemes
    for atmospheric models—a review. Mon. Weather Rev. 119(9), 2206–2223.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import RESULTS
from jaxtyping import Array, Float, PyTree, ScalarLike

from integrix.exceptions import ValidationError


__all__ = [
    "SemiLagrangianSolver",
    "SemiLagrangianTerm",
    "departure_points",
    "semi_lagrangian_step",
]


Velocity = Callable[[ScalarLike, Float[Array, "node axis"]], Float[Array, "node axis"]]
"""The velocity u(t, x) at N points x in D dimensions."""

Readout = Callable[[Float[Array, "node axis"]], Callable[[PyTree], PyTree]]
"""Departure points x_d ↦ the map q ↦ q(x_d) (a functax plan, say)."""


def _check_iterations(iterations: int) -> None:
    """Raise unless ``iterations`` is a non-negative integer.

    Args:
        iterations: The number of midpoint iterations.

    Raises:
        ValidationError: If it is not a non-negative ``int``.
    """
    if isinstance(iterations, bool) or not isinstance(iterations, int):
        raise ValidationError(
            f"'iterations' must be a non-negative int, got {iterations!r}"
        )
    if iterations < 0:
        raise ValidationError(f"'iterations' must be non-negative, got {iterations}")


def departure_points(
    nodes: Float[Array, "node axis"],
    velocity: Velocity,
    t: ScalarLike,
    dt: ScalarLike,
    *,
    iterations: int = 2,
) -> Float[Array, "node axis"]:
    """The points x_d that the flow carries to the nodes x over one step.

        x_d = x − Δt u(t + Δt/2, (x + x_d)/2)

    The midpoint rule on the trajectory, solved by fixed-point iteration
    from one Euler step. One iteration or more makes x_d second order in
    Δt; ``iterations=0`` is the first-order Euler departure point
    x_d = x − Δt u(t, x).

    Algorithm:
        1. x_d⁽⁰⁾ = x − Δt u(t, x).
        2. For k = 0, …, K − 1: x_d⁽ᵏ⁺¹⁾ = x − Δt u(t + Δt/2, (x + x_d⁽ᵏ⁾)/2).

    Args:
        nodes: The arrival points x, N of them in D dimensions.
        velocity: The velocity u(t, x), evaluated at N points at once.
        t: The time t at the start of the step.
        dt: The step Δt (negative steps backward).
        iterations: The number K of midpoint iterations.

    Returns:
        The departure points x_d, shaped like ``nodes``.

    Raises:
        ValidationError: If ``iterations`` is not a non-negative int.

    Examples:
        A uniform flow moves every node back by u Δt:

        >>> import jax.numpy as jnp
        >>> nodes = jnp.array([[0.0], [0.5]])
        >>> x_d = departure_points(nodes, lambda t, x: jnp.ones_like(x), 0.0, 0.1)
        >>> bool(jnp.allclose(x_d, jnp.array([[-0.1], [0.4]])))
        True

        A solid-body rotation by a quarter turn, from the point (1, 0); the
        midpoint departure point is close to the exact (0, −1):

        >>> def rotation(t, x):
        ...     return jnp.column_stack([-x[:, 1], x[:, 0]])
        >>> start = jnp.array([[1.0, 0.0]])
        >>> euler = departure_points(start, rotation, 0.0, 0.1, iterations=0)
        >>> midpoint = departure_points(start, rotation, 0.0, 0.1)
        >>> exact = jnp.array([[jnp.cos(0.1), -jnp.sin(0.1)]])
        >>> bool(
        ...     jnp.abs(midpoint - exact).max() < jnp.abs(euler - exact).max() / 10
        ... )
        True
    """
    _check_iterations(iterations)
    # x_d⁽⁰⁾ = x − Δt u(t, x)
    departure = nodes - dt * velocity(t, nodes)
    midtime = t + 0.5 * dt
    for _ in range(iterations):
        # (x + x_d⁽ᵏ⁾)/2
        midpoint = 0.5 * (nodes + departure)
        # x_d⁽ᵏ⁺¹⁾ = x − Δt u(t + Δt/2, (x + x_d⁽ᵏ⁾)/2)
        departure = nodes - dt * velocity(midtime, midpoint)
    return departure


def semi_lagrangian_step(
    field: PyTree,
    nodes: Float[Array, "node axis"],
    velocity: Velocity,
    readout: Readout,
    t: ScalarLike,
    dt: ScalarLike,
    *,
    iterations: int = 2,
) -> PyTree:
    """One semi-Lagrangian step: the old field read at the departure points.

        q(t + Δt, x) = q(t, x_d)

    There is no CFL limit, since the step only reads the old field. A
    convex read-out (linear) stays within the old extrema; a cubic or
    B-spline read-out can over- and undershoot, which a monotone plan
    (functax's ``Pchip``) avoids. Departure points outside the domain are
    the read-out's business (its periodic axes or boundary policy).

    Algorithm:
        1. x_d = `departure_points` (nodes, u, t, Δt).
        2. H = readout(x_d), the read-out of the old field at x_d.
        3. q(t + Δt) = H q(t).

    Args:
        field: The old field q(t), in whatever layout ``readout`` reads.
        nodes: The arrival points x, N of them in D dimensions.
        velocity: The velocity u(t, x), evaluated at N points at once.
        readout: Departure points ↦ the map q ↦ q(x_d); a functax
            interpolation plan, ``lambda x_d: scheme.plan(grid, Points(x_d))``.
        t: The time t at the start of the step.
        dt: The step Δt.
        iterations: The number of midpoint iterations of the trajectory.

    Returns:
        The new field q(t + Δt), as the read-out returns it.

    Raises:
        ValidationError: If ``iterations`` is not a non-negative int.

    Examples:
        A periodic sine translated by a quarter period with a cubic
        B-spline read-out:

        >>> import einx
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis, Points
        >>> from functax.interp import BSpline
        >>> x = jnp.arange(32) / 32
        >>> axis = Axis(x, period=1.0)
        >>> def readout(x_d):
        ...     return BSpline(3).plan(axis, Points(x_d))
        >>> nodes = einx.id("node -> node 1", x)
        >>> q = jnp.sin(2 * jnp.pi * x)
        >>> q_new = semi_lagrangian_step(
        ...     q, nodes, lambda t, p: jnp.ones_like(p), readout, 0.0, 0.25
        ... )
        >>> bool(jnp.allclose(q_new, jnp.sin(2 * jnp.pi * (x - 0.25)), atol=1e-4))
        True
    """
    # x_d
    departure = departure_points(nodes, velocity, t, dt, iterations=iterations)
    # q(t + Δt, x) = q(t, x_d)
    return readout(departure)(field)


class SemiLagrangianTerm(dfx.AbstractTerm):
    """The characteristic equation Ẋ = u(t, X), as a diffrax term.

    `SemiLagrangianSolver` integrates the characteristics backward from its
    nodes by calling ``velocity(t, x)`` at N points x itself. The field has
    no tendency, so ``vf(t, y, args)`` is a zero field shaped like y: it is
    what `diffrax.diffeqsolve` evaluates on ``y0`` to check the term, and a
    velocity that only reads N points (a functax plan, say) never sees the
    field.

    Attributes:
        velocity: The velocity u(t, x), evaluated at N points at once.

    Examples:
        >>> import jax.numpy as jnp
        >>> term = SemiLagrangianTerm(lambda t, x: 2.0 * x)
        >>> term.velocity(0.0, jnp.array([[1.0]])).tolist()
        [[2.0]]
        >>> term.vf(0.0, jnp.ones(3), None).tolist()
        [0.0, 0.0, 0.0]
    """

    velocity: Velocity

    def vf(self, t, y, args):
        return jax.tree.map(jnp.zeros_like, y)

    def contr(self, t0, t1, **kwargs):
        del kwargs
        return t1 - t0

    def prod(self, vf, control):
        return jax.tree.map(lambda v: v * control, vf)


class SemiLagrangianSolver(dfx.AbstractSolver):
    """Semi-Lagrangian advection as a diffrax solver.

        q(tₙ₊₁, x) = q(tₙ, x_d),   x_d = x − h u(tₙ + h/2, (x + x_d)/2)

    Run it under `diffrax.diffeqsolve` with a `SemiLagrangianTerm` and the
    field as ``y0``. Order 2 in time with one midpoint iteration or more,
    order 1 with none (the read-out's spatial order is separate). There is
    no error estimate, so it runs with fixed steps only, and dense output is
    linear between steps.

    Attributes:
        nodes: The arrival points x, N of them in D dimensions.
        readout: Departure points ↦ the map q ↦ q(x_d).
        iterations: The number of midpoint iterations of the trajectory.

    Raises:
        ValidationError: If ``iterations`` is not a non-negative int.

    Examples:
        Two steps of a uniform flow with a linear read-out, against the
        exact translation:

        >>> import diffrax as dfx
        >>> import einx
        >>> import jax.numpy as jnp
        >>> from functax.geometry import Axis, Points
        >>> from functax.interp import Linear
        >>> x = jnp.arange(8) / 8
        >>> axis = Axis(x, period=1.0)
        >>> solver = SemiLagrangianSolver(
        ...     nodes=einx.id("node -> node 1", x),
        ...     readout=lambda x_d: Linear().plan(axis, Points(x_d)),
        ... )
        >>> term = SemiLagrangianTerm(lambda t, p: jnp.ones_like(p))
        >>> q0 = jnp.cos(2 * jnp.pi * x)
        >>> solution = dfx.diffeqsolve(term, solver, 0.0, 0.25, 0.125, q0)
        >>> bool(jnp.allclose(solution.ys[0], jnp.roll(q0, 2)))
        True

    References:
        Staniforth, A. and Côté, J. (1991). Semi-Lagrangian integration
        schemes for atmospheric models—a review. Mon. Weather Rev. 119(9),
        2206–2223.
    """

    nodes: Float[Array, "node axis"]
    readout: Readout
    iterations: int = eqx.field(static=True, default=2)

    term_structure: ClassVar[Any] = SemiLagrangianTerm
    interpolation_cls: ClassVar[Any] = dfx.LocalLinearInterpolation

    def __check_init__(self) -> None:
        _check_iterations(self.iterations)

    def order(self, terms):
        return 2 if self.iterations > 0 else 1

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        # diffeqsolve wraps the term with its time direction (diffrax's
        # WrapTerm, whose vf evaluates at direction · t); read the velocity
        # from the term inside and step in real time
        term = getattr(terms, "term", terms)
        direction = getattr(terms, "direction", 1)
        # q(tₙ₊₁, x) = q(tₙ, x_d)
        y1 = semi_lagrangian_step(
            y0,
            self.nodes,
            term.velocity,
            self.readout,
            direction * t0,
            direction * (t1 - t0),
            iterations=self.iterations,
        )
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func(self, terms, t0, y0, args):
        raise NotImplementedError(
            "SemiLagrangianSolver has no tendency of the field to evaluate; "
            "pass dt0 to diffeqsolve instead of letting it pick one."
        )
