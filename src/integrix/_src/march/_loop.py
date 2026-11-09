"""`AbstractStepper` and `run`: a fixed-step loop for schemes that carry history.

Design decision 8 (#18): single-step schemes stay diffrax solvers; schemes
that carry state between steps (multistep, filtered leapfrog, split-explicit)
are `AbstractStepper`s, run by this `jax.lax.scan` loop with their history in
the loop carry, never on the module. `run` also accepts a diffrax solver and
hands it to `diffrax.diffeqsolve`, so one entry point covers every stepper.
"""

from __future__ import annotations

import abc
import operator
from collections.abc import Callable
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree, ScalarLike

from integrix.exceptions import ValidationError


__all__ = [
    "AbstractStepper",
    "run",
    "state_time_dtype",
]


def state_time_dtype(state: PyTree) -> Any:
    """The real float dtype that time takes alongside ``state``.

    Time in the state's own float precision keeps a float32 state float32
    even when the vector field depends on t (and float64 stays float64); a
    complex state gets real time of the same precision. The
    promotion runs over the leaves' dtypes, not the leaves, because a weakly
    typed leaf such as ``jnp.array(1.0)`` would otherwise lose to float32.
    """
    dtypes = [jnp.asarray(leaf).dtype for leaf in jax.tree.leaves(state)]
    promoted = jnp.result_type(*dtypes, jnp.float32)
    # a complex state keeps real time, in its own precision
    return jnp.finfo(promoted).dtype


class AbstractStepper(eqx.Module):
    """A fixed-step scheme whose history lives in the loop carry.

    A stepper takes the diffrax vector field ``rhs(t, y, args)``, so a model
    writes its tendencies once for both kinds of stepper. ``init`` takes the
    first step, bootstrapping whatever history the scheme needs, and ``step``
    takes every later one; neither mutates the module.

    Examples:
        >>> import jax.numpy as jnp
        >>> class Euler(AbstractStepper):
        ...     @property
        ...     def order(self):
        ...         return 1
        ...
        ...     def init(self, rhs, t0, y0, dt, args):
        ...         return None, y0 + dt * rhs(t0, y0, args)
        ...
        ...     def step(self, rhs, t, y, dt, state, args):
        ...         return y + dt * rhs(t, y, args), state
        >>> y = run(Euler(), lambda t, y, args: -y, 0.0, jnp.array(1.0), 0.1, 10)
        >>> bool(jnp.allclose(y, 0.9**10))
        True
    """

    @property
    @abc.abstractmethod
    def order(self) -> int:
        """The global order of accuracy p, error O(hᵖ)."""

    @abc.abstractmethod
    def init(
        self,
        rhs: Any,
        t0: ScalarLike,
        y0: PyTree,
        dt: ScalarLike,
        args: Any,
    ) -> tuple[PyTree, PyTree]:
        """Take the first step from (t₀, y₀): returns ``(state, y₁)``."""

    @abc.abstractmethod
    def step(
        self,
        rhs: Any,
        t: ScalarLike,
        y: PyTree,
        dt: ScalarLike,
        state: PyTree,
        args: Any,
    ) -> tuple[PyTree, PyTree]:
        """Take one step from (tₙ, yₙ) with the history: returns ``(yₙ₊₁, state)``."""


def run(
    stepper: AbstractStepper | dfx.AbstractSolver,
    rhs: Callable[..., PyTree] | Any,
    t0: ScalarLike,
    y0: PyTree,
    dt: ScalarLike,
    n_steps: int,
    *,
    args: Any = None,
    saveat: dfx.SaveAt | None = None,
    adjoint: dfx.AbstractAdjoint | None = None,
) -> PyTree:
    """March y₀ through ``n_steps`` constant steps of h from t₀.

    yₙ₊₁ = Φ(yₙ, tₙ, h) for n = 0, …, N − 1, with tₙ = t₀ + n h.

    Algorithm:
        1. A diffrax solver goes to `diffrax.diffeqsolve` on the step times
           t₀ + n h (`diffrax.StepTo`), so it takes exactly N steps.
        2. An `AbstractStepper` takes its first step with ``init`` and the
           other N − 1 with ``step`` inside `jax.lax.scan`, carrying the
           state and its history.

    Args:
        stepper: An `AbstractStepper`, or a diffrax solver (the steppers of
            `integrix.march`, or diffrax's own).
        rhs: The vector field ``rhs(t, y, args)``. A diffrax solver may also
            take a diffrax term (a `diffrax.MultiTerm` for IMEX, say); a
            split-explicit stepper takes its `SplitExplicitTerms`.
        t0: The start time t₀.
        y0: The initial state y₀, any pytree.
        dt: The step h.
        n_steps: The number of steps N ≥ 1, a Python integer.
        args: Passed to the vector field.
        saveat: ``None`` for the final state y_N, or ``SaveAt(steps=True)``
            for every state y₁, …, y_N, stacked on a new leading axis.
        adjoint: A diffrax adjoint, for a diffrax solver only. The loop of an
            `AbstractStepper` differentiates by reverse-mode through
            `jax.lax.scan`; integrix's checkpointed and truncated adjoints
            for it arrive with #22 and #23.

    Returns:
        y_N, or the stacked y₁, …, y_N.

    Raises:
        ValidationError: If ``n_steps`` is not a positive integer, ``saveat``
            asks for anything but the final state or every step, ``adjoint``
            is given for an `AbstractStepper`, or ``stepper`` is neither kind.

    Examples:
        >>> import jax.numpy as jnp
        >>> from integrix.march import RK4Classic
        >>> y = run(
        ...     RK4Classic(), lambda t, y, args: -y, 0.0, jnp.array(1.0), 0.1, 10
        ... )
        >>> bool(jnp.allclose(y, jnp.exp(-1.0), rtol=1e-6))
        True
        >>> from integrix.march import AB2Solver
        >>> every = run(
        ...     AB2Solver(),
        ...     lambda t, y, args: -y,
        ...     0.0,
        ...     jnp.ones(3),
        ...     0.1,
        ...     5,
        ...     saveat=dfx.SaveAt(steps=True),
        ... )
        >>> every.shape
        (5, 3)
    """
    try:
        n_steps = operator.index(n_steps)
    except TypeError as err:
        raise ValidationError(
            f"'n_steps' must be a Python integer, got {type(n_steps).__name__}"
        ) from err
    if n_steps < 1:
        raise ValidationError(f"'n_steps' must be at least 1, got {n_steps}")
    save_steps = _save_steps(saveat)

    if isinstance(stepper, dfx.AbstractSolver):
        return _run_diffrax(
            stepper, rhs, t0, y0, dt, n_steps, args, save_steps, adjoint
        )
    if not isinstance(stepper, AbstractStepper):
        raise ValidationError(
            f"'stepper' must be an AbstractStepper or a diffrax solver, got "
            f"{type(stepper).__name__}"
        )
    if adjoint is not None:
        raise ValidationError(
            "'adjoint' applies to a diffrax solver only; an AbstractStepper's "
            "loop differentiates through jax.lax.scan (integrix's checkpointed "
            "and truncated adjoints arrive with #22 and #23)"
        )

    time_dtype = state_time_dtype(y0)
    start = jnp.asarray(t0, dtype=time_dtype)
    step_size = jnp.asarray(dt, dtype=time_dtype)

    state, y1 = stepper.init(rhs, start, y0, step_size, args)

    def body(carry, index):
        y, history = carry
        # tₙ = t₀ + n h
        t = start + index.astype(time_dtype) * step_size
        y_next, history = stepper.step(rhs, t, y, step_size, history, args)
        return (y_next, history), (y_next if save_steps else None)

    indices = jnp.arange(1, n_steps)
    (y_final, _), later = jax.lax.scan(body, (y1, state), indices)
    if not save_steps:
        return y_final
    return jax.tree.map(
        lambda first, rest: jnp.concatenate([first[None], rest]), y1, later
    )


def _save_steps(saveat: dfx.SaveAt | None) -> bool:
    """Whether ``saveat`` asks for every step (else only the final state)."""
    if saveat is None:
        return False
    subs = saveat.subs
    extras = (
        saveat.dense or saveat.solver_state or saveat.controller_state
    ) or saveat.made_jump
    plain = (
        isinstance(subs, dfx.SubSaveAt)
        and subs.ts is None
        and not subs.t0
        and subs.fn is _SAVE_Y
        and not extras
    )
    if plain and subs.steps:
        return True
    if plain and subs.t1:
        return False
    raise ValidationError(
        "'saveat' must be None (the final state), SaveAt(t1=True) or "
        "SaveAt(steps=True) (every step), with no fn, dense output or saved "
        "solver or controller state"
    )


# diffrax's default SubSaveAt.fn, which saves y itself
_SAVE_Y = dfx.SaveAt(t1=True).subs.fn


def _run_diffrax(
    solver: dfx.AbstractSolver,
    rhs: Any,
    t0: ScalarLike,
    y0: PyTree,
    dt: ScalarLike,
    n_steps: int,
    args: Any,
    save_steps: bool,
    adjoint: dfx.AbstractAdjoint | None,
) -> PyTree:
    """``diffeqsolve`` on the N + 1 step times t₀ + n h."""
    terms = rhs if isinstance(rhs, dfx.AbstractTerm) else dfx.ODETerm(rhs)
    time_dtype = state_time_dtype(y0)
    # tₙ = t₀ + n h, n = 0, …, N, in the state's precision as the loop has it
    start = jnp.asarray(t0, dtype=time_dtype)
    step_size = jnp.asarray(dt, dtype=time_dtype)
    step_times = start + step_size * jnp.arange(n_steps + 1, dtype=time_dtype)
    options = {} if adjoint is None else {"adjoint": adjoint}
    solution = dfx.diffeqsolve(
        terms,
        solver,
        t0=step_times[0],
        t1=step_times[-1],
        dt0=None,
        y0=y0,
        args=args,
        saveat=dfx.SaveAt(steps=save_steps, t1=not save_steps),
        stepsize_controller=dfx.StepTo(ts=step_times),
        max_steps=n_steps,
        **options,
    )
    if save_steps:
        return solution.ys
    return jax.tree.map(lambda saved: saved[0], solution.ys)
