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

from integrix._src.adjoint._checkpoint import Checkpointed, checkpointed_scan
from integrix._src.adjoint._diffrax import to_diffrax_adjoint
from integrix._src.adjoint._truncated import Truncated, truncated_scan
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
    adjoint: Any = None,
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
        adjoint: How reverse mode runs back through the march. ``None`` is
            plain reverse mode (every state stored for a stepper; diffrax's
            default for a solver). `integrix.adjoint.Checkpointed` stores
            every k-th state and recomputes between (for a diffrax solver,
            ``RecursiveCheckpointAdjoint(checkpoints=⌈N / k⌉)``).
            `integrix.adjoint.Truncated` cuts the gradient through the state
            before the last k steps, for a stepper only. A diffrax solver
            also takes a diffrax adjoint or a pipekit-cycle spec
            (`integrix.adjoint.to_diffrax_adjoint`). Forward values never
            depend on the adjoint.

    Returns:
        y_N, or the stacked y₁, …, y_N.

    Raises:
        ValidationError: If ``n_steps`` is not a positive integer, ``saveat``
            asks for anything but the final state or every step, ``adjoint``
            does not apply to the stepper (`Truncated` for a diffrax solve, a
            diffrax adjoint for an `AbstractStepper`), or ``stepper`` is
            neither kind.

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
    scan = _loop_scan(adjoint, n_steps)

    time_dtype = state_time_dtype(y0)
    start = jnp.asarray(t0, dtype=time_dtype)
    step_size = jnp.asarray(dt, dtype=time_dtype)

    state, y1 = stepper.init(rhs, start, y0, step_size, args)
    # y₁ as saved: its own step's gradient survives any truncation
    y1_saved = y1
    if isinstance(adjoint, Truncated) and adjoint.k < n_steps:
        # the first step lies before the last k, so the carry it hands on is cut
        y1, state = jax.lax.stop_gradient((y1, state))

    def body(carry, index):
        y, history = carry
        # tₙ = t₀ + n h
        t = start + index.astype(time_dtype) * step_size
        y_next, history = stepper.step(rhs, t, y, step_size, history, args)
        return (y_next, history), (y_next if save_steps else None)

    indices = jnp.arange(1, n_steps)
    (y_final, _), later = scan(body, (y1, state), indices)
    if not save_steps:
        return y_final
    return jax.tree.map(
        lambda first, rest: jnp.concatenate([first[None], rest]), y1_saved, later
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
    adjoint: Any,
) -> PyTree:
    """``diffeqsolve`` on the N + 1 step times t₀ + n h."""
    terms = rhs if isinstance(rhs, dfx.AbstractTerm) else dfx.ODETerm(rhs)
    time_dtype = state_time_dtype(y0)
    # tₙ = t₀ + n h, n = 0, …, N, in the state's precision as the loop has it
    start = jnp.asarray(t0, dtype=time_dtype)
    step_size = jnp.asarray(dt, dtype=time_dtype)
    step_times = start + step_size * jnp.arange(n_steps + 1, dtype=time_dtype)
    options = {} if adjoint is None else {"adjoint": _diffrax_adjoint(adjoint, n_steps)}
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


def _loop_scan(adjoint: Any, n_steps: int) -> Callable[..., Any]:
    """The scan the `AbstractStepper` loop runs, for the chosen adjoint."""
    if adjoint is None:
        return jax.lax.scan
    if isinstance(adjoint, Checkpointed):
        every = adjoint.every

        def scan(body, carry, xs):
            return checkpointed_scan(body, carry, xs, every=every)

        return scan
    if isinstance(adjoint, Truncated):
        # the loop scans N − 1 steps after init; the last k of N are live
        k = adjoint.k

        def scan(body, carry, xs):
            if k >= n_steps:
                return jax.lax.scan(body, carry, xs)
            return truncated_scan(body, carry, xs, k=k)

        return scan
    raise ValidationError(
        f"'adjoint' for an AbstractStepper must be None, Checkpointed or "
        f"Truncated, got {type(adjoint).__name__} (a diffrax adjoint applies to "
        f"a diffrax solver only)"
    )


def _diffrax_adjoint(adjoint: Any, n_steps: int) -> dfx.AbstractAdjoint:
    """The diffrax adjoint for a solver: Checkpointed maps, Truncated refuses."""
    if isinstance(adjoint, Truncated):
        raise ValidationError(
            "'adjoint' Truncated needs an AbstractStepper; a diffrax solver takes "
            "Checkpointed, a diffrax adjoint or a pipekit-cycle spec (truncation "
            "belongs to an outer rollout, not to a single solve)"
        )
    if isinstance(adjoint, Checkpointed):
        # diffrax counts checkpoints rather than spacing them: ⌈N / k⌉
        return dfx.RecursiveCheckpointAdjoint(checkpoints=-(-n_steps // adjoint.every))
    return to_diffrax_adjoint(adjoint)
