"""Split-explicit stepping: a fast mode subcycled inside each slow step.

Moved from finitevolX (``split_explicit.py`` and ``diffrax_solvers.py`` at
``82090a3``). The fast (barotropic, 2-D) mode takes M classical RK4 substeps
of h/M inside each slow (baroclinic, 3-D) step, its trapezoidal time mean
drives one forward-Euler step of the slow mode, and an optional coupling makes
the slow state consistent with that mean. `split_explicit_step` keeps
finitevolX's interface; `SplitExplicitRKSolver` is the `AbstractStepper` form,
run by `integrix.march.run` with its two vector fields in a
`SplitExplicitTerms`.

References:
    Shchepetkin, A. F. and McWilliams, J. C. (2005). The regional oceanic
    modeling system (ROMS): a split-explicit, free-surface,
    topography-following-coordinate oceanic model. Ocean Modelling 9,
    347–404.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree, ScalarLike

from integrix._src.march._erk import rk4_step
from integrix._src.march._loop import AbstractStepper, state_time_dtype
from integrix.exceptions import ValidationError


__all__ = [
    "SplitExplicitRKSolver",
    "SplitExplicitTerms",
    "split_explicit_step",
]


def _barotropic_cycle(
    state_2d: PyTree,
    rhs_2d: Callable[[ScalarLike, PyTree], PyTree],
    t0: ScalarLike,
    dt_fast: ScalarLike,
    n_substeps: int,
    forcing: PyTree | None = None,
) -> tuple[PyTree, PyTree]:
    """Subcycle the fast mode with RK4; return its end state and trapezoidal mean.

    yₖ₊₁ = RK4(yₖ) for k = 0, …, M − 1 on y′ = rhs_2d(t, y) + F, with F held
    fixed, and ⟨y⟩ = (y₀/2 + y₁ + … + y_{M−1} + y_M/2) / M, which is exact for
    a state linear in time.
    """
    if n_substeps < 1:
        raise ValidationError(f"n_substeps must be >= 1, got {n_substeps}")

    def _rhs(s):
        # s = (y, t): time rides along as a state with dt/dt = 1, so RK4
        # evaluates rhs_2d at its exact stage times.
        y, t = s
        f = rhs_2d(t, y)
        if forcing is not None:
            f = jax.tree.map(jnp.add, f, forcing)
        return f, jnp.ones_like(t)

    def _body(carry, _):
        y, t, acc = carry
        y_new, t_new = rk4_step((y, t), _rhs, dt_fast)
        acc = jax.tree.map(jnp.add, acc, y_new)
        return (y_new, t_new, acc), None

    # time in the state's float dtype, so a float32 state stays float32 in the
    # scan carry even with x64 on (and a time-dependent rhs_2d cannot promote it)
    t_dtype = state_time_dtype(state_2d)
    t_start = jnp.asarray(t0, dtype=t_dtype)
    # y₀/2
    acc0 = jax.tree.map(lambda y: 0.5 * y, state_2d)
    (y_end, _, acc), _ = jax.lax.scan(
        _body, (state_2d, t_start, acc0), None, length=n_substeps
    )
    # acc = y₀/2 + y₁ + … + y_M: subtract y_M/2, divide by M
    y_mean = jax.tree.map(lambda s, yn: (s - 0.5 * yn) / n_substeps, acc, y_end)
    return y_end, y_mean


def split_explicit_step(
    state_3d: PyTree,
    state_2d: PyTree,
    rhs_3d: Callable[[PyTree, PyTree], PyTree],
    rhs_2d: Callable[[ScalarLike, PyTree, PyTree], PyTree],
    couple_2d_to_3d: Callable[[PyTree, PyTree], PyTree],
    dt_slow: ScalarLike,
    n_substeps: int,
    slow_forcing: Callable[[PyTree], PyTree] | None = None,
    t0: ScalarLike = 0.0,
) -> tuple[PyTree, PyTree]:
    """One split-explicit barotropic/baroclinic step.

    The fast subcycle is neutrally stable for the oscillatory gravity-wave
    mode (RK4 amplification 1 − (ωh)⁶/144 per substep); the slow step is first
    order in h. The returned 2-D state is the end-of-cycle y_M, which stays on
    time; the slow step used the mean ⟨y⟩, and making the 3-D state consistent
    with it (replacing the depth mean of the 3-D velocity by the averaged
    transport, say) is ``couple_2d_to_3d``'s job.

    Algorithm:
        1. F = ``slow_forcing(state_3d)``, if given, held fixed over the cycle.
        2. Subcycle the 2-D state with M RK4 steps of h/M on
           y′ = ``rhs_2d(t, y, state_3d)`` + F, and form its trapezoidal
           mean ⟨y⟩ over the cycle.
        3. One forward-Euler step of the 3-D state:
           y₃ + h ``rhs_3d(state_3d, ⟨y⟩)``.
        4. ``couple_2d_to_3d(new state_3d, ⟨y⟩)``.

    Args:
        state_3d: The slow (baroclinic) state.
        state_2d: The fast (barotropic) state.
        rhs_3d: The slow tendency ``rhs_3d(state_3d, mean_2d)``.
        rhs_2d: The fast tendency ``rhs_2d(t, state_2d, state_3d)``, at the
            absolute RK4 stage time t.
        couple_2d_to_3d: ``couple_2d_to_3d(state_3d, mean_2d)``, applied after
            the slow step.
        dt_slow: The slow step h.
        n_substeps: The number of fast substeps M ≥ 1 per slow step.
        slow_forcing: ``slow_forcing(state_3d)``, the baroclinic forcing of the
            barotropic mode, evaluated once per slow step and added to every
            fast tendency.
        t0: The time at the start of the slow step.

    Returns:
        ``(new_state_3d, new_state_2d)``.

    Raises:
        ValidationError: If ``n_substeps`` is below 1.

    Examples:
        >>> import jax.numpy as jnp
        >>> slow, fast = split_explicit_step(
        ...     jnp.array(1.0),
        ...     jnp.array(1.0),
        ...     lambda y3, mean_2d: -mean_2d,
        ...     lambda t, y2, y3: 0.0 * y2,
        ...     lambda y3, mean_2d: y3,
        ...     0.1,
        ...     5,
        ... )
        >>> bool(jnp.allclose(slow, 0.9)), bool(fast == 1.0)
        (True, True)
    """
    if n_substeps < 1:
        raise ValidationError(f"n_substeps must be >= 1, got {n_substeps}")

    dt_fast = dt_slow / n_substeps
    forcing = None if slow_forcing is None else slow_forcing(state_3d)

    # --- fast (barotropic) subcycle ---
    y_2d_end, y_2d_avg = _barotropic_cycle(
        state_2d,
        lambda t_sub, y_2d: rhs_2d(t_sub, y_2d, state_3d),
        t0,
        dt_fast,
        n_substeps,
        forcing,
    )

    # --- slow (baroclinic) step ---
    f_slow = rhs_3d(state_3d, y_2d_avg)
    y_3d_new = jax.tree.map(lambda y, f: y + dt_slow * f, state_3d, f_slow)

    # --- coupling ---
    y_3d_new = couple_2d_to_3d(y_3d_new, y_2d_avg)

    return y_3d_new, y_2d_end


class SplitExplicitTerms(eqx.Module):
    """The vector fields of a split-explicit model, passed to `run` as ``rhs``.

    Attributes:
        slow: The slow tendency ``slow(t, y_3d, mean_2d, args)``.
        fast: The fast tendency ``fast(t, y_2d, y_3d, args)``, at the
            absolute RK4 stage time t.
        slow_forcing: ``slow_forcing(t, y_3d, args)``, the baroclinic forcing
            of the barotropic mode, held fixed over the cycle; ``None`` for
            none.
        couple: ``couple(y_3d, mean_2d, args)``, applied after the slow step;
            ``None`` leaves the slow state as it is.

    Examples:
        >>> terms = SplitExplicitTerms(
        ...     slow=lambda t, y3, mean_2d, args: -y3,
        ...     fast=lambda t, y2, y3, args: -y2,
        ... )
        >>> terms.couple is None
        True
    """

    slow: Callable[..., PyTree]
    fast: Callable[..., PyTree]
    slow_forcing: Callable[..., PyTree] | None = None
    couple: Callable[..., PyTree] | None = None


class SplitExplicitRKSolver(AbstractStepper):
    """Split-explicit stepping as an `AbstractStepper`, for `integrix.march.run`.

    The state is the pair y = (y₃, y₂) of the slow and fast modes, and ``rhs``
    is a `SplitExplicitTerms`. Each step is `split_explicit_step` with the
    fast mode in M RK4 substeps; M follows the external-wave CFL limit. The
    scheme carries no history, so ``init`` is a plain first step. First order
    in the slow step.

    Attributes:
        n_substeps: The number of fast substeps M ≥ 1 per slow step.

    Raises:
        ValidationError: If ``n_substeps`` is below 1.

    Examples:
        >>> import jax.numpy as jnp
        >>> from integrix.march import run
        >>> terms = SplitExplicitTerms(
        ...     slow=lambda t, y3, mean_2d, args: -mean_2d,
        ...     fast=lambda t, y2, y3, args: 0.0 * y2,
        ... )
        >>> slow, fast = run(
        ...     SplitExplicitRKSolver(n_substeps=5),
        ...     terms,
        ...     0.0,
        ...     (jnp.array(1.0), jnp.array(1.0)),
        ...     0.1,
        ...     10,
        ... )
        >>> bool(jnp.allclose(slow, 0.0, atol=1e-6)), bool(fast == 1.0)
        (True, True)
    """

    n_substeps: int = eqx.field(static=True, default=50)

    def __check_init__(self) -> None:
        if isinstance(self.n_substeps, bool):
            raise ValidationError("'n_substeps' must be an integer, got a bool")
        try:
            count = operator.index(self.n_substeps)
        except TypeError as err:
            raise ValidationError(
                f"'n_substeps' must be an integer, got {self.n_substeps!r}"
            ) from err
        if count < 1:
            raise ValidationError(f"'n_substeps' must be at least 1, got {count}")

    @property
    def order(self) -> int:
        """First order in the slow step."""
        return 1

    def init(self, rhs, t0, y0, dt, args):
        """No history to bootstrap: the first step is an ordinary one."""
        y1, _ = self.step(rhs, t0, y0, dt, None, args)
        return None, y1

    def step(self, rhs, t, y, dt, state, args):
        """One `split_explicit_step` of the pair (y₃, y₂)."""
        terms = _as_terms(rhs)
        state_3d, state_2d = y
        slow_forcing = None
        if terms.slow_forcing is not None:
            forcing_fn = terms.slow_forcing

            def slow_forcing(y3: PyTree) -> PyTree:
                return forcing_fn(t, y3, args)

        couple = terms.couple

        def couple_2d_to_3d(y3: PyTree, mean_2d: PyTree) -> PyTree:
            return y3 if couple is None else couple(y3, mean_2d, args)

        new_3d, new_2d = split_explicit_step(
            state_3d,
            state_2d,
            lambda y3, mean_2d: terms.slow(t, y3, mean_2d, args),
            lambda t_sub, y2, y3: terms.fast(t_sub, y2, y3, args),
            couple_2d_to_3d,
            dt,
            self.n_substeps,
            slow_forcing=slow_forcing,
            t0=t,
        )
        return (new_3d, new_2d), state


def _as_terms(rhs: Any) -> SplitExplicitTerms:
    """Raise `ValidationError` unless ``rhs`` is a `SplitExplicitTerms`."""
    if not isinstance(rhs, SplitExplicitTerms):
        raise ValidationError(
            f"'rhs' must be a SplitExplicitTerms for a split-explicit stepper, "
            f"got {type(rhs).__name__}"
        )
    return rhs
