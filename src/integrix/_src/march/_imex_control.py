"""A matrix-free IMEX solver, its step-size control, and the explicit CFL bound.

Moved from sesmax-core (``sesmax_core/_src/solvers.py`` at sesmax
``f08f59d``, itself copied from somax), which keeps its model terms. The
signatures are sesmax-core's. One behaviour changes: `stable_step` takes
arrays and traces, returning a JAX scalar, where sesmax-core's worked on
Python floats (sesmax#116 records the change).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optimistix as optx
from jaxtyping import Array, ArrayLike, Bool, Float

from integrix.exceptions import ValidationError


__all__ = [
    "imex_solver",
    "imex_stepsize_controller",
    "stable_step",
]


def imex_solver(
    *,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    gmres_restart: int = 20,
) -> dfx.KenCarp3:
    """A ``KenCarp3`` IMEX solver whose implicit stage is matrix-free.

    For y′ = F_E(y) + F_I(y) with F_I stiff, each implicit stage solves
    (I − hγ J_I) Δ = r by Newton (optimistix) with a GMRES inner solve
    (lineax), so the Jacobian J_I is never formed: memory O(N) rather than the
    O(N²) dense Jacobian of diffrax's default root finder, which runs out of
    memory on a 256 × 256 field. Pair it with `imex_stepsize_controller`
    carrying the same tolerances: a fixed step cannot reject a step whose
    Newton solve failed.

    Args:
        rtol: Relative tolerance of the Newton solve (and of GMRES).
        atol: Absolute tolerance of the Newton solve (and of GMRES).
        gmres_restart: The Krylov subspace size before GMRES restarts.

    Returns:
        A `diffrax.KenCarp3` for a `diffrax.MultiTerm` of an explicit and an
        implicit term.

    Examples:
        >>> import diffrax as dfx
        >>> import jax.numpy as jnp
        >>> terms = dfx.MultiTerm(
        ...     dfx.ODETerm(lambda t, y, args: -y),
        ...     dfx.ODETerm(lambda t, y, args: -50.0 * y),
        ... )
        >>> solution = dfx.diffeqsolve(
        ...     terms,
        ...     imex_solver(rtol=1e-5, atol=1e-7),
        ...     0.0,
        ...     1.0,
        ...     0.01,
        ...     jnp.ones(4),
        ...     stepsize_controller=imex_stepsize_controller(rtol=1e-5, atol=1e-7),
        ... )
        >>> bool(jnp.allclose(solution.ys[0], jnp.exp(-51.0), atol=1e-5))
        True
    """
    root_finder = optx.Newton(
        rtol=rtol,
        atol=atol,
        linear_solver=lx.GMRES(rtol=rtol, atol=atol, restart=gmres_restart),
    )
    return dfx.KenCarp3(root_finder=root_finder)


def imex_stepsize_controller(
    *,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> dfx.PIDController:
    """An adaptive PID step-size controller for an IMEX solve.

    A fixed-step controller with an implicit solver needs the implicit
    tolerances set and cannot reject a step when the Newton solve fails to
    converge; an adaptive controller is the fix. Match its tolerances to the
    ones given to `imex_solver`.

    Args:
        rtol: Relative tolerance of the step-size control.
        atol: Absolute tolerance of the step-size control.

    Returns:
        A `diffrax.PIDController`.

    Examples:
        >>> controller = imex_stepsize_controller(rtol=1e-6, atol=1e-8)
        >>> controller.rtol, controller.atol
        (1e-06, 1e-08)
    """
    return dfx.PIDController(rtol=rtol, atol=atol)


def stable_step(
    spacing: Sequence[float] | ArrayLike,
    *,
    speed: Sequence[float | complex] | ArrayLike = (),
    diffusivity: Sequence[float] | ArrayLike = (),
    cfl: float | ArrayLike = 1.0,
) -> Float[Array, ""]:
    """The largest stable forward-Euler step for advection and diffusion together.

    h_max = CFL / (∑ᵢ |uᵢ| / Δxᵢ + 2 ∑ᵢ Kᵢ / Δxᵢ²). Advection and diffusion
    act in the same explicit step, so their rates add; taking the smaller of
    two separate bounds would allow up to twice this step.

    This is a first-order, forward-Euler estimate that catches a step orders
    of magnitude too large. It is not a stability certificate for a
    high-order reconstruction or a Runge–Kutta scheme, whose stability
    regions differ.

    Args:
        spacing: The grid spacing Δxᵢ along each axis.
        speed: The largest speed |uᵢ| along each axis (its sign is ignored),
            or empty for no advection.
        diffusivity: The largest diffusivity Kᵢ along each axis, or empty for
            no diffusion.
        cfl: The safety factor on the bound.

    Returns:
        The bound, a JAX scalar; ``inf`` when nothing moves or diffuses.

    Raises:
        ValidationError: If ``speed`` or ``diffusivity`` is neither empty nor
            one value per axis, or, for Python or NumPy inputs, if a spacing
            or ``cfl`` is not finite and positive, or a speed or diffusivity
            is not finite or a diffusivity is negative. JAX-array inputs are
            checked with `equinox.error_if`, so the same checks hold under
            `jax.jit`.

    Examples:
        Advection only, along one axis of two:

        >>> bool(stable_step((0.5, 1.0), speed=(1.0, 0.0)) == 0.5)
        True

        Advection and diffusion add, so the bound is below each alone:

        >>> bool(stable_step((1.0,), speed=(1.0,), diffusivity=(0.5,)) == 0.5)
        True

        Nothing moves or diffuses:

        >>> bool(jnp.isinf(stable_step((1.0, 1.0))))
        True
    """
    # the dtype comes from the JAX arguments; host values are cast to it
    dtype = _working_dtype(spacing, speed, diffusivity, cfl)
    spacings = _as_vector(spacing, "spacing", dtype)
    # a speed's sign (or phase) is ignored: |uᵢ|
    speeds = _as_vector(speed, "speed", dtype, magnitude=True)
    diffusivities = _as_vector(diffusivity, "diffusivity", dtype)
    cfl_value = _as_vector(cfl, "cfl", dtype, scalar=True)
    n_axes = spacings.shape[0]
    for name, values in (("speed", speeds), ("diffusivity", diffusivities)):
        if values.shape[0] not in (0, n_axes):
            raise ValidationError(
                f"stable_step: '{name}' must be empty or have one value per axis, "
                f"got {values.shape[0]} for {n_axes} axes"
            )

    # A blown-up state's NaN speed must not come back as a NaN step.
    traced_checks = [
        _check(spacing, spacings, "spacing", positive=True),
        _check(cfl, cfl_value, "cfl", positive=True),
        _check(speed, speeds, "|speed|", positive=False),
        _check(diffusivity, diffusivities, "diffusivity", positive=False),
    ]

    zero = jnp.zeros((), dtype=dtype)
    # ∑ᵢ |uᵢ| / Δxᵢ
    advective_rate = zero
    if speeds.shape[0]:
        advective_rate = jnp.sum(speeds / spacings)
    # 2 ∑ᵢ Kᵢ / Δxᵢ²
    diffusive_rate = zero
    if diffusivities.shape[0]:
        diffusive_rate = 2.0 * jnp.sum(diffusivities / spacings**2)
    total_rate = advective_rate + diffusive_rate
    # h_max = CFL / rate, inf for a zero rate; the untaken branch divides by one
    moving = total_rate > 0.0
    safe_rate = jnp.where(moving, total_rate, 1.0)
    bound = jnp.where(moving, cfl_value[0] / safe_rate, jnp.inf).astype(dtype)
    # the traced checks ride on the result, so jit cannot drop them even when
    # the inputs they check do not reach the bound (nothing moves)
    for check in traced_checks:
        if check is not None:
            bad, message = check
            bound = eqx.error_if(bound, bad, message)
    return bound


def _working_dtype(*arguments: Any) -> Any:
    """The float dtype of the JAX-array arguments, or the default float.

    Python and NumPy values carry no precision the caller chose, so a float32
    spacing with a Python speed tuple stays float32.
    """
    dtypes = [
        jnp.finfo(leaf.dtype).dtype if jnp.issubdtype(leaf.dtype, jnp.inexact) else None
        for argument in arguments
        for leaf in jax.tree.leaves(argument)
        if isinstance(leaf, jax.Array) and leaf.size
    ]
    given = [dtype for dtype in dtypes if dtype is not None]
    return jnp.result_type(*given) if given else jnp.result_type(float)


def _as_vector(
    values: Any,
    name: str,
    dtype: Any,
    *,
    magnitude: bool = False,
    scalar: bool = False,
) -> Float[Array, " axis"]:
    """One value per axis (or, for ``scalar``, one value) as a real 1-D array.

    ``magnitude`` takes |values| first, so a complex or negative speed counts
    by its size, as sesmax-core's ``abs`` did.
    """
    array = jnp.abs(jnp.asarray(values)) if magnitude else jnp.asarray(values)
    if scalar and array.ndim != 0:
        raise ValidationError(
            f"stable_step: '{name}' must be a scalar, got shape {array.shape}"
        )
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise ValidationError(f"stable_step: '{name}' must be real, got complex")
    vector = array.astype(dtype)
    if vector.ndim == 0:
        vector = vector[None]
    if vector.ndim != 1:
        raise ValidationError(
            f"stable_step: '{name}' must hold one value per axis, got shape "
            f"{vector.shape}"
        )
    return vector


def _check(
    raw: Any,
    values: Float[Array, " axis"],
    label: str,
    *,
    positive: bool,
) -> tuple[Bool[Array, ""], str] | None:
    """Check one argument: eagerly on the host, or return a traced check.

    Python and NumPy inputs are concrete, so they are checked on the host and
    raise `ValidationError` with sesmax-core's message. A JAX array may be
    traced, so its check comes back as ``(bad, message)`` for the caller to
    attach to the result with `equinox.error_if`.

    Args:
        raw: The argument as the caller passed it.
        values: The same values as a real JAX vector (|speed| for a speed).
        label: The argument's name, for the message.
        positive: Require > 0 (else ≥ 0); finite either way.

    Returns:
        ``None`` for a host argument that passed, else the traced check.

    Raises:
        ValidationError: If a host argument fails the check.
    """
    expected = "positive" if positive else "non-negative"
    message = f"stable_step: '{label}' must be a finite {expected} number"
    if any(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(raw)):
        bad = ~jnp.isfinite(values) | (values <= 0 if positive else values < 0)
        return jnp.any(bad), message
    host = (
        np.abs(np.asarray(raw)).reshape(-1)
        if label == "|speed|"
        else (np.asarray(raw, dtype=np.float64).reshape(-1))
    )
    host = host.astype(np.float64)
    bad = ~np.isfinite(host) | (host <= 0 if positive else host < 0)
    if bad.any():
        index = int(np.argmax(bad))
        raise ValidationError(f"{message}, got {host[index]!r} at index {index}")
    return None
