"""Samples of a forcing or a control, read as a diffrax path.

A solver reads a forcing at its stage times, between the sample times, so the
samples (t_k, u_k) must first become a function û(t). functax owns that
reconstruction (its 1-D schemes); this module gives a functax fit diffrax's
path protocol, so it can force an ODE or drive a CDE.

References:
    Morrill, J., Kidger, P., Yang, L. and Lyons, T. (2021). Neural controlled
    differential equations for online prediction tasks. arXiv:2106.11028.
"""

from __future__ import annotations

from typing import Any

import diffrax as dfx
import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from functax.geometry import Axis
from functax.interp import (
    Interpolant,
    LocalScheme,
    Policies,
    SeriesInterpolant,
    SeriesScheme,
)
from jax.core import Tracer
from jaxtyping import Array, Float, ScalarLike

from integrix.exceptions import ValidationError


__all__ = [
    "SampledPath",
]


class SampledPath(dfx.AbstractPath):
    """û(t), a functax fit of samples (t_k, u_k), read as a diffrax path.

    û interpolates the samples in time with the scheme that fitted it. Force an
    ODE with it, ẋ = f(t, x) + û(t), by calling `evaluate` in the vector field,
    or drive a controlled differential equation dx = f(x) dû(t) with
    ``diffrax.ControlTerm``, which reads the increments û(t₁) − û(t₀).

    û has a kink at every t_k (a jump in û′ for `functax.interp.Linear`, in û″
    for `functax.interp.Pchip`, in û‴ for `functax.interp.CubicSpline`), and a
    step that straddles a kink loses order. Pass ``ts`` to the step control
    (`imex_stepsize_controller(step_ts=path.ts)`, or
    ``diffrax.ClipStepSizeController``), or take fixed steps that divide the
    sample spacing.

    Args:
        field: The functax field over the samples, with the samples on its last
            axis (`fit` builds it).

    Examples:
        Two channels sampled at uneven times:

        >>> import jax.numpy as jnp
        >>> from functax.interp import Linear, Policies
        >>> ts = jnp.array([0.0, 1.0, 3.0])
        >>> us = jnp.array([[0.0, 0.0], [1.0, 10.0], [5.0, 30.0]])
        >>> path = SampledPath.fit(Linear(), ts, us)
        >>> path.evaluate(2.0).tolist()
        [3.0, 20.0]

        An increment, and the derivative on either side of a sample:

        >>> path.evaluate(1.0, 3.0).tolist()
        [4.0, 20.0]
        >>> left, right = path.derivative(1.0), path.derivative(1.0, left=False)
        >>> bool(jnp.allclose(left, jnp.array([1.0, 10.0])))
        True
        >>> bool(jnp.allclose(right, jnp.array([2.0, 10.0])))
        True

        Past the samples the default holds the end values; a policy changes it:

        >>> path.evaluate(4.0).tolist()
        [5.0, 30.0]
        >>> policies = Policies(outside="extrapolate")
        >>> SampledPath.fit(Linear(), ts, us, policies=policies).evaluate(
        ...     4.0
        ... ).tolist()
        [7.0, 40.0]

        A forcing in an ODE, ẋ = û(t), with the steps on the samples:

        >>> import diffrax as dfx
        >>> from integrix.march import RK2Heun, imex_stepsize_controller
        >>> solution = dfx.diffeqsolve(
        ...     dfx.ODETerm(lambda t, x, args: path.evaluate(t)),
        ...     RK2Heun(),
        ...     path.t0,
        ...     path.t1,
        ...     0.5,
        ...     jnp.zeros(2),
        ...     stepsize_controller=imex_stepsize_controller(step_ts=path.ts),
        ... )
        >>> bool(jnp.allclose(solution.ys[0], jnp.array([6.5, 45.0])))
        True
    """

    field: Interpolant | SeriesInterpolant

    @classmethod
    def fit(
        cls,
        # Any: a LocalScheme | SeriesScheme, checked below, so that a bad one
        # raises ValidationError before the type hook sees it
        scheme: Any,
        ts: Float[Array, " sample"],
        us: Float[Array, "sample ..."],
        *,
        policies: Policies | None = None,
    ) -> SampledPath:
        """The path that interpolates the samples with ``scheme``.

        The fit runs in JAX, so gradients reach both the samples and their
        times.

        Args:
            scheme: A functax 1-D scheme: a `functax.interp.LocalScheme`
                (`Linear`, `Cubic`, `BSpline`, …) or a
                `functax.interp.SeriesScheme` (`CubicSpline`, `Pchip`, `Akima`,
                `Steffen`).
            ts: The sample times t_k, increasing, at least two.
            us: The samples u_k, time first.
            policies: functax's rules off the samples and at missing ones;
                ``Policies(outside="clamp")`` when ``None``, since functax's
                own default reads NaN a rounding error past the last sample,
                where a solver's final stage can land.

        Returns:
            The `SampledPath`.

        Raises:
            ValidationError: If ``scheme`` is not a functax local or series
                scheme, there are fewer than two times, ``us`` does not have
                one sample per time, or concrete times decrease (traced ones
                are checked with `equinox.error_if`).
            functax.exceptions.GeometryError: If the times are not strictly
                monotone, or uneven for a scheme that reads in index space
                (`Cubic`, and `BSpline` from order 2).
        """
        if not isinstance(scheme, LocalScheme | SeriesScheme):
            raise ValidationError(
                "'scheme' must be a functax LocalScheme or SeriesScheme, got "
                f"{type(scheme).__name__}"
            )
        ts = jnp.asarray(ts)
        us = jnp.asarray(us)
        n_samples = ts.shape[0]
        if n_samples < 2:
            raise ValidationError(f"'ts' must have at least two times, got {n_samples}")
        if us.shape[0] != n_samples:
            raise ValidationError(
                f"'us' must have one sample per time, got shape {us.shape} for "
                f"{n_samples} times"
            )
        message = "'ts' must increase"
        if isinstance(ts, Tracer):
            ts = eqx.error_if(ts, ts[0] > ts[-1], message)
            axis = Axis(ts)
        else:
            # on the host: indexing a concrete array under jit stages a tracer
            first, last = np.asarray(ts)[[0, -1]]
            if first > last:
                raise ValidationError(f"{message}, got {first} to {last}")
            # concrete times stay concrete under jit, so functax can check them
            with jax.ensure_compile_time_eval():
                axis = Axis(ts)
        policies = Policies(outside="clamp") if policies is None else policies
        # functax keeps the samples on the last axis
        data = einx.id("sample ... -> ... sample", us)
        return cls(scheme.fit(axis, data, policies=policies))

    @property
    def ts(self) -> Float[Array, " sample"]:
        """The sample times t_k, where û has its kinks."""
        return self.field.source.values

    # properties over diffrax's abstract t0 and t1, as its own paths define them;
    # ty reads the AbstractVar annotation as a class variable
    @property
    def t0(  # ty: ignore[invalid-attribute-override]
        self,
    ) -> Float[Array, ""]:
        """The first sample time."""
        return self.ts[0]

    @property
    def t1(  # ty: ignore[invalid-attribute-override]
        self,
    ) -> Float[Array, ""]:
        """The last sample time."""
        return self.ts[-1]

    def evaluate(
        self, t0: ScalarLike, t1: ScalarLike | None = None, left: bool = True
    ) -> Float[Array, "*channel"]:
        """û(t0), or the increment û(t1) − û(t0) when ``t1`` is given.

        Args:
            t0: The time.
            t1: The end of the increment, or ``None`` for a value.
            left: Unused: û is continuous (`functax.interp.Nearest`, which
                jumps halfway between samples, is left-continuous there).

        Returns:
            The value or the increment, in the dtype of the samples.
        """
        del left
        if t1 is not None:
            # û(t₁) − û(t₀)
            return self.evaluate(t1) - self.evaluate(t0)
        return self._read(self.field, self._time(t0))

    def derivative(self, t: ScalarLike, left: bool = True) -> Float[Array, "*channel"]:
        """û′(t), from the closed-form derivative of the functax field.

        At a sample time inside (t0, t1), where û′ may jump, the one-sided
        derivative is read 4 ε max(|t0|, |t1|) to that side, since functax
        reads whichever cell rounding puts the target in.

        Args:
            t: The time.
            left: Whether to read the left or the right derivative.

        Returns:
            The derivative, in the dtype of the samples.
        """
        time = self._time(t)
        # δ = 4 ε max(|t₀|, |t₁|), a few roundings of a time
        largest = jnp.maximum(jnp.abs(self.t0), jnp.abs(self.t1))
        rounding = 4.0 * jnp.finfo(self.ts.dtype).eps * largest
        signed_shift = -rounding if left else rounding
        # only between the ends, so the shift never crosses one
        inside = (time > self.t0 + rounding) & (time < self.t1 - rounding)
        shift = jax.lax.stop_gradient(jnp.where(inside, signed_shift, 0.0))
        side = time + shift.astype(time.dtype)
        return self._read(self.field.derivative(0), side)

    def _time(self, t: ScalarLike) -> Float[Array, ""]:
        """t as a floating scalar, in its own dtype or that of the sample times."""
        time = jnp.asarray(t)
        if not jnp.issubdtype(time.dtype, jnp.floating):
            time = time.astype(self.ts.dtype)
        return time

    def _read(
        self, field: Interpolant | SeriesInterpolant, time: Float[Array, ""]
    ) -> Float[Array, "*channel"]:
        """The field at one time, in the dtype of the samples."""
        # the (M, 1) targets functax reads, with M = 1
        values = field(time[None, None])
        # a float64 solver clock must not promote a float32 state
        return values[..., 0].astype(field.coefficients.dtype)
