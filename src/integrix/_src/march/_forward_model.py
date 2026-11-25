"""A forward model that one diffrax solve computes.

Moved from pipekit-jax (``diffrax_model.py`` at pipekit ``5fb9a12``) with its
interface unchanged, so that pipekit depends on no library above it
(pipekit#283). The default adjoint is diffrax's own class in place of
pipekit-cycle's spec of the same name.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp

from integrix._src.adjoint._diffrax import to_diffrax_adjoint
from integrix.exceptions import ValidationError


__all__ = [
    "DiffraxForwardModel",
]


class DiffraxForwardModel(eqx.Module):
    """A forward model M: x(t₀) ↦ x(t₁) that one `diffrax.diffeqsolve` computes.

    The state follows ẋ = f(t, x), or dx = f(t, x) dt + g(t, x) dW if a
    ``diffusion`` is given, whose Brownian path is a
    `diffrax.VirtualBrownianTree` over each span. `step` is autonomous: it
    integrates over [0, dt], because an assimilation cycle calls it with no
    absolute time. `as_dynamics` gives the form with an explicit span.

    It satisfies pipekit-cycle's ``ForwardModel`` and ``SupportsAdjoint``
    structurally, and filterax's dynamics slots take `as_dynamics`.

    Args:
        vector_field: The drift f(t, x, args), in diffrax's convention.
        solver: Any diffrax solver, the schemes of `integrix.march` included.
        dt0: The first step of the solver, and every step under a constant one.
        dt: The window length that the model advertises to a cycle.
        adjoint: How a reverse-mode gradient goes through the solve: a diffrax
            adjoint, or a spec of pipekit-cycle that
            `integrix.adjoint.to_diffrax_adjoint` resolves.
        stepsize_controller: The step-size controller; constant steps by default.
        diffusion: g(t, x, args), which makes the solve an SDE: a linear map from
            the Brownian increment to the state, as
            ``lineax.DiagonalLinearOperator(sigma)`` for diagonal noise or an
            (N_x, N_w) matrix.
        sde_key: The PRNG key of the Brownian path. It is fixed, so rebuild the
            model with another one (``dataclasses.replace``) for an independent
            path per window.
        brownian_shape: The shape of the Brownian increment. ``None`` takes the
            state's, which is right for diagonal noise; pass ``(N_w,)`` for a
            matrix g.
        max_steps: The largest number of steps of one solve.

    Examples:
        >>> import diffrax as dfx
        >>> import jax.numpy as jnp
        >>> model = DiffraxForwardModel(lambda t, x, args: -x, dt0=0.01)
        >>> x0 = jnp.array([1.0, 2.0])
        >>> bool(jnp.allclose(model.step(x0, 1.0), x0 * jnp.exp(-1.0), atol=1e-6))
        True

        An explicit span, and another adjoint:

        >>> model = model.with_adjoint(dfx.BacksolveAdjoint())
        >>> dynamics = model.as_dynamics()
        >>> bool(jnp.allclose(dynamics(x0, 1.0, 3.0), x0 * jnp.exp(-2.0)))
        True
    """

    vector_field: Callable[[Any, Any, Any], Any]
    solver: dfx.AbstractSolver = dfx.Tsit5()
    dt0: float = 0.01
    dt: float = 1.0
    # Any: a diffrax adjoint, or a spec that to_diffrax_adjoint resolves
    adjoint: Any = eqx.field(
        static=True, default_factory=dfx.RecursiveCheckpointAdjoint
    )
    stepsize_controller: dfx.AbstractStepSizeController = dfx.ConstantStepSize()
    diffusion: Callable[[Any, Any, Any], Any] | None = None
    sde_key: jax.Array | None = None
    brownian_shape: tuple[int, ...] | None = eqx.field(static=True, default=None)
    max_steps: int = eqx.field(static=True, default=4096)

    def __check_init__(self) -> None:
        if self.diffusion is not None and self.sde_key is None:
            raise ValidationError(
                "'sde_key' is needed for the Brownian path when 'diffusion' is given"
            )

    def _terms(self, t0: Any, t1: Any, state: Any) -> Any:
        drift = dfx.ODETerm(self.vector_field)
        if self.diffusion is None:
            return drift
        bm_shape = (
            self.brownian_shape if self.brownian_shape is not None else jnp.shape(state)
        )
        brownian = dfx.VirtualBrownianTree(
            t0,
            t1,
            tol=self.dt0 / 2,
            shape=jax.ShapeDtypeStruct(bm_shape, jnp.result_type(state)),
            key=self.sde_key,
        )
        return dfx.MultiTerm(drift, dfx.ControlTerm(self.diffusion, brownian))

    def _solve(self, state: Any, t0: Any, t1: Any) -> Any:
        sol = dfx.diffeqsolve(
            self._terms(t0, t1, state),
            self.solver,
            t0=t0,
            t1=t1,
            dt0=self.dt0,
            y0=state,
            stepsize_controller=self.stepsize_controller,
            adjoint=to_diffrax_adjoint(self.adjoint),
            max_steps=self.max_steps,
            saveat=dfx.SaveAt(t1=True),
        )
        return jax.tree.map(lambda y: y[-1], sol.ys)

    def step(self, state: Any, dt: Any) -> Any:
        """The state after dt, integrated over [0, dt] (autonomous dynamics).

        Args:
            state: x(0), an array or, for an ODE, a pytree of them.
            dt: The length of the span.

        Returns:
            x(dt), in the structure of ``state``.
        """
        return self._solve(state, jnp.zeros_like(jnp.asarray(dt)), dt)

    @property
    def state_signature(self) -> None:
        """``None``: the model advertises no shape contract to a pipekit cycle."""
        return None

    def with_adjoint(self, spec: Any) -> DiffraxForwardModel:
        """A copy of the model with another adjoint.

        Args:
            spec: A diffrax adjoint, or a spec of pipekit-cycle.

        Returns:
            The copy.
        """
        return dataclasses.replace(self, adjoint=spec)

    def as_dynamics(self) -> Callable[[Any, Any, Any], Any]:
        """The model as (state, t₀, t₁) ↦ state, with an explicit span.

        Returns:
            The callable, for an API such as filterax's dynamics slots or a
            4D-Var ``forward`` argument.
        """
        return self._solve
