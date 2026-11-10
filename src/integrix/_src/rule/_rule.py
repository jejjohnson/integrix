"""Integrate values or an integrand against quadrature weights, over any axes."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, Literal

import einx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Inexact, Int

from integrix._src._axes import (
    as_inexact,
    check_node_shape,
    flatten,
    nodes_first,
    normalise_axes,
    working_dtype,
)
from integrix._src.rule._pairwise import pairwise_sum
from integrix._typing import Axes, RuleLike
from integrix.exceptions import ValidationError


__all__ = [
    "Integral",
    "integrate",
    "weighted_mean",
]


type Rule = RuleLike | tuple[RuleLike, ...] | ArrayLike
type NanPolicy = Literal["propagate", "omit"]


class Integral(eqx.Module):
    """The result of an integral: its value, an error estimate and diagnostics.

    A plain rule has no error estimate, so ``error`` is ``None`` and
    ``converged`` is ``True``; adaptive procedures (`integrix.adapt`) fill both.

    Attributes:
        value: The integral I, one per batch entry.
        error: An estimate of |I − Î|, or ``None`` for a plain rule.
        n_evaluations: The number of integrand values summed.
        converged: Whether the error target was met.

    Examples:
        >>> import jax.numpy as jnp
        >>> result = integrate(jnp.array([0.5, 0.5]), jnp.array([1.0, 3.0]))
        >>> bool(result.value == 2.0), result.error, bool(result.converged)
        (True, None, True)
    """

    value: Inexact[Array, "*batch"]
    error: Inexact[Array, "*batch"] | None
    n_evaluations: Int[Array, ""]
    converged: Bool[Array, ""]


def integrate(
    rule: Rule,
    values_or_fn: ArrayLike | Callable[..., Any],
    *,
    axis: Axes | None = None,
    args: Any = None,
    where: ArrayLike | None = None,
    nan_policy: NanPolicy = "propagate",
) -> Integral:
    """Integrate values, or an integrand, against a rule's weights.

    I ≈ ∑ₙ wₙ f(xₙ), contracted over one axis or several and batched over the
    rest; with a mask or ``nan_policy="omit"``, I ≈ ∑ over kept n of wₙ fₙ.

    Algorithm:
        1. Read the weights w: a rule's ``weights``, the outer product of a
           tuple of 1-D rules (a tensor rule, one axis each), or a plain array
           (a spectral response, cell areas).
        2. Get the samples: the values as given, or ``f(nodes)`` (with
           ``args``, ``f(nodes, args)``), whose node axes come first.
        3. Keep the samples where ``where`` holds and, with
           ``nan_policy="omit"``, that are not NaN; put 0 in place of every
           other sample, so it adds nothing and gets a zero gradient.
        4. Move the node axes first and flatten them into one, (N, …).
        5. Cast w to the samples' dtype and sum wₙ fₙ pairwise over N.

    Args:
        rule: A rule with ``weights`` (functax's ``Rule``), a tuple of 1-D
            rules, or a weights array w.
        values_or_fn: The samples fₙ, or a callable f evaluated at the rule's
            ``nodes`` and returning its node axes first.
        axis: The axes of the samples to integrate over, one per weights
            axis. ``None`` takes the last ones (``-1`` for a 1-D rule); a
            callable's node axes are always first, so it takes no ``axis``.
        args: Extra arguments passed to the callable as ``f(nodes, args)``.
        where: A boolean mask that broadcasts against the samples (a plume
            over an image, the valid levels of each column); samples where it
            is false are left out. ``None`` keeps every sample.
        nan_policy: ``"propagate"`` (a NaN among the kept samples makes the
            integral NaN, as without the argument) or ``"omit"`` (NaN samples
            are left out, like ``np.nansum``). A callable's own derivative is
            not guarded: if f itself produces a NaN in a gradient, mask inside
            f.

    Returns:
        An `Integral` whose ``value`` has the shape of the samples without
        the integrated axes; ``error`` is ``None``.

    Raises:
        ValidationError: If the rule has no weights, a tensor rule holds a
            rule that is not 1-D, a callable meets a rule without ``nodes``
            or an ``axis``, an axis is out of range or repeated, the samples
            do not have one entry per node along the integrated axes,
            ``where`` is not boolean or does not broadcast to the samples, or
            ``nan_policy`` is neither ``"propagate"`` nor ``"omit"``.

    Examples:
        >>> import jax.numpy as jnp
        >>> weights = jnp.array([0.25, 0.75])
        >>> total = integrate(weights, jnp.array([[1.0, 2.0], [10.0, 20.0]]))
        >>> bool(jnp.allclose(total.value, jnp.array([1.75, 17.5])))
        True

        A callable returns its node axis first:

        >>> import equinox as eqx
        >>> class Trapezoid(eqx.Module):
        ...     nodes: jnp.ndarray
        ...     weights: jnp.ndarray
        >>> rule = Trapezoid(jnp.array([0.0, 1.0]), jnp.array([0.5, 0.5]))
        >>> total = integrate(rule, lambda x: jnp.outer(x, jnp.array([1.0, 10.0])))
        >>> bool(jnp.allclose(total.value, jnp.array([0.5, 5.0])))
        True

        A tensor rule integrates two axes, one rule each:

        >>> image = jnp.ones((2, 2, 4))
        >>> integrate((rule, rule), image, axis=(0, 1)).value.shape
        (4,)

        A mask and a NaN, both left out (``np.nansum`` over the mask):

        >>> enhancement = jnp.array([1.0, jnp.nan, 3.0, 4.0])
        >>> plume = jnp.array([True, True, True, False])
        >>> mass = integrate(
        ...     jnp.ones(4), enhancement, where=plume, nan_policy="omit"
        ... )
        >>> bool(mass.value == 4.0)
        True
    """
    weights, nodes = _read_rule(rule)
    if callable(values_or_fn):
        if nodes is None:
            raise ValidationError(
                "'rule' must have 'nodes' to evaluate a callable integrand"
            )
        if axis is not None:
            raise ValidationError(
                "'axis' must be None for a callable integrand, whose node axes "
                "come first"
            )
        evaluated = values_or_fn(nodes) if args is None else values_or_fn(nodes, args)
        values = jnp.asarray(evaluated)
        axes = tuple(range(weights.ndim))
    else:
        values = jnp.asarray(values_or_fn)
        axes = normalise_axes(axis, values.ndim, weights.ndim)
    check_node_shape(values, axes, weights.shape)
    keep = _kept_samples(values, where, nan_policy)

    dtype = working_dtype(values.dtype, weights.dtype)
    samples = nodes_first(values.astype(dtype), axes)
    if keep is not None:
        kept = nodes_first(keep, axes) > 0
        # 0 in place of every dropped sample, so wₙ · 0 carries no NaN forward
        # and the dropped sample's gradient is exactly 0
        samples = jnp.where(kept, samples, jnp.zeros((), dtype))
    flat_weights = flatten(weights).astype(dtype)
    value = _contract(flat_weights, samples)
    return Integral(
        value=value,
        error=None,
        n_evaluations=jnp.asarray(flat_weights.shape[0]),
        converged=jnp.asarray(True),
    )


def weighted_mean(
    rule: Rule,
    values: ArrayLike,
    *,
    weight: ArrayLike,
    axis: Axes | None = None,
    where: ArrayLike | None = None,
    nan_policy: NanPolicy = "propagate",
) -> Inexact[Array, "*batch"]:
    """The mean of the values under a weight function, with one rule for both sums.

    L̄ = ∫ L(x) R(x) dx / ∫ R(x) dx ≈ ∑ₙ wₙ Rₙ Lₙ / ∑ₙ wₙ Rₙ, a band radiance
    from a spectrum L and a spectral response R, say. With a mask or
    ``nan_policy="omit"``, both sums run over the kept samples K only, so the
    mean renormalises: ∑ over K of wₙ Rₙ Lₙ / ∑ over K of wₙ Rₙ.

    Algorithm:
        1. Place R along the integrated axes (or use it as given when it has
           the values' shape).
        2. Keep the samples as `integrate` does, with 0 in place of the
           others in both sums.
        3. Sum wₙ Rₙ Lₙ and wₙ Rₙ pairwise with the same weights.
        4. Divide.

    Args:
        rule: A rule with ``weights``, a tuple of 1-D rules, or a weights
            array w, as in `integrate`.
        values: The values Lₙ.
        weight: The weight function Rₙ, with the shape of ``values`` (one per
            sample, laid out like the values) or of the rule's weights (one
            per node, in the rule's order). A weight that has both shapes is
            read per sample.
        axis: The axes to average over, as in `integrate`.
        where: A boolean mask that broadcasts against the values, as in
            `integrate`.
        nan_policy: ``"propagate"`` or ``"omit"``, as in `integrate`; only NaN
            values are omitted, not NaN weights.

    Returns:
        The weighted mean, with the shape of ``values`` without the averaged
        axes. A weight that sums to zero, or a mean with no kept sample,
        gives ``nan``.

    Raises:
        ValidationError: As `integrate`, or if ``weight`` has neither the
            weights' shape nor the values' shape.

    Examples:
        >>> import jax.numpy as jnp
        >>> weights = jnp.array([0.5, 1.0, 0.5])
        >>> radiance = jnp.array([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]])
        >>> response = jnp.array([0.0, 1.0, 1.0])
        >>> band = weighted_mean(weights, radiance, weight=response)
        >>> bool(jnp.allclose(band, jnp.array([7.0 / 3.0, 2.0])))
        True

        A NaN sample is left out and the mean renormalises over the rest:

        >>> cloudy = jnp.array([[1.0, 2.0, jnp.nan]])
        >>> band = weighted_mean(
        ...     weights, cloudy, weight=response, nan_policy="omit"
        ... )
        >>> bool(jnp.allclose(band, jnp.array([2.0])))
        True
    """
    weights, _ = _read_rule(rule)
    values = jnp.asarray(values)
    weight = jnp.asarray(weight)
    axes = normalise_axes(axis, values.ndim, weights.ndim)
    check_node_shape(values, axes, weights.shape)
    keep = _kept_samples(values, where, nan_policy)

    dtype = working_dtype(values.dtype, weights.dtype, weight.dtype)
    samples = nodes_first(values.astype(dtype), axes)
    flat_weights = flatten(weights).astype(dtype)
    zero = jnp.zeros((), dtype)
    kept = None if keep is None else nodes_first(keep, axes) > 0
    if kept is not None:
        # 0 in place of every dropped sample, as in `integrate`
        samples = jnp.where(kept, samples, zero)
    # A weight shaped like the values is read per sample, in the values'
    # layout, even when it also has the weights' shape (a square grid).
    if weight.shape == values.shape:
        # one Rₙ per sample: the same contraction on R ⊙ L and on R
        response = nodes_first(weight.astype(dtype), axes)
        if kept is not None:
            # a dropped sample leaves the denominator too: ∑ over K of wₙ Rₙ
            response = jnp.where(kept, response, zero)
        numerator = _contract(flat_weights, response * samples)
        denominator = _contract(flat_weights, response)
    elif weight.shape == weights.shape:
        # one Rₙ per node, in the rule's order: wₙ Rₙ
        flat_response = flatten(weight).astype(dtype)
        combined = flat_weights * flat_response
        numerator = _contract(combined, samples)
        if kept is None:
            denominator = pairwise_sum(combined)
        else:
            # ∑ over K of wₙ Rₙ, one sum per batch entry
            denominator = _contract(combined, kept.astype(dtype))
    else:
        raise ValidationError(
            f"'weight' must have the rule's weights shape {weights.shape} or the "
            f"values' shape {values.shape}, got {weight.shape}"
        )
    # L̄ = ∑ wR L / ∑ wR
    return numerator / denominator


def _read_rule(rule: Rule) -> tuple[Inexact[Array, ...], Any]:
    """The weights of a rule, a tensor rule or a weights array, and its nodes."""
    if isinstance(rule, tuple):
        if not rule:
            raise ValidationError("'rule' must hold at least one rule, got ()")
        factors = []
        for index, factor in enumerate(rule):
            if not isinstance(factor, RuleLike):
                raise ValidationError(
                    f"'rule' must hold rules with 'weights', got "
                    f"{type(factor).__name__} at position {index}"
                )
            factor_weights = jnp.asarray(factor.weights)
            if factor_weights.ndim != 1:
                raise ValidationError(
                    f"'rule' must hold 1-D rules, got weights of shape "
                    f"{factor_weights.shape} at position {index}"
                )
            factors.append(factor_weights)
        # w_{n₁ n₂ …} = w⁽¹⁾_{n₁} w⁽²⁾_{n₂} …
        weights = functools.reduce(
            lambda outer, factor: einx.multiply("..., n -> ... n", outer, factor),
            factors,
        )
        nodes_list = [getattr(factor, "nodes", None) for factor in rule]
        nodes = None if any(n is None for n in nodes_list) else tuple(nodes_list)
    elif isinstance(rule, RuleLike):
        weights = jnp.asarray(rule.weights)
        nodes = getattr(rule, "nodes", None)
    else:
        try:
            weights = jnp.asarray(rule)
        except TypeError as err:
            raise ValidationError(
                f"'rule' must be a rule with 'weights', a tuple of rules or a "
                f"weights array, got {type(rule).__name__}"
            ) from err
        nodes = None
    if weights.ndim == 0 or weights.size == 0:
        raise ValidationError(
            f"'rule' must have at least one weight per axis, got weights of "
            f"shape {weights.shape}"
        )
    return as_inexact(weights), nodes


def _kept_samples(
    values: Array, where: ArrayLike | None, nan_policy: str
) -> Bool[Array, " ..."] | None:
    """The samples to keep, in the values' shape, or ``None`` to keep them all."""
    if nan_policy not in ("propagate", "omit"):
        raise ValidationError(
            f"'nan_policy' must be 'propagate' or 'omit', got {nan_policy!r}"
        )
    keep = None
    if where is not None:
        mask = jnp.asarray(where)
        if mask.dtype != jnp.bool_:
            raise ValidationError(f"'where' must be boolean, got dtype {mask.dtype}")
        try:
            shape = jnp.broadcast_shapes(mask.shape, values.shape)
        except ValueError as err:
            raise ValidationError(
                f"'where' must broadcast to the values' shape {values.shape}, "
                f"got {mask.shape}"
            ) from err
        if shape != values.shape:
            raise ValidationError(
                f"'where' must broadcast to the values' shape {values.shape}, "
                f"got {mask.shape}"
            )
        keep = jnp.broadcast_to(mask, values.shape)
    if nan_policy == "omit":
        present = ~jnp.isnan(values)
        keep = present if keep is None else keep & present
    return keep


def _contract(
    weights: Inexact[Array, " node"], samples: Inexact[Array, " node *rest"]
) -> Inexact[Array, "*rest"]:
    """∑ₙ wₙ fₙ over the leading axis, summed pairwise."""
    # tₙ = wₙ fₙ
    terms = einx.multiply("n ..., n -> n ...", samples, weights)
    return pairwise_sum(terms)
