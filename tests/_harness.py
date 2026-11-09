"""Shared checks for rules, scans, steppers and estimators.

One call per property: exactness on polynomials, a SciPy reference, an
order-of-accuracy slope, a finite-difference gradient, a standard-error bound,
transform invariance and float32 drift. Each helper raises `AssertionError`
with the numbers behind the failure (the worst index, the fitted slope, the
number of standard errors), so a failing test says where and by how much.

The helpers need JAX's 64-bit mode, which `tests/conftest.py` turns on for the
whole suite, and they refuse to run without it rather than test in float32.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree


__all__ = [
    "assert_exact_on_polynomials",
    "assert_float32_drift",
    "assert_grad_matches_fd",
    "assert_matches_scipy",
    "assert_order",
    "assert_traces",
    "assert_within_stderr",
]


def _require_x64() -> None:
    """Fail when JAX's 64-bit mode is off, since every tolerance assumes float64."""
    if not jax.config.read("jax_enable_x64"):
        raise AssertionError(
            "the reference harness needs float64: enable jax_enable_x64 "
            "(tests/conftest.py does this for the suite)"
        )


def assert_matches_scipy(
    actual: Any,
    reference: Any,
    *,
    rtol: float,
    atol: float = 0.0,
) -> None:
    """Assert ``|actual − reference| ≤ atol + rtol·|reference|`` entry by entry.

    NaNs match only NaNs at the same index.

    Args:
        actual: The values under test.
        reference: The reference values (``scipy.integrate``, quadax, a closed
            form), the same shape.
        rtol: The relative tolerance.
        atol: The absolute tolerance, for references that cross zero.

    Raises:
        AssertionError: If the shapes differ or any entry is out of tolerance.
            The message gives the number of failures, the worst index, both
            values there, the error and the tolerance.

    Examples:
        >>> import numpy as np
        >>> from scipy.integrate import trapezoid
        >>> x = np.array([0.0, 0.5, 2.0])
        >>> assert_matches_scipy(2.0, trapezoid(x, x), rtol=1e-14)
    """
    actual_values = np.asarray(actual)
    reference_values = np.asarray(reference)
    if actual_values.shape != reference_values.shape:
        raise AssertionError(
            f"shape mismatch: actual {actual_values.shape}, "
            f"reference {reference_values.shape}"
        )

    # e = |a − b|; a NaN on one side only counts as an infinite error
    error = np.abs(actual_values - reference_values)
    both_nan = np.isnan(actual_values) & np.isnan(reference_values)
    error = np.where(both_nan, 0.0, error)
    error = np.where(np.isnan(error), np.inf, error)
    # τ = atol + rtol·|b|; a NaN reference contributes no tolerance
    reference_scale = np.where(np.isnan(reference_values), 0.0, reference_values)
    tolerance = atol + rtol * np.abs(reference_scale)
    failing = error > tolerance
    if not failing.any():
        return

    # The worst entry is the one furthest past its own tolerance.
    excess = np.where(failing, error - tolerance, -np.inf)
    worst = np.unravel_index(np.argmax(excess), excess.shape)
    raise AssertionError(
        f"{int(failing.sum())} of {failing.size} entries out of tolerance; "
        f"worst at index {tuple(int(i) for i in worst)}: "
        f"actual {actual_values[worst]!r}, reference {reference_values[worst]!r}, "
        f"error {error[worst]:.3e} > tolerance {tolerance[worst]:.3e} "
        f"(rtol={rtol}, atol={atol})"
    )


def assert_exact_on_polynomials(
    integrate_fn: Callable[[Callable[[jax.Array], jax.Array]], Any],
    max_degree: int,
    *,
    interval: tuple[float, float] = (0.0, 1.0),
    rtol: float = 1e-13,
) -> None:
    """Assert ``integrate_fn`` integrates xᵏ on [a, b] exactly for k = 0, …, d.

    The exact value is ∫ₐᵇ xᵏ dx = (bᵏ⁺¹ − aᵏ⁺¹) / (k + 1). The error is
    measured against the scale (b − a)·max(|a|, |b|)ᵏ, which bounds ∫ₐᵇ |x|ᵏ dx,
    so a monomial whose integral is zero (an odd power on [−1, 1]) is held to
    the same tolerance as the rest.

    Args:
        integrate_fn: Maps a vectorised integrand f to its integral over
            [a, b] (a rule's ``integrate``, closed over its rule).
        max_degree: The highest degree d the rule must integrate exactly
            (2N − 1 for an N-point Gauss–Legendre rule, 1 for the trapezoid).
        interval: The interval [a, b] the rule covers.
        rtol: The tolerance, relative to the scale above.

    Raises:
        AssertionError: If x64 is off or any degree up to d is not exact; the
            message gives the first failing degree and both values.

    Examples:
        >>> import jax.numpy as jnp
        >>> nodes = jnp.array([0.0, 0.5, 1.0])
        >>> simpson = jnp.array([1.0, 4.0, 1.0]) / 6.0
        >>> assert_exact_on_polynomials(lambda f: simpson @ f(nodes), 3)
    """
    _require_x64()
    lower, upper = interval
    width = upper - lower
    largest = max(abs(lower), abs(upper))
    for degree in range(max_degree + 1):
        # ∫ₐᵇ xᵏ dx = (bᵏ⁺¹ − aᵏ⁺¹) / (k + 1)
        exact = (upper ** (degree + 1) - lower ** (degree + 1)) / (degree + 1)
        actual = float(integrate_fn(lambda x, k=degree: x**k))
        scale = width * largest**degree
        if abs(actual - exact) > rtol * scale:
            raise AssertionError(
                f"not exact for degree {degree} on [{lower}, {upper}]: "
                f"actual {actual!r}, exact {exact!r}, error "
                f"{abs(actual - exact):.3e} > {rtol * scale:.3e} (rtol={rtol})"
            )


def assert_order(
    step_errors: Mapping[float, float],
    expected_order: float,
    *,
    slack: float = 0.1,
) -> None:
    """Assert the least-squares slope of log e against log h is p ± slack.

    A method of order p has global error e(h) ≈ C hᵖ at a fixed end time, so
    log e = log C + p log h, and the slope fitted over a sequence of step
    sizes estimates p.

    Args:
        step_errors: The global error e(h) at each step size h, at least two
            of them, every error positive.
        expected_order: The order p the method should show.
        slack: The largest accepted |slope − p|.

    Raises:
        AssertionError: If there are fewer than two step sizes, an error is
            not positive (the sequence reached round-off or the reference is
            the method itself), or the slope is off by more than ``slack``.

    Examples:
        >>> assert_order({0.1: 1e-2, 0.05: 2.5e-3, 0.025: 6.25e-4}, 2)
    """
    if len(step_errors) < 2:
        raise AssertionError(
            f"an order needs at least two step sizes, got {len(step_errors)}"
        )
    if any(error <= 0.0 for error in step_errors.values()):
        raise AssertionError(
            f"every error must be positive to take its log, got {dict(step_errors)}"
        )
    log_steps = np.log(np.array(list(step_errors.keys()), dtype=np.float64))
    log_errors = np.log(np.array(list(step_errors.values()), dtype=np.float64))
    # p̂ = argmin ∑ (log eᵢ − log C − p log hᵢ)²
    slope, _ = np.polyfit(log_steps, log_errors, deg=1)
    if abs(slope - expected_order) > slack:
        raise AssertionError(
            f"observed order {slope:.3f}, expected {expected_order} ± {slack} "
            f"(step sizes {sorted(step_errors)})"
        )


def assert_grad_matches_fd(
    fn: Callable[[Any], Any],
    x: Any,
    *,
    eps: float = 1e-6,
    rtol: float = 1e-6,
) -> None:
    """Assert ``jax.jacrev(fn)`` agrees with central finite differences at x.

    For each input entry k, (f(x + ε eₖ) − f(x − ε eₖ)) / 2ε approximates
    column k of the Jacobian. Inputs and outputs may be any pytree of arrays;
    both are flattened. The tolerance is relative to the largest entry of the
    finite-difference Jacobian, since single entries near zero carry the
    O(ε²) truncation error at full size.

    Args:
        fn: The differentiable function under test.
        x: The point, a float64 array or pytree of them.
        eps: The step ε.
        rtol: The relative tolerance.

    Raises:
        AssertionError: If x64 is off or the Jacobians differ; the message
            gives the worst (output, input) index.

    Examples:
        >>> import jax.numpy as jnp
        >>> assert_grad_matches_fd(
        ...     lambda x: jnp.sum(jnp.sin(x)), jnp.array([0.1, 0.7])
        ... )
    """
    _require_x64()
    flat_x, unravel = ravel_pytree(x)

    def flat_fn(flat_input: jax.Array) -> jax.Array:
        output, _ = ravel_pytree(fn(unravel(flat_input)))
        return output

    jacobian = jax.jacrev(flat_fn)(flat_x)

    def central_difference(step: jax.Array) -> jax.Array:
        # (f(x + ε eₖ) − f(x − ε eₖ)) / 2ε
        return (flat_fn(flat_x + step) - flat_fn(flat_x - step)) / (2.0 * eps)

    steps = eps * jnp.eye(flat_x.shape[0], dtype=flat_x.dtype)
    # one column per input entry: (input, output) → (output, input)
    finite_difference = jax.vmap(central_difference, out_axes=1)(steps)
    scale = float(jnp.max(jnp.abs(finite_difference))) if jacobian.size else 0.0
    assert_matches_scipy(jacobian, finite_difference, rtol=0.0, atol=rtol * scale)


def assert_within_stderr(
    estimate: Any,
    standard_error: Any,
    exact: Any,
    *,
    n_sigma: float = 4.0,
) -> None:
    """Assert ``|Î − I| ≤ n_σ σ̂``, entry by entry.

    The bound comes from the estimator's own sampling distribution: for an
    unbiased estimator with a near-normal sampling distribution, a 4σ̂
    excursion happens about once in 16,000 draws. Never replace it with a
    tolerance tuned on one draw.

    Args:
        estimate: The estimate Î.
        standard_error: Its standard error σ̂, the same shape, positive.
        exact: The exact value I.
        n_sigma: The number of standard errors n_σ allowed.

    Raises:
        AssertionError: If a standard error is not positive or an estimate is
            more than n_σ standard errors from the exact value; the message
            gives the worst index and how many standard errors it is off.

    Examples:
        >>> assert_within_stderr(2.01, 0.01, 2.0)
    """
    estimates = np.asarray(estimate, dtype=np.float64)
    errors = np.broadcast_to(np.asarray(standard_error, np.float64), estimates.shape)
    exacts = np.broadcast_to(np.asarray(exact, dtype=np.float64), estimates.shape)
    if not (errors > 0.0).all():
        raise AssertionError(f"every standard error must be positive, got {errors}")
    # z = |Î − I| / σ̂
    z_scores = np.abs(estimates - exacts) / errors
    if (z_scores <= n_sigma).all():
        return
    worst = np.unravel_index(np.argmax(z_scores), z_scores.shape)
    raise AssertionError(
        f"estimate {estimates[worst]!r} is {z_scores[worst]:.2f} standard errors "
        f"from {exacts[worst]!r} at index {tuple(int(i) for i in worst)} "
        f"(σ̂ = {errors[worst]:.3e}, allowed {n_sigma})"
    )


def _assert_trees_match(actual: Any, expected: Any, *, rtol: float, what: str) -> None:
    """Compare two pytrees leaf by leaf, naming the leaf and ``what`` on failure."""
    actual_leaves, actual_tree = jax.tree.flatten(actual)
    expected_leaves, expected_tree = jax.tree.flatten(expected)
    if actual_tree != expected_tree:
        raise AssertionError(f"{what}: tree structure {actual_tree} != {expected_tree}")
    for index, (got, want) in enumerate(
        zip(actual_leaves, expected_leaves, strict=True)
    ):
        try:
            assert_matches_scipy(got, want, rtol=rtol)
        except AssertionError as err:
            raise AssertionError(f"{what}, leaf {index}: {err}") from err


def _batch_size(args: tuple[Any, ...]) -> int:
    """The common leading length of every array leaf in ``args``."""
    sizes = {leaf.shape[0] for leaf in jax.tree.leaves(args) if np.ndim(leaf) > 0}
    if len(sizes) != 1:
        raise AssertionError(
            f"every array argument needs the same leading batch axis, got {sizes}"
        )
    return sizes.pop()


def assert_traces(fn: Callable[..., Any], *args: Any, rtol: float = 1e-12) -> None:
    """Assert ``jit(fn)``, ``vmap(fn)`` and an eager loop agree over a batch.

    Every array leaf of ``args`` carries a leading batch axis of the same
    length B. The reference is the eager loop ``[fn(*args[b]) for b in B]``,
    stacked; ``jit(fn)`` on each slice, ``vmap(fn)`` and ``jit(vmap(fn))`` on
    the batch must match it.

    Args:
        fn: The function under test; it may take and return pytrees.
        *args: The batched arguments.
        rtol: The relative tolerance.

    Raises:
        AssertionError: If x64 is off, the batch axes differ, or any variant
            disagrees with the eager loop; the message names the variant.

    Examples:
        >>> import jax.numpy as jnp
        >>> assert_traces(lambda x: jnp.cumsum(x), jnp.ones((3, 4)))
    """
    _require_x64()
    batch = _batch_size(args)

    def slice_at(index: int) -> tuple[Any, ...]:
        return jax.tree.map(lambda leaf: leaf[index], args)

    eager = [fn(*slice_at(index)) for index in range(batch)]
    reference = jax.tree.map(lambda *leaves: jnp.stack(leaves), *eager)

    jitted_fn = jax.jit(fn)
    jitted = [jitted_fn(*slice_at(index)) for index in range(batch)]
    jitted_stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), *jitted)
    _assert_trees_match(jitted_stacked, reference, rtol=rtol, what="jit(fn)")

    vmapped = jax.vmap(fn)(*args)
    _assert_trees_match(vmapped, reference, rtol=rtol, what="vmap(fn)")

    jit_vmapped = jax.jit(jax.vmap(fn))(*args)
    _assert_trees_match(jit_vmapped, reference, rtol=rtol, what="jit(vmap(fn))")


def assert_float32_drift(
    fn: Callable[[jax.Array], jax.Array],
    values64: Any,
    *,
    rtol: float,
) -> None:
    """Assert ``fn`` in float32 keeps float32 and stays within rtol of float64.

    The reference is ``fn(values64)`` in float64; the run under test is
    ``fn(values64.astype(float32))``. A long left-to-right sum loses the small
    late increments in float32, and this is the check that sees it.

    Args:
        fn: The function under test, taking one array.
        values64: The float64 input (a long spectrum, say).
        rtol: The relative tolerance, against the largest |float64 output|.

    Raises:
        AssertionError: If x64 is off, the input is not float64, the float32
            run does not return float32, or the outputs drift apart; the
            message gives the worst index and the relative drift.

    Examples:
        >>> import jax.numpy as jnp
        >>> assert_float32_drift(jnp.sum, jnp.full(1000, 0.1), rtol=1e-5)
    """
    _require_x64()
    reference_input = jnp.asarray(values64)
    if reference_input.dtype != jnp.float64:
        raise AssertionError(f"values64 must be float64, got {reference_input.dtype}")
    reference = np.asarray(fn(reference_input))
    single = fn(reference_input.astype(jnp.float32))
    if single.dtype != jnp.float32:
        raise AssertionError(f"float32 input came back as {single.dtype}")
    scale = float(np.max(np.abs(reference))) if reference.size else 0.0
    drift = np.abs(np.asarray(single, dtype=np.float64) - reference)
    if (drift <= rtol * scale).all():
        return
    worst = np.unravel_index(np.argmax(drift), drift.shape)
    relative = drift[worst] / scale if scale else math.inf
    raise AssertionError(
        f"float32 drifts {relative:.3e} relative at index "
        f"{tuple(int(i) for i in worst)} (allowed {rtol}): float32 "
        f"{np.asarray(single)[worst]!r}, float64 {reference[worst]!r}"
    )
