"""Semi-Lagrangian advection (#92).

Parity with gridax-jax's first-order step, the translation gate with a cubic
B-spline read-out, the order in Δt of the midpoint trajectory, boundedness
above the CFL limit with a linear read-out, gradients, transforms, the
diffrax solver form and input validation.
"""

from __future__ import annotations

import diffrax as dfx
import einx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _harness import assert_grad_matches_fd, assert_order, assert_traces
from functax.geometry import Axis, Points, RectilinearGrid
from functax.interp import BSpline, Linear, Policies

import integrix.march as im
from integrix.exceptions import ValidationError


def _periodic_line(n: int, dtype=jnp.float64):
    x = jnp.arange(n, dtype=dtype) / n
    return x, Axis(x, period=1.0)


def _gaussian(s):
    # exp(−(s/w)²) with w = 0.05, centred at ½ on the unit circle
    offset = (s - 0.5 + 0.5) % 1.0 - 0.5
    return jnp.exp(-((offset / 0.05) ** 2))


def _translate(scheme, n=128, cfl=0.37, steps=200, dtype=jnp.float64):
    x, axis = _periodic_line(n, dtype)
    nodes = einx.id("node -> node 1", x)
    dt = cfl / n

    def velocity(t, p):
        return jnp.ones_like(p)

    def readout(departure):
        return scheme.plan(axis, Points(departure))

    def body(q, step):
        t = step * dt
        return im.semi_lagrangian_step(q, nodes, velocity, readout, t, dt), None

    q, _ = jax.lax.scan(body, _gaussian(x), jnp.arange(steps, dtype=dtype))
    return q, _gaussian(x - steps * dt)


def test_euler_linear_step_matches_gridax_jax():
    # gridax-jax semi_lagrangian_step at gridax c69768c: x_d = x − u Δt / Δx in
    # index space, map_coordinates(order=1, mode="wrap")
    ny, nx = 12, 16
    rows, cols = jnp.meshgrid(
        jnp.arange(ny, dtype=float), jnp.arange(nx, dtype=float), indexing="ij"
    )
    rng = np.random.default_rng(0)
    field = jnp.asarray(rng.standard_normal((ny, nx)))
    u = 1.3 + 0.4 * jnp.sin(2 * jnp.pi * cols / nx)
    v = -0.7 + 0.2 * jnp.cos(2 * jnp.pi * rows / ny)
    dt = 0.6
    reference = jax.scipy.ndimage.map_coordinates(
        field, [rows - v * dt, cols - u * dt], order=1, mode="wrap"
    )

    grid = RectilinearGrid(
        (
            Axis(jnp.arange(ny, dtype=float), period=float(ny)),
            Axis(jnp.arange(nx, dtype=float), period=float(nx)),
        )
    )
    nodes = jnp.column_stack([rows.ravel(), cols.ravel()])
    speed = jnp.column_stack([v.ravel(), u.ravel()])

    def readout(departure):
        return Linear().plan(grid, Points(departure))

    step = im.semi_lagrangian_step(
        field.ravel(), nodes, lambda t, p: speed, readout, 0.0, dt, iterations=0
    )
    scale = float(jnp.max(jnp.abs(reference)))
    assert float(jnp.max(jnp.abs(step - reference.ravel()))) <= 1e-13 * scale


@pytest.mark.parametrize(
    ("scheme", "peak", "error"),
    # measured in float64: linear 0.552 and 0.448, cubic B-spline 0.997 and 0.0034
    [(Linear(), (0.5, 0.6), (0.4, 0.5)), (BSpline(3), (0.99, 1.0), (0.0, 0.01))],
    ids=["linear", "bspline3"],
)
def test_translation_gate(scheme, peak, error):
    q, exact = _translate(scheme)
    assert peak[0] <= float(q.max()) <= peak[1]
    assert error[0] <= float(jnp.abs(q - exact).max()) <= error[1]


def test_translation_keeps_float32():
    q, exact = _translate(BSpline(3), dtype=jnp.float32)
    assert q.dtype == jnp.float32
    assert float(jnp.abs(q - exact).max()) <= 0.01


def _rotation_error(steps: int, iterations: int) -> float:
    """Max error of an affine field after one revolution of a solid-body flow.

    The rate ω(t) = 2π (1 + ½ cos 2πt) varies in time and integrates to 2π
    over [0, 1], so the exact field returns to its start. Linear read-out
    with extrapolation reproduces an affine field exactly, so the error is
    the trajectory's alone (a cone's read-out error, O(h²/Δt), would grow as
    Δt shrinks and hide the slope).
    """
    m = 6
    axis = jnp.linspace(0.0, 1.0, m)
    grid = RectilinearGrid((Axis(axis), Axis(axis)))
    rows, cols = jnp.meshgrid(axis, axis, indexing="ij")
    nodes = jnp.column_stack([rows.ravel(), cols.ravel()])
    field = 2.0 * rows.ravel() - 3.0 * cols.ravel() + 1.0
    policies = Policies(outside="extrapolate")

    def velocity(t, p):
        rate = 2 * jnp.pi * (1 + 0.5 * jnp.cos(2 * jnp.pi * t))
        # u = ω (−(x₁ − ½), x₀ − ½)
        return rate * jnp.column_stack([-(p[:, 1] - 0.5), p[:, 0] - 0.5])

    def readout(departure):
        return Linear().plan(grid, Points(departure), policies=policies)

    dt = 1.0 / steps

    def body(q, step):
        t = step * dt
        q_new = im.semi_lagrangian_step(
            q, nodes, velocity, readout, t, dt, iterations=iterations
        )
        return q_new, None

    q, _ = jax.lax.scan(body, field, jnp.arange(steps, dtype=float))
    return float(jnp.abs(q - field).max())


@pytest.mark.slow
@pytest.mark.parametrize(("iterations", "order"), [(0, 1), (1, 2), (2, 2)])
def test_trajectory_order_in_time(iterations, order):
    errors = {1.0 / k: _rotation_error(k, iterations) for k in (64, 128, 256, 512)}
    assert_order(errors, order, slack=0.3)


def test_above_the_cfl_limit_stays_bounded():
    # CFL up to 3 with a spatially varying speed and a linear read-out
    n = 64
    x, axis = _periodic_line(n)
    nodes = einx.id("node -> node 1", x)
    dt = 3.0 / n
    field = jnp.asarray(np.random.default_rng(1).standard_normal(n))

    def velocity(t, p):
        return 0.6 + 0.4 * jnp.sin(2 * jnp.pi * p)

    def readout(departure):
        return Linear().plan(axis, Points(departure))

    q = im.semi_lagrangian_step(field, nodes, velocity, readout, 0.0, dt)
    assert float(q.min()) >= float(field.min())
    assert float(q.max()) <= float(field.max())


def _smooth_problem():
    n = 16
    x, axis = _periodic_line(n)
    nodes = einx.id("node -> node 1", x)
    field = jnp.sin(2 * jnp.pi * x) + 0.3 * jnp.cos(6 * jnp.pi * x)

    def readout(departure):
        return BSpline(3).plan(axis, Points(departure))

    def velocity_at(speed):
        def velocity(t, p):
            return speed * (1 + 0.3 * jnp.sin(2 * jnp.pi * p))

        return velocity

    return field, nodes, readout, velocity_at


def test_gradients_match_finite_differences():
    field, nodes, readout, velocity_at = _smooth_problem()
    dt = 0.05

    def of_field(q):
        return im.semi_lagrangian_step(q, nodes, velocity_at(1.0), readout, 0.0, dt)

    def of_speed(speed):
        velocity = velocity_at(speed)
        return im.semi_lagrangian_step(field, nodes, velocity, readout, 0.0, dt)

    assert_grad_matches_fd(of_field, field)
    assert_grad_matches_fd(of_speed, jnp.asarray(1.0))


def test_jit_and_vmap_over_a_batch_match_a_loop():
    field, nodes, readout, velocity_at = _smooth_problem()
    batch = jnp.stack([field, 2.0 * field, jnp.roll(field, 3)])

    def step(q):
        return im.semi_lagrangian_step(q, nodes, velocity_at(1.0), readout, 0.0, 0.05)

    assert_traces(step, batch)
    # a leading batch axis goes straight through the read-out too
    assert jnp.allclose(step(batch), jax.vmap(step)(batch), rtol=1e-13)


def test_solver_under_diffeqsolve_matches_the_step():
    field, nodes, readout, velocity_at = _smooth_problem()
    velocity = velocity_at(1.0)
    solver = im.SemiLagrangianSolver(nodes=nodes, readout=readout)
    term = im.SemiLagrangianTerm(velocity)
    solution = dfx.diffeqsolve(term, solver, 0.0, 0.2, 0.05, field)

    q = field
    for k in range(4):
        q = im.semi_lagrangian_step(q, nodes, velocity, readout, 0.05 * k, 0.05)
    assert jnp.allclose(solution.ys[0], q, rtol=1e-12, atol=1e-12)
    assert solver.order(term) == 2
    assert im.SemiLagrangianSolver(nodes, readout, iterations=0).order(term) == 1
    with pytest.raises(NotImplementedError, match="dt0"):
        solver.func(term, 0.0, field, None)


@pytest.mark.parametrize("iterations", [-1, True])
def test_bad_iterations_raise(iterations):
    _, nodes, readout, velocity_at = _smooth_problem()
    with pytest.raises(ValidationError, match="'iterations'"):
        im.departure_points(nodes, velocity_at(1.0), 0.0, 0.1, iterations=iterations)
    with pytest.raises(ValidationError, match="'iterations'"):
        im.SemiLagrangianSolver(nodes, readout, iterations=iterations)
