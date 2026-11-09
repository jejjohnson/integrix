"""finitevolX's tests for the pure IMEX-SSP2 step, run against integrix.

Ported from finitevolX ``tests/test_timestepping_imex.py`` at ``82090a3`` for
gate G1-port (#19): every test body is unchanged; only the import path moves
from ``finitevolx`` to ``integrix.march``. The split-explicit tests arrive
with #20.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from integrix.march import imex_ssp2_step


class TestIMEX_SSP2:
    """Test IMEX-SSP2 on a simple advection-diffusion-like split."""

    def test_pure_explicit_decay(self):
        """When implicit part is zero, IMEX reduces to explicit averaging."""
        y0 = jnp.array(1.0)
        dt = 0.01
        n_steps = 100

        def rhs_explicit(y):
            return -y

        def rhs_implicit(_y):
            return jnp.array(0.0)

        def implicit_solve(rhs, _gamma_dt):
            return rhs  # No implicit contribution

        y = y0
        for _ in range(n_steps):
            y = imex_ssp2_step(y, rhs_explicit, rhs_implicit, implicit_solve, dt)

        exact = float(jnp.exp(-1.0))
        np.testing.assert_allclose(float(y), exact, rtol=1e-2)

    def test_pure_implicit_decay(self):
        """When explicit part is zero, IMEX handles implicit decay."""
        y0 = jnp.array(1.0)
        dt = 0.01
        n_steps = 100

        def rhs_explicit(_y):
            return jnp.array(0.0)

        def rhs_implicit(y):
            return -y

        def implicit_solve(rhs, gamma_dt):
            # Solve: Y - gamma_dt * (-Y) = rhs => Y * (1 + gamma_dt) = rhs
            return rhs / (1.0 + gamma_dt)

        y = y0
        for _ in range(n_steps):
            y = imex_ssp2_step(y, rhs_explicit, rhs_implicit, implicit_solve, dt)

        exact = float(jnp.exp(-1.0))
        np.testing.assert_allclose(float(y), exact, rtol=5e-2)

    def test_mixed_explicit_implicit(self):
        """Both explicit and implicit parts active simultaneously.

        Split dy/dt = -0.3*y (explicit) + -0.7*y (implicit).
        Total: dy/dt = -y => y(t) = exp(-t).
        """
        y0 = jnp.array(1.0)
        dt = 0.01
        n_steps = 100

        def rhs_explicit(y):
            return -0.3 * y

        def rhs_implicit(y):
            return -0.7 * y

        def implicit_solve(rhs, gamma_dt):
            # Solve: Y - gamma_dt * (-0.7*Y) = rhs => Y(1 + 0.7*gamma_dt) = rhs
            return rhs / (1.0 + 0.7 * gamma_dt)

        y = y0
        for _ in range(n_steps):
            y = imex_ssp2_step(y, rhs_explicit, rhs_implicit, implicit_solve, dt)

        exact = float(jnp.exp(-1.0))
        np.testing.assert_allclose(float(y), exact, rtol=5e-2)

    @staticmethod
    def _rates(lam_e, lam_i):
        """Observed orders of y' = (lam_e + lam_i) y on [0, 1], N = 20..160."""
        exact = np.exp(lam_e + lam_i)
        errors = []
        for n in [20, 40, 80, 160]:
            dt = 1.0 / n
            y = jnp.array(1.0)
            for _ in range(n):
                y = imex_ssp2_step(
                    y,
                    lambda y: lam_e * y,
                    lambda y: lam_i * y,
                    lambda rhs, gdt: rhs / (1.0 - lam_i * gdt),
                    dt,
                )
            errors.append(abs(float(y) - exact))
        return np.log2(np.array(errors[:-1]) / np.array(errors[1:]))

    def test_mixed_convergence(self):
        """Second order on a mixed explicit + implicit split (issue #311)."""
        rates = self._rates(-0.4, -0.6)
        assert np.all(rates > 1.8), f"mixed rates {rates}"

    def test_implicit_only_convergence(self):
        """Second order when only the implicit part is active (issue #311)."""
        rates = self._rates(0.0, -1.0)
        assert np.all(rates >= 1.9), f"implicit-only rates {rates}"

    def test_explicit_only_convergence(self):
        """Second order when only the explicit part is active."""
        rates = self._rates(-1.0, 0.0)
        assert np.all(rates >= 1.9), f"explicit-only rates {rates}"

    def test_pytree_state(self):
        """IMEX works with tuple pytree states."""
        y0 = (jnp.array(1.0), jnp.array(2.0))
        dt = 0.01

        def rhs_explicit(y):
            return jax.tree.map(lambda x: -x, y)

        def rhs_implicit(_y):
            return jax.tree.map(jnp.zeros_like, _y)

        def implicit_solve(rhs, _gamma_dt):
            return rhs

        y = imex_ssp2_step(y0, rhs_explicit, rhs_implicit, implicit_solve, dt)
        assert isinstance(y, tuple)
        assert len(y) == 2

    def test_jit_compatible(self):
        """IMEX step compiles under jax.jit."""
        y0 = jnp.array(1.0)

        def rhs_e(y):
            return -y

        def rhs_i(_y):
            return jnp.array(0.0)

        def solve(rhs, _gdt):
            return rhs

        step = jax.jit(lambda y: imex_ssp2_step(y, rhs_e, rhs_i, solve, 0.01))
        y1 = step(y0)
        y1_ref = imex_ssp2_step(y0, rhs_e, rhs_i, solve, 0.01)
        np.testing.assert_allclose(y1, y1_ref, atol=1e-15)
