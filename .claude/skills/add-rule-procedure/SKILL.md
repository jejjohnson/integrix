---
name: add-rule-procedure
description: Add a procedure that sums, scans, adapts or takes an expectation with quadrature weights to integrix (integrix.rule, integrix.scan, integrix.adapt, integrix.expect) — logic under integrix/_src/<area>/, a re-export from the public area module, functax's rules as input rather than a private copy, pairwise sums, an executable docstring, the exactness, SciPy, float32-drift, gradient and jit/vmap tests from tests/_harness.py, the API page and the capability index. Use when asked to add, port or implement a quadrature, cumulative, adaptive or Monte Carlo procedure in src/integrix.
---

# Add a rule procedure

Read "The contracts", "JAX and Equinox" and "Boundaries" in `AGENTS.md` first; this is the step-by-step. Then read the live issue (`gh issue view <N>`): it names the reference, the tolerance and the gate, and it is newer than the design page.

## 1. Make sure it does not exist yet

- Search `docs/api/capabilities.md` (integrix and the upstream sections for JAX, Equinox, diffrax, lineax, optimistix, functax and gaussx) and the "Reuse before you write" table in `AGENTS.md`.
- Nodes and weights are functax's (decision 3): a procedure takes a rule as an argument and reads its `nodes` and `weights`; it never builds trapezoid, Simpson or Gauss weights itself. A Gaussian expectation with a closed rule is gaussx's.

## 2. Pick its home

| It is… | Private home | Public module |
|---|---|---|
| A weighted sum over one axis or several | `src/integrix/_src/rule/` | `integrix.rule` |
| A running (cumulative) integral | `src/integrix/_src/scan/` | `integrix.scan` |
| An adaptive integral with an error estimate | `src/integrix/_src/adapt/` | `integrix.adapt` |
| An expectation from samples or draws | `src/integrix/_src/expect/` | `integrix.expect` |

One concept per file. The first feature in an area creates the public module, `src/integrix/<area>.py`, which only re-exports.

## 3. Write it

- **Imports** under the fixed aliases: `import einx`, `import equinox as eqx`, `import jax`, `import jax.numpy as jnp`, `from jaxtyping import Array, Float, Int`.
- **Results** are `eqx.Module`s (`Integral` from `integrix.rule`, with `value`, `error`, `n_evaluations`, `converged`); reuse it rather than returning a tuple.
- **Sums are pairwise.** A long sum is a tree reduction, not a left-to-right accumulation, so a float32 sum over a long spectrum does not drift; reuse the private pairwise helper of `integrix._src.rule` rather than writing another.
- **It traces.** No Python control flow on traced values, static output shapes, no host round-trip, no host RNG (a `key` argument instead). The library never calls `jax.jit`; the caller does.
- **Precision is the caller's.** Weights are cast to the values' dtype, never the other way, and constants are made in that dtype.
- **Validate once.** Static checks raise `ValidationError` naming the argument (`'axis' …`, `'weights' …`); checks on traced values use `eqx.error_if`.
- **Docstring** in the `python-style` layout: one-line summary (the capability-index entry), the defining equation in Unicode, `Algorithm:` with numbered steps, `Args:` with meaning and symbol, `Returns:`, `Raises:`, two or three `Examples:` whose outputs are shapes, `allclose` booleans or reprs, and `References:` when a paper defines the method.

## 4. Export and document

- Re-export from the area module's sorted `__all__`. Nothing is added to the top-level `integrix.__all__`.
- A **new area module** joins `SUBMODULES` in `tests/test_public_api.py`, and gets `docs/api/<area>.md` with a `::: integrix.<area>` block, a `nav` entry in `mkdocs.yml` and a row in `docs/api/index.md`.
- `uv run poe capabilities`.

## 5. Tests

In `tests/<area>/test_<name>.py`, with x64 on (from `tests/conftest.py`), using the helpers in `tests/_harness.py`:

- **Exactness:** `assert_exact_on_polynomials` up to the degree the rule must integrate exactly, at 1e-13 relative (G0-exact).
- **Reference:** `assert_matches_scipy` against the `scipy.integrate` function the issue names (`trapezoid`, `cumulative_trapezoid`, `simpson`, `quad`), or quadax for an adaptive rule, on random non-uniform nodes from a fixed seed.
- **Long sums:** `assert_float32_drift` over a long sample (10⁵ points).
- **Gradient:** `assert_grad_matches_fd` in the values and, where they are inputs, the weights.
- **Transforms:** `assert_traces` (eager, `jit` and `vmap` agree with a Python loop).
- **Estimators:** `assert_within_stderr`, never a tolerance tuned on one draw.
- **Edges:** a single node, several axes, a descending axis, float32 input keeps float32, each invalid argument raises `ValidationError` (`match=` on the argument name).
- **Tiers** ("Test tiers" in `AGENTS.md`): sweeps (orders, hypothesis, long rollouts, stiff solves) are `@pytest.mark.slow`, a ported parity suite is `integration`, and everything else stays fast; the fast tier alone must still clear the coverage gate.

## 6. Verify

```bash
uv run pytest --no-cov tests/<area>/ src/integrix/_src/<area>/ tests/test_public_api.py tests/test_capabilities.py -v
uv run poe test-cov
```

then the `pre-pr-check` skill.
