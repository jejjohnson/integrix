---
name: add-stepper
description: Add a time stepper to integrix.march — a diffrax solver subclass for a single-step scheme, or an AbstractStepper run by integrix's own loop for a multistep or split scheme, with its pure step function, a parity test when it moves in from finitevolX or sesmax-core, the order-of-accuracy slope, the stability limit, the jit/vmap/grad smoke test, the API page and the capability index. Use when asked to add, port or implement a time-stepping scheme or step-size control in src/integrix.
---

# Add a stepper

Read "The contracts", "JAX and Equinox" and "Boundaries" in `AGENTS.md` first; this is the step-by-step. Then read the live issue (`gh issue view <N>`) and design decision 8 ([#18](https://github.com/jejjohnson/integrix/issues/18)).

## 1. Make sure it does not exist yet

- Search `docs/api/capabilities.md`, including the diffrax section: a scheme diffrax already ships (`Tsit5`, `Heun`, `Dopri5`, …) is used, not rewritten.
- A stepper moving in from finitevolX or sesmax-core is moved, not rewritten: keep its behaviour, and add a parity test against the source at the commit its issue pins.

## 2. Pick its kind

| The scheme… | Kind | Runs under |
|---|---|---|
| needs only the current state (Euler, RK2–RK4, SSP-RK, IMEX) | a subclass of the matching diffrax solver (`dfx.AbstractERK`, `dfx.AbstractSolver`) | `dfx.diffeqsolve`, its controllers and adjoints |
| carries history between steps (AB2, leapfrog with a Robert–Asselin filter, split-explicit) | an `ix.march.AbstractStepper` | `ix.march.run`, integrix's `lax.scan` loop |

Both live in `src/integrix/_src/march/`, one scheme family per file, re-exported from `integrix.march`. The pure step function (`state → state`) sits next to the class and is exported too.

## 3. Write it

- **Vector field** `vector_field(t, y, args)`, the diffrax convention; history lives in the loop carry or the diffrax solver state, never on the module.
- **Configuration** (an order, a filter coefficient α, a number of substeps) is `eqx.field(static=True)`, checked in `__check_init__` with `ValidationError` naming the field.
- **Order.** A diffrax subclass declares `order` (and `strong_order` where it applies) exactly as diffrax's protocol asks.
- **It traces.** No Python control flow on traced values, static shapes, no host round-trip; the library never calls `jax.jit`.
- **Precision is the caller's.** Butcher coefficients are Python scalars or made in the state's dtype.
- **Docstring** in the `python-style` layout: the update in Unicode (yₙ₊₁ = yₙ + h ∑ᵢ bᵢ kᵢ), the Butcher tableau or recurrence under `Algorithm:`, its order and stability limit, executable `Examples:` and `References:`.

## 4. Export and document

- Re-export from `integrix.march`'s sorted `__all__`; `docs/api/march.md` picks it up through its `:::` block.
- `uv run poe capabilities`.

## 5. Tests

In `tests/march/test_<family>.py`, with x64 on, using `tests/_harness.py`:

- **Order:** `assert_order` on y′ = λy with y(T) = y₀ e^{λT} over a halving sequence of h; the slope equals the declared order within the helper's slack.
- **Parity:** equal to the finitevolX or sesmax-core implementation at its pinned commit to 1e-12 relative, from fixtures stored in `tests/fixtures/` (the source library is never a test dependency).
- **Stability:** one step size inside the stated limit stays bounded and one outside grows, on the linear test problem.
- **Transforms and gradients:** `assert_traces` for `jit` and `vmap`; `assert_grad_matches_fd` through a short march.
- **Edges:** a pytree state, float32 state keeps float32, each invalid configuration raises `ValidationError`.
- **Tiers** ("Test tiers" in `AGENTS.md`): sweeps (orders, hypothesis, long rollouts, stiff solves) are `@pytest.mark.slow`, a ported parity suite is `integration`, and everything else stays fast; the fast tier alone must still clear the coverage gate.

## 6. Verify

```bash
uv run pytest --no-cov tests/march/ src/integrix/_src/march/ tests/test_public_api.py tests/test_capabilities.py -v
uv run poe test-cov
```

then the `pre-pr-check` skill.
