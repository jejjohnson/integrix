# Code Review Agent Instructions

Standing instructions for **all** agents performing code reviews on this repository. Read "Reuse before you write", "The contracts" and "What enforces them" in [`AGENTS.md`](AGENTS.md) first; this file is the checklist and the report format. Most defects worth finding are a contract broken in a way the tests cannot see (a gather that silently clamps targets outside the grid, a `where` whose untaken branch poisons the gradient, a silent dtype cast, an adjoint that is not the transpose, a docstring that promises what the code does not do) or a re-implementation of something the package already has, not style.

---

## How to Obtain the Diff

Use the following command to get the diff for review:

```bash
BASE_BRANCH="$(git rev-parse --verify main >/dev/null 2>&1 && echo main || echo master)"
git --no-pager diff --no-prefix --unified=100000 --minimal $(git merge-base --fork-point "$BASE_BRANCH")...HEAD
```

If that fails (e.g. detached HEAD, shallow clone), fall back to:

```bash
git --no-pager diff --no-prefix --unified=100000 --minimal "$BASE_BRANCH"...HEAD
```

### Reading the diff

| Prefix | Meaning |
|--------|---------|
| `+` | Added line |
| `-` | Removed line |
| ` ` (space) | Unchanged context |
| `@@` | Hunk header |

---

## Review Checklist

Skip anything ruff, ty or the tests already enforce (formatting, import order, `__all__` ordering, doctest output, the coverage gate); review what they cannot see.

### 1. Reuse and boundaries

- Every function, class or module the diff **adds** has been checked against [`docs/api/capabilities.md`](docs/api/capabilities.md) and the private helpers in "Reuse before you write" in `AGENTS.md`. Re-implemented quadrature weights (functax's), a Gaussian rule (gaussx's), a sum, scan or stepping loop the package already has, or a copied stepper is a **High** finding, with the existing name to use.
- No new runtime dependency without discussion; an optional one sits behind an extra and an in-function import that names the extra. `uv.lock` is updated with any dependency change.
- Imports keep pointing one way (the module table in `AGENTS.md`). integrix is Core tier 1: it may import functax and gaussx and extend diffrax; an import of pipekit, pipekit-cycle, argminax, finitevolX or any GeoModels library is a **Critical** finding.
- SciPy (beyond what JAX brings), quadax and hypothesis stay in the test group: a runtime import of any of them is a **Critical** finding.
- A rename or removal of a public name keeps the old name working with a `DeprecationWarning` for at least one release.

### 2. The contracts

- **Public functions and modules** validate once, at the boundary (`__check_init__` for fields, `eqx.error_if` for traced values), raise `ValidationError` whose message names the argument, never mutate their input, preserve the input dtype, and trace under `jit`, `vmap` and `grad` with static shapes.
- **Traced code** has no host RNG, no Python branching on traced values and no `jax.jit` of its own; the caller jits.
- **Long sums** are pairwise, and weights take the values' dtype.
- **`jnp.where`** guards the untaken branch too, or its `nan` reaches the gradient.
- **Rules and scans** integrate their exact polynomials to 1e-13 relative (G0-exact) and equal `scipy.integrate.trapezoid` or `cumulative_trapezoid` to 1e-14 on random non-uniform nodes (G0-scipy); **steppers** show their declared order as a log–log slope; `jit`, `vmap` and `grad` agree with an eager loop.
- **Solves** go through lineax, optimistix or gaussx; `jnp.linalg.inv` and `jnp.linalg.solve` are findings.
- **Exceptions** derive from `IntegrixError` and the closest stdlib exception; errors are chained (`raise … from err`).

### 3. Public API and documentation

- New names: logic in `integrix._src.<area>`, re-exported from the area module's `__all__`; a new module in `SUBMODULES` or a new package in `SUBPACKAGES` (`tests/test_public_api.py`), with its `docs/api/<module>.md` page, a `nav` entry in `mkdocs.yml` and a row in `docs/api/index.md`; `docs/api/capabilities.md` regenerated (`uv run poe capabilities`).
- Docstrings in the `python-style` layout: summary, the defining equation in Unicode (`eman-style-math`; `$…$` in a docstring is a finding), `Algorithm:` for a method, `Args:` with meaning and symbol, `Returns:` and `Raises:` matching the code, and two or three executable `Examples:` (typical, edge, composition) whose output is what the code prints (shapes, `allclose` booleans or reprs, never raw floats).
- Maths in MyST and MkDocs pages uses the `eman-style-math` macros, defines every symbol with its shape, keeps each expression on one line and has no spacing macros (`\,`, `\quad`, …).
- Prose pages: MyST directives, not MkDocs-Material syntax; API links by the top-level name (`xref:api#integrix.<Name>`); a new page or notebook in the `docs/myst.yml` toc with a basename unique across `docs/`.
- Notebooks committed executed, with no error outputs, and every number quoted in the prose matching the printed output.

### 4. Tests

- Each behaviour is checked against a closed form, a hand computation or a published value; edge cases (a single node, several axes, a descending axis, float32 inputs, invalid arguments) have their own tests. A public forward function has the four `python-style` tests: a reference, hypothesis properties, gradient check, transform invariance.
- New code is covered (the 90 % gate is global, so a well-covered package can hide an untested module: look at the diff, not the total).
- Randomness, if any, is seeded; a statistical assertion is bounded by the statistic's own sampling distribution, with the bound's source in a comment.
- Tier markers right: unmarked under ~1 s, `slow` above, `integration` for end-to-end workflows and optional dependencies.

### 5. Modern Python and dependencies

- `from __future__ import annotations`; type hints on every public function; `X | None`, built-in generics, PEP 695 `type` aliases.
- `eqx.Module` for anything holding arrays (arrays as leaves, configuration as static fields); frozen dataclasses for host-side records that are never traced; `Enum` / `StrEnum` for fixed sets of options; `pathlib`; f-strings; context managers for resources.
- Specific exceptions, chained with `raise … from …`; guard clauses over deep nesting.
- `python-style` comments and naming: comments above the line, never beside it; a Unicode equation above each mathematical line; descriptive names, with the symbol only in the comment; one operation per line. Its JAX rules too ("JAX and Equinox" in `AGENTS.md`): jaxtyping dims, einx, lineax, `eqx.error_if`, `key` keyword-only and last.
- New CLIs and scripts use `cyclopts` and `loguru`, not argparse, click, typer, `print` progress or stdlib `logging`.
- A dependency, when one is justified, follows the preferences in `.github/instructions/python-coding.instructions.md`.

### 6. Security

- No secrets or credentials in code, tests, notebooks or docs.
- `subprocess` without `shell=True`; file paths taken from users are not joined into places they could escape.

---

## Package-Specific Checks

Every example here runs, with `jax_enable_x64` on as in the test suite.

### Sum pairwise, not left to right

```python
import jax
import jax.numpy as jnp

values = jnp.full(100_000, 0.1, dtype=jnp.float32)


# ❌ A running float32 accumulation drifts: each step adds an increment far
# smaller than the total, and the rounding errors pile up
def total(values):
    carry, _ = jax.lax.scan(lambda acc, v: (acc + v, None), jnp.float32(0.0), values)
    return carry


# ✅ A tree reduction rounds O(log N) times per value, not O(N)
def total(values):
    while values.shape[0] > 1:
        if values.shape[0] % 2:
            values = jnp.concatenate([values, jnp.zeros(1, values.dtype)])
        values = values[0::2] + values[1::2]
    return values[0]
```

### Guard both branches of `where`

```python
import jax
import jax.numpy as jnp


# ❌ The untaken branch still runs: jax.grad(...)(0.0) is nan, not 0.0
def inverse_distance(d):
    return jnp.where(d > 0, 1.0 / d, 0.0)


# ✅ Feed the untaken branch a safe value: jax.grad(...)(0.0) is 0.0
def inverse_distance(d):
    positive = d > 0
    safe_d = jnp.where(positive, d, 1.0)
    return jnp.where(positive, 1.0 / safe_d, 0.0)
```

### Weights take the values' dtype

```python
import jax.numpy as jnp

values = jnp.ones(4, dtype=jnp.float32)
weights = jnp.full(4, 0.25)  # float64 under x64


# ❌ A float32 spectrum comes back as float64
def integrate(weights, values):
    return jnp.sum(weights * values)


# ✅ Weights are cast to the values' dtype, never the other way
def integrate(weights, values):
    weights = weights.astype(values.dtype)
    return jnp.sum(weights * values)
```

### A reversed scan reverses the increments, not the nodes

```python
import jax.numpy as jnp

nodes = jnp.array([0.0, 1.0, 3.0])
values = jnp.array([1.0, 2.0, 3.0])
# Δₖ = ½ (xₖ − xₖ₋₁) (fₖ₋₁ + fₖ): the trapezoid increment of each interval
increments = 0.5 * (nodes[1:] - nodes[:-1]) * (values[:-1] + values[1:])

# ❌ Flipping the nodes makes every width negative, so the integral changes sign
flipped = nodes[::-1]
wrong = jnp.cumsum(
    0.5 * (flipped[1:] - flipped[:-1]) * (values[::-1][:-1] + values[::-1][1:])
)

# ✅ Fₖ = ∑ⱼ₌ₖ Δⱼ: the same increments, summed from the far end
from_top = jnp.cumsum(increments[::-1])[::-1]
```

### Solve, never invert

```python
import jax.numpy as jnp
import lineax as lx

matrix = jnp.array([[4.0, 1.0], [1.0, 3.0]])
rhs = jnp.array([1.0, 2.0])

# ❌ Forms the inverse: slower, less accurate, and blind to structure
solution = jnp.linalg.inv(matrix) @ rhs

# ✅ A lineax solve, with the structure declared
operator = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
solution = lx.linear_solve(operator, rhs, solver=lx.Cholesky()).value
```

### No host RNG

```python
import jax
import numpy as np


# ❌ NumPy draws once, at trace time: under jit every call reuses one sample
def estimate(f, n):
    return f(np.random.default_rng().standard_normal(n)).mean()


# ✅ A key argument, keyword-only and last; the draw is part of the trace
def estimate(f, n, *, key):
    return f(jax.random.normal(key, (n,))).mean()
```

---

## Output Format

Format each review using this structure:

````
# Code Review for ${feature_description}

Overview of the changes, including the purpose, context, and files involved.

## Suggestions

### ${emoji} ${Summary of suggestion with necessary context}

* **Priority**: ${priority_emoji} ${priority_label}
* **File**: `${relative/path/to/file.py}`
* **Line(s)**: ${line_numbers}
* **Details**: Explanation of the issue and why it matters
* **Current Code**:
  ```python
  # problematic code
  ```
* **Suggested Change**:
  ```python
  # improved code with explanation
  ```

### (additional suggestions…)

## Summary

Brief summary of overall code quality and key action items.
````

---

## Priority Levels

| Emoji | Level | Use when |
|-------|-------|----------|
| 🔥 | **Critical** | Bugs, security issues, or code that will fail |
| ⚠️ | **High** | Significant issues affecting maintainability or correctness |
| 🟡 | **Medium** | Improvements for code quality or consistency |
| 🟢 | **Low** | Minor polish or optional enhancements |

## Suggestion Type Emojis

Prefix each suggestion title with a type indicator:

| Emoji | Type |
|-------|------|
| 🐛 | Bug or potential bug |
| 🔒 | Security concern |
| 🔧 | Change request (must fix) |
| ♻️ | Refactor suggestion |
| 📝 | Documentation improvement |
| 🎨 | Style / formatting issue |
| ⚡ | Performance consideration |
| 🧪 | Testing suggestion |
| ❓ | Question or clarification needed |
| ⛏️ | Nitpick (very minor) |
| 💭 | Design consideration |
| 👍 | Positive feedback (highlight good patterns) |
| 🌱 | Future consideration (not blocking) |

---

## Review Tone

- Be **constructive** and **specific**
- **Acknowledge** good patterns and decisions (use 👍 liberally)
- Explain the *why* behind every suggestion
- Offer **concrete alternatives**, not just criticism
- Recognize that context matters — ask clarifying questions when needed
- Keep feedback **actionable**: every suggestion should have a clear next step
