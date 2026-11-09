# integrix

[![Tests](https://github.com/jejjohnson/integrix/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/integrix/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/integrix/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/integrix/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/integrix/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/integrix/actions/workflows/typecheck.yml)
[![Deploy Docs](https://github.com/jejjohnson/integrix/actions/workflows/pages.yml/badge.svg)](https://github.com/jejjohnson/integrix/actions/workflows/pages.yml)
[![codecov](https://codecov.io/gh/jejjohnson/integrix/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/integrix)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)

Author: J. Emmanuel Johnson

Integration for JAX: rules, scans and time steppers, with shared adaptivity and adjoints.

integrix computes one integral, I = ∫ f dμ, in three ways: a rule sums weights against samples (a band radiance, a column amount, an integrated mass), a scan builds the running integral (optical depth down a column), and a march steps an integrand that depends on the solution (time stepping). Adaptivity and adjoints are shared by all three. It sits at Core tier 1 of the [GeoStack](https://github.com/jejjohnson/geostack): it sums with functax's rules, solves with gaussx, and extends diffrax with the stack's steppers. It takes over finitevolX's generic steppers and pipekit-jax's adjoints, and replaces the hand-written trapezoid sums in plumax, radtrax and xrtoolz.

## Status

integrix is in Wave 0: the package, its CI, the reference test harness, `integrix.rule` and `integrix.scan`. The design is the GeoStack [integrix design page](https://github.com/jejjohnson/geostack/blob/main/docs/design/libraries/integrix.md), and the backlog is six wave epics under [geostack#136](https://github.com/jejjohnson/geostack/issues/136).

| Wave | Delivers | Epic |
|---|---|---|
| 0 | the package, the reference harness, `integrix.rule` and `integrix.scan` | [#8](https://github.com/jejjohnson/integrix/issues/8) |
| 1 | `integrix.march`: steppers moved in from finitevolX and sesmax-core | [#9](https://github.com/jejjohnson/integrix/issues/9) |
| 2 | `integrix.adjoint`: checkpointed and truncated backward sweeps | [#10](https://github.com/jejjohnson/integrix/issues/10) |
| 3 | `integrix.adapt`: adaptive Gauss–Kronrod, tanh–sinh and Romberg | [#11](https://github.com/jejjohnson/integrix/issues/11) |
| 4 | `integrix.expect`: Monte Carlo and quasi-Monte Carlo expectations | [#12](https://github.com/jejjohnson/integrix/issues/12) |
| 5 | the stack's libraries switch to integrix | [#13](https://github.com/jejjohnson/integrix/issues/13) |

## Installation

integrix is not on PyPI; the stack pins it by git tag.

```bash
pip install "integrix @ git+https://github.com/jejjohnson/integrix.git"
```

## Development

```bash
git clone https://github.com/jejjohnson/integrix.git
cd integrix
uv run poe install     # every dependency group + the pre-commit hooks
uv run poe check       # lint, format check, typecheck, tests with the 90 % coverage gate
uv run poe docs        # the MyST prose and the Zensical API reference, assembled into public/
```

`uv run poe` lists every task. [`AGENTS.md`](AGENTS.md) holds the rules for contributors and coding agents: the module map, the contracts, the tests that enforce them and the house style. [`docs/contributing.md`](docs/contributing.md) has the label taxonomy and the epic model.

## License

MIT; see [`LICENSE`](LICENSE).
