# API Reference

Everything listed in `integrix.__all__` is importable from the top level, and each area of the library is a public module that re-exports its private implementation:

```python
import integrix as ix

ix.IntegrixError  # the base of every integrix error
ix.rule.integrate  # weights against samples
ix.scan.cumulative  # running integrals
```

The [capability index](capabilities.md) lists every public name on one page, with a one-line summary each, together with the public API of JAX, Equinox, diffrax, lineax, optimistix, einx, functax and gaussx.

## Modules

| Module | Contents |
|---|---|
| [Adjoints](adjoint.md) | `Checkpointed` and `checkpointed_scan`, `Truncated` and `truncated_scan`, `to_diffrax_adjoint` (Wave 2) |
| [Marches](march.md) | the diffrax-solver steppers (Euler, Heun, SSP-RK2/3, RK4, SSP-RK(10,4), IMEX-SSP2, AB3), `AbstractStepper` and `run` for AB2, leapfrog-RAF and split-explicit, their step functions, and the matrix-free IMEX solver, its step control and the CFL bound (Wave 1) |
| [Rules](rule.md) | `Integral`, `integrate`, `weighted_mean`: weights against samples over any axes (Wave 0) |
| [Scans](scan.md) | `cumulative`: running integrals, forward and reverse, by associative scan (Wave 0) |
| [Type aliases](typing.md) | `Axes`, `RuleLike` |
| [Exceptions](exceptions.md) | `IntegrixError`, `ValidationError` |

## Package overview

::: integrix
    options:
      members: false
      show_root_heading: false
      show_source: false
