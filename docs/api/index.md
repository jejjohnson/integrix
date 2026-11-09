# API Reference

Everything listed in `integrix.__all__` is importable from the top level, and each area of the library is a public module that re-exports its private implementation:

```python
import integrix as ix

ix.IntegrixError  # the base of every integrix error
ix.rule.integrate  # weights against samples
```

The [capability index](capabilities.md) lists every public name on one page, with a one-line summary each, together with the public API of JAX, Equinox, diffrax, lineax, optimistix, einx, functax and gaussx.

## Modules

| Module | Contents |
|---|---|
| [Rules](rule.md) | `Integral`, `integrate`, `weighted_mean`: weights against samples over any axes (Wave 0) |
| [Type aliases](typing.md) | `Axes`, `RuleLike` |
| [Exceptions](exceptions.md) | `IntegrixError`, `ValidationError` |

## Package overview

::: integrix
    options:
      members: false
      show_root_heading: false
      show_source: false
