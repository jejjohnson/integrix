# integrix

> Integration for JAX: rules, scans and time steppers, with shared adaptivity and adjoints.

integrix computes one integral, $I = \int_\Omega f(x) \mathrm{d}\mu(x)$, in three ways: a rule sums weights against samples, $I \approx \vect{w}^\top \vect{f}$; a scan builds the running integral $F_k = F_{k-1} + w_k f_k$; and a march steps an integrand that depends on the solution, $\vect{y}_{n+1} = \Phi(\vect{y}_n, t_n, h)$. Adaptivity and adjoints are shared by all three.

integrix sits at Core tier 1 of the GeoStack: it sums with functax's rules, solves with gaussx, and extends diffrax with the stack's steppers. The design is in the GeoStack [integrix design page](https://github.com/jejjohnson/geostack/blob/main/docs/design/libraries/integrix.md), and the backlog is tracked in waves under [geostack#136](https://github.com/jejjohnson/geostack/issues/136).

:::{note} Status
integrix is in Wave 0: the package, its CI, the reference harness, rules and scans. The steppers arrive in Wave 1.
:::

## Installation

::::{tab-set}
:::{tab-item} From GitHub

```bash
pip install "integrix @ git+https://github.com/jejjohnson/integrix.git"
```
:::
:::{tab-item} From source

```bash
git clone https://github.com/jejjohnson/integrix.git
cd integrix
uv run poe install
```
:::
::::

## Next steps

- [Quickstart](guide/quickstart.md): the three operations and where each lives
- [API reference](https://jejjohnson.github.io/integrix/reference/): every module and the capability index
