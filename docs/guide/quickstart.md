# Quickstart

## One integral, three operations

Every integral integrix computes is $I = \int_\Omega f(x) \mathrm{d}\mu(x)$, over a time interval, a column, a band, an image or a probability space. It reduces to one of three operations, each with its own module:

| Operation | Computes | Module | Arrives in |
|---|---|---|---|
| Rule | $I \approx \vect{w}^\top \vect{f}$, weights against samples | `integrix.rule` | Wave 0 |
| Scan | $F_k = F_{k-1} + w_k f_k$, a cumulative integral | `integrix.scan` | Wave 0 |
| March | $\vect{y}_{n+1} = \Phi(\vect{y}_n, t_n, h)$, a time step | `integrix.march` | Wave 1 |

## Who owns the weights

functax owns rules: a rule is data, its nodes and weights. integrix owns what is done with them (summing, scanning, adapting, marching and differentiating), and Gaussian expectations with a closed rule stay in gaussx.

## Errors

Every error integrix raises derives from [`IntegrixError`](xref:api#integrix.IntegrixError). Bad input raises [`ValidationError`](xref:api#integrix.ValidationError), which is also a `ValueError`, so an existing `except ValueError` still catches it.

```python
import integrix as ix

ix.__version__  # '0.0.1'
```
