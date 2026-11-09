# Rules

A rule integrates by weights against samples, $I \approx \sum_n w_n f(x_n)$. The weights come from functax's rules, from a tuple of 1-D rules (a tensor rule, one axis each), or from a plain array such as a spectral response or cell areas; integrix only sums with them. Every sum is pairwise, so a float32 sum over a long spectrum does not drift.

::: integrix.rule
