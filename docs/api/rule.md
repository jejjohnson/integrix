# Rules

A rule integrates by weights against samples, $I \approx \sum_n w_n f(x_n)$. The weights come from functax's rules, from a tuple of 1-D rules (a tensor rule, one axis each), or from a plain array such as a spectral response or cell areas; integrix only sums with them. Every sum is pairwise, so a float32 sum over a long spectrum does not drift.

Both `integrate` and `weighted_mean` take `where=`, a boolean mask that broadcasts against the values (a plume over an image, the valid levels of each column), and `nan_policy=`: `"propagate"` (the default, so a NaN among the kept samples gives NaN) or `"omit"` (NaN samples are left out, like `np.nansum`). A dropped sample adds nothing to the sum and gets an exactly zero, finite gradient, and `weighted_mean` renormalises over the samples it keeps.

The theory is in the [Rules](../rules/) chapter of the mathematical reference.

::: integrix.rule
