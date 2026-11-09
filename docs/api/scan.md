# Scans

A scan builds the running integral of samples along one axis, $F_k = F_{k-1} + \Delta_k$ with the trapezoid increments $\Delta_k = \frac{1}{2}(x_k - x_{k-1})(f_{k-1} + f_k)$, by `jax.lax.associative_scan` in $O(\log N)$ depth. With `reverse=True` it integrates from the far end, as optical depth from the top of a column does; the increments are reversed, never the nodes.

::: integrix.scan
