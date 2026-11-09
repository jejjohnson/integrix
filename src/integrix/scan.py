"""Scans: cumulative integrals, forward and reverse, by associative scan.

`cumulative` builds the running integral Fₖ of samples along one axis in
O(log N) depth, from the first node or (``reverse=True``) from the last, as
optical depth down a column needs.
"""

from __future__ import annotations

from integrix._src.scan._cumulative import cumulative


__all__ = [
    "cumulative",
]
