"""Integration for JAX: rules, scans, marches, adaptivity, adjoints and expectations.

integrix computes one integral, I = ∫ f dμ, in three ways: a rule sums weights
against samples, a scan builds the running integral, and a march steps an
integrand that depends on the solution. Each operation lives in its own public
module, and each module appears with the first feature that needs it.

Examples:
    >>> import integrix
    >>> major, minor, patch = integrix.__version__.split(".")[:3]
    >>> all(part.isdigit() for part in (major, minor, patch))
    True
"""

from __future__ import annotations

from integrix.exceptions import IntegrixError, ValidationError


__version__ = "0.0.2"

__all__ = [
    "IntegrixError",
    "ValidationError",
    "__version__",
]
