"""Exception hierarchy for `integrix`.

Every error raised by this package derives from
[`IntegrixError`][integrix.IntegrixError], so a caller can trap the whole
package with a single ``except`` clause. Each concrete error also inherits from
the closest standard-library exception, so code written against the stdlib
keeps working without importing anything from this package.
"""

from __future__ import annotations


__all__ = [
    "IntegrixError",
    "ValidationError",
]


class IntegrixError(Exception):
    """Base class for every exception raised by ``integrix``.

    Examples:
        >>> from integrix import IntegrixError, ValidationError
        >>> issubclass(ValidationError, IntegrixError)
        True
    """


class ValidationError(IntegrixError, ValueError):
    """Raised when an argument fails validation.

    Also a `ValueError`, so ``except ValueError`` still catches it.

    Examples:
        >>> from integrix import ValidationError
        >>> issubclass(ValidationError, ValueError)
        True
    """
