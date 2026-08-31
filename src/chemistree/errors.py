"""Structured errors for reference resolution.

A lookup returns exactly one result or raises ``NotFound``, so the caller reacts to
"nothing matched" rather than receiving a guess.
"""

from __future__ import annotations


class ResolutionError(Exception):
    """A reference could not be resolved to a single node or atom."""


class NotFound(ResolutionError):
    """No candidate matched the reference."""
