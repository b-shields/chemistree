"""Structured errors for deterministic reference resolution.

A resolution step returns exactly one result or raises one of these, so the caller
can react to "nothing matched" or "several matched" rather than receive a guess.
"""

from __future__ import annotations


class ResolutionError(Exception):
    """A reference could not be resolved to a single node or atom."""


class NotFound(ResolutionError):
    """No candidate matched the reference."""


class Ambiguous(ResolutionError):
    """More than one candidate matched the reference.

    Attributes:
        candidates: Ids of the nodes that matched.
    """

    def __init__(self, message: str, candidates: list[int]):
        super().__init__(message)
        self.candidates = candidates
