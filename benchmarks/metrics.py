"""Pull per-case metrics from a Claude Code result and the output contract.

Claude Code's ``--output-format json`` prints one object per run whose ``usage``
is cumulative over the whole agentic loop. The context footprint — every token the
model had to process — is the uncached input plus the cache writes plus the cache
reads; cache only changes their price, not that they were context.
"""

from __future__ import annotations

import re

_FINAL_SMILES = re.compile(r"FINAL_SMILES:\s*(\S+)")


def context_footprint(usage: dict) -> int:
    """Total tokens the model processed: uncached input + cache writes + cache reads.

    Args:
        usage: The ``usage`` object from a Claude Code result.

    Returns:
        The summed token count.
    """
    return int(
        usage.get("input_tokens", 0)
        + usage.get("cache_creation_input_tokens", 0)
        + usage.get("cache_read_input_tokens", 0)
    )


def parse_result(result: dict) -> dict:
    """Extract the token, cost, turn, and timing metrics we track.

    Args:
        result: The decoded JSON object from ``claude -p --output-format json``.

    Returns:
        A flat dict of metrics, ready to store on a result row.
    """
    usage = result.get("usage", {})
    details = usage.get("output_tokens_details", {})
    return {
        "input_tokens": usage.get("input_tokens", 0),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
        "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
        "context_footprint": context_footprint(usage),
        "output_tokens": usage.get("output_tokens", 0),
        "thinking_tokens": details.get("thinking_tokens", 0),
        "cost_usd": result.get("total_cost_usd"),
        "num_turns": result.get("num_turns"),
        "duration_ms": result.get("duration_ms"),
        "api_error": bool(result.get("is_error", False)),
    }


def parse_final_smiles(text: str) -> str | None:
    """The SMILES after the last ``FINAL_SMILES:`` marker, or None if absent.

    The last marker wins so a stray earlier mention does not shadow the answer.

    Args:
        text: The agent's final reply text.

    Returns:
        The SMILES string, or None when the contract was not met.
    """
    matches = _FINAL_SMILES.findall(text or "")
    return matches[-1] if matches else None
