"""Tests for the MCP client wrapper's error handling.

The app returns a JSON ``{"error": ...}`` body with a 4xx status. ``urlopen``
raises ``HTTPError`` on that status, so the wrapper must read the body to recover
the real message instead of leaking a bare "HTTP Error 400".
"""

import io
import json
import urllib.error

from chemistree.app.mcp import format_result, read_error


def _http_error(status: int, body: bytes) -> urllib.error.HTTPError:
    """Build an HTTPError with a readable body, as urlopen raises on 4xx/5xx."""
    return urllib.error.HTTPError(
        "http://x/command", status, "error", {}, io.BytesIO(body)  # type: ignore[arg-type]
    )


def test_read_error_extracts_the_app_error_message():
    body = json.dumps(
        {"error": "group must have at least one dummy attachment (*)"}
    ).encode()
    err = _http_error(400, body)
    assert read_error(err) == "group must have at least one dummy attachment (*)"


def test_read_error_falls_back_to_raw_body_when_not_json():
    err = _http_error(500, b"boom")
    assert read_error(err) == "boom"


_DATA = {
    "message": "swapped node 0",
    "state": {"smiles": "Fc1ccccc1", "describe": "- [0] fluoro on [1] benzene"},
}


def test_format_result_is_lean_without_state():
    # Explore mode: just the outcome and the new SMILES, no fragment listing.
    result = format_result(_DATA, with_state=False)
    assert result == "swapped node 0 | SMILES: Fc1ccccc1"


def test_format_result_appends_the_listing_with_state():
    # Primed mode: the refreshed listing rides along so the agent skips a lookup.
    result = format_result(_DATA, with_state=True)
    assert result.startswith("swapped node 0 | SMILES: Fc1ccccc1")
    assert "- [0] fluoro on [1] benzene" in result
