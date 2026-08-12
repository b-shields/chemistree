"""Tests for the headless Claude Code chat driver.

These cover the pure logic only: how a user turn becomes a command line, and how
a raw stream-json message becomes a UI event. Driving the real ``claude`` binary
is integration and is not unit tested.
"""

from chemistree.app.chat import build_command, to_events, tool_phrase


def test_build_command_first_turn_has_no_resume():
    cmd = build_command("swap the chloro for fluoro", session_id=None)
    assert "-p" in cmd
    assert "swap the chloro for fluoro" in cmd
    assert cmd[cmd.index("--model") + 1] == "haiku"
    assert cmd[cmd.index("--output-format") + 1] == "stream-json"
    assert "--resume" not in cmd


def test_build_command_resumes_a_known_session():
    cmd = build_command("now undo that", session_id="abc-123")
    assert cmd[cmd.index("--resume") + 1] == "abc-123"


def test_build_command_allows_only_mcp_tools():
    cmd = build_command("swap it", session_id=None)
    assert cmd[cmd.index("--allowedTools") + 1] == "mcp__chemistree"
    blocked = cmd[cmd.index("--disallowedTools") + 1]
    # The file and shell tools must be off limits during a demo.
    for tool in ("Bash", "Edit", "Write", "Read"):
        assert tool in blocked.split(",")


def test_init_message_yields_the_session_id():
    events = to_events({"type": "system", "subtype": "init", "session_id": "s-9"})
    assert events == [{"kind": "session", "session_id": "s-9"}]


def test_assistant_text_becomes_a_text_event():
    message = {
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": "Done — Cl is now F."}]},
    }
    assert to_events(message) == [{"kind": "text", "text": "Done — Cl is now F."}]


def test_assistant_tool_use_becomes_a_domain_phrase():
    message = {
        "type": "assistant",
        "message": {"content": [{"type": "tool_use", "name": "mcp__chemistree__swap"}]},
    }
    assert to_events(message) == [
        {"kind": "tool", "name": "swap", "phrase": "swapping the fragment"}
    ]


def test_internal_tools_are_hidden_from_the_feed():
    # Only molecule edits narrate; the agent's own tool discovery stays quiet.
    message = {
        "type": "assistant",
        "message": {"content": [{"type": "tool_use", "name": "ToolSearch"}]},
    }
    assert to_events(message) == []


def test_result_error_becomes_an_error_event():
    events = to_events(
        {"type": "result", "subtype": "error_during_execution", "result": "boom"}
    )
    assert events == [{"kind": "error", "message": "boom"}]


def test_successful_result_is_silent():
    # The final text already arrived as an assistant event; the turn ends via the
    # driver's own "done" signal, so the result line adds nothing to show.
    assert to_events({"type": "result", "subtype": "success", "result": "ok"}) == []


def test_tool_phrase_falls_back_to_the_bare_name():
    assert tool_phrase("swap") == "swapping the fragment"
    assert tool_phrase("teleport") == "teleport"
