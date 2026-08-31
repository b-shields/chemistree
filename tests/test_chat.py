"""Tests for the headless Claude Code chat driver.

These cover the pure logic only: how a user turn becomes a command line, and how
a raw stream-json message becomes a UI event. Driving the real ``claude`` binary
is integration and is not unit tested.
"""

import json

from chemistree.app.chat import (
    EXPLORE,
    PRIMED,
    build_command,
    interrupt_line,
    to_events,
    tool_phrase,
    user_message_line,
)


def test_build_command_runs_a_persistent_stream_session():
    cmd = build_command(EXPLORE)
    assert "-p" in cmd
    assert cmd[cmd.index("--input-format") + 1] == "stream-json"
    assert cmd[cmd.index("--output-format") + 1] == "stream-json"
    assert cmd[cmd.index("--model") + 1] == "haiku"
    # One process serves every turn, so no per-turn message or resume id.
    assert "--resume" not in cmd


def test_build_command_runs_the_requested_model():
    cmd = build_command(EXPLORE, model="sonnet")
    assert cmd[cmd.index("--model") + 1] == "sonnet"


def test_build_command_narrates_by_default():
    cmd = build_command(EXPLORE)
    prompt = cmd[cmd.index("--append-system-prompt") + 1]
    assert "before each meaningful edit" in prompt


def test_skip_narration_drops_the_rationale_note():
    cmd = build_command(EXPLORE, narrate=False)
    prompt = cmd[cmd.index("--append-system-prompt") + 1]
    assert "before each meaningful edit" not in prompt


def test_build_command_allows_only_mcp_tools():
    cmd = build_command(EXPLORE)
    assert cmd[cmd.index("--allowedTools") + 1] == "mcp__chemistree"
    blocked = cmd[cmd.index("--disallowedTools") + 1]
    # The file and shell tools must be off limits during a demo.
    for tool in ("Bash", "Edit", "Write", "Read"):
        assert tool in blocked.split(",")


def test_build_command_scopes_mcp_to_this_project_only():
    # Strict scoping keeps the user's global MCP servers out of every turn.
    cmd = build_command(EXPLORE)
    assert cmd[cmd.index("--mcp-config") + 1] == ".mcp.json"
    assert "--strict-mcp-config" in cmd


def test_primed_mode_seeds_the_fragment_listing():
    # In primed mode the current listing is embedded in the system prompt so the
    # agent can act on node ids without a find/describe call.
    listing = "- [0] chloro attached to [1] benzene"
    cmd = build_command(PRIMED, context=listing)
    prompt = cmd[cmd.index("--append-system-prompt") + 1]
    assert listing in prompt
    assert "directly" in prompt  # the primed instruction to use ids directly


def test_explore_mode_never_seeds_even_with_context():
    listing = "- [0] chloro attached to [1] benzene"
    cmd = build_command(EXPLORE, context=listing)
    prompt = cmd[cmd.index("--append-system-prompt") + 1]
    assert listing not in prompt


def test_primed_mode_blocks_the_redundant_overview_tool():
    # Priming seeds and refreshes the overview, so the describe tool is disallowed
    # and dropped from the prompt's tool list. describe_group stays (positions are
    # always pulled on demand).
    cmd = build_command(PRIMED)
    blocked = cmd[cmd.index("--disallowedTools") + 1].split(",")
    assert "mcp__chemistree__describe" in blocked
    assert "mcp__chemistree__describe_group" not in blocked
    prompt = cmd[cmd.index("--append-system-prompt") + 1]
    assert "describe_group, smiles, swap" in prompt  # the primed tool list
    assert "describe, describe_group" not in prompt  # the bare overview is absent


def test_explore_mode_keeps_the_overview_tool():
    # The other mode is unchanged: the agent may still read the overview itself.
    cmd = build_command(EXPLORE)
    blocked = cmd[cmd.index("--disallowedTools") + 1].split(",")
    assert "mcp__chemistree__describe" not in blocked
    prompt = cmd[cmd.index("--append-system-prompt") + 1]
    assert "describe, describe_group, smiles" in prompt


def test_user_message_line_encodes_a_turn():
    line = user_message_line("swap the chloro for fluoro")
    assert line.endswith(b"\n")
    payload = json.loads(line)
    assert payload["type"] == "user"
    assert payload["message"]["content"][0]["text"] == "swap the chloro for fluoro"


def test_interrupt_line_encodes_a_control_request():
    line = interrupt_line()
    assert line.endswith(b"\n")
    payload = json.loads(line)
    assert payload["type"] == "control_request"
    assert payload["request"]["subtype"] == "interrupt"
    assert payload["request_id"]  # a non-empty id, so the response can be matched


def test_init_message_yields_nothing():
    # Each turn emits an init line, but with a persistent process there is no
    # session to track, so it produces no UI event.
    assert to_events({"type": "system", "subtype": "init", "session_id": "s-9"}) == []


def test_assistant_text_becomes_a_text_event():
    message = {
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": "Done — Cl is now F."}]},
    }
    assert to_events(message) == [{"kind": "text", "text": "Done — Cl is now F."}]


def test_assistant_thinking_becomes_a_thinking_event():
    message = {
        "type": "assistant",
        "message": {
            "content": [{"type": "thinking", "thinking": "This pocket is lipophilic."}]
        },
    }
    assert to_events(message) == [
        {"kind": "thinking", "text": "This pocket is lipophilic."}
    ]


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
