"""Serve a command (Claude Code by default) as a browser terminal over a pty.

A WebSocket carries keystrokes and resize events to a pseudo-terminal and streams
the pty output back. The child inherits the app's working directory and
environment, so ``claude`` finds the project's ``.mcp.json`` and conda env.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import pty
import shlex
import signal
import struct
import termios

from fastapi import WebSocket, WebSocketDisconnect

# The command the terminal runs; override for testing.
COMMAND = shlex.split(
    os.environ.get("CHEMISTREE_TERMINAL_CMD", "claude --model sonnet")
)


async def terminal_session(socket: WebSocket) -> None:
    """Run ``COMMAND`` in a pty and bridge it to a WebSocket."""
    await socket.accept()
    pid, master_fd = pty.fork()
    if pid == 0:  # child: becomes the terminal program
        os.execvp(COMMAND[0], COMMAND)
        os._exit(1)  # unreachable unless exec fails

    loop = asyncio.get_running_loop()

    def on_output() -> None:
        try:
            data = os.read(master_fd, 65536)
        except OSError:
            data = b""
        if data:
            loop.create_task(socket.send_text(data.decode(errors="ignore")))
        else:
            loop.remove_reader(master_fd)

    loop.add_reader(master_fd, on_output)
    try:
        while True:
            event = json.loads(await socket.receive_text())
            if event["type"] == "input":
                os.write(master_fd, event["data"].encode())
            elif event["type"] == "resize":
                _set_winsize(master_fd, event["rows"], event["cols"])
    except (WebSocketDisconnect, KeyError, ValueError):
        pass
    finally:
        loop.remove_reader(master_fd)
        _terminate(pid, master_fd)


def _set_winsize(fd: int, rows: int, cols: int) -> None:
    """Set the pty window size so full-screen programs render correctly."""
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


def _terminate(pid: int, master_fd: int) -> None:
    """Stop the child process and close the pty."""
    try:
        os.kill(pid, signal.SIGHUP)
    except ProcessLookupError:
        pass
    try:
        os.close(master_fd)
    except OSError:
        pass
