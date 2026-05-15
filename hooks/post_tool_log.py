#!/usr/bin/env python3
"""
post_tool_log.py: PostToolUse hook for Read, Write, and Edit tools.

Converts file operations to real bash commands, sends them to the
tmux session via libtmux, and captures the screen into the
screen_stream symbolic tensor. All entries are real terminal output.

Input (stdin): JSON with tool_name, tool_input, tool_result
Output: none (logging only)
"""

import json
import os
import sys
import tempfile
import time

import libtmux

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage0_tensor import append_capture

SESSION_NAME = "claude_bash_stage0"
POLL_INTERVAL = 0.3
MAX_WAIT = 30


def _get_pane():
    server = libtmux.Server()
    for s in server.sessions:
        if s.session_name == SESSION_NAME:
            return s.active_window.active_pane
    # Create session if not exists
    session = server.new_session(session_name=SESSION_NAME, x=200, y=50)
    time.sleep(0.5)
    return session.active_window.active_pane


def _capture(pane):
    return "\n".join(pane.capture_pane())


def _has_idle_prompt(pane):
    lines = pane.capture_pane()
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return False
    last = non_empty[-1]
    if "λ" in last:
        after = last.split("λ", 1)[1].strip()
        parts = after.split()
        pi = next((i for i, p in enumerate(parts) if p.startswith(("/", "~"))), -1)
        if pi == -1:
            return len(parts) == 0
        return len(parts) == pi + 1
    if "$ " in last:
        return last.split("$ ", 1)[1].strip() == ""
    if last.rstrip().endswith("$"):
        return True
    return False


def _wait_idle(pane, timeout=MAX_WAIT):
    start = time.time()
    while time.time() - start < timeout:
        if _has_idle_prompt(pane):
            return True
        time.sleep(POLL_INTERVAL)
    return False


def _send_and_capture(pane, lines):
    """Send lines to tmux, wait for idle, capture screen."""
    _wait_idle(pane, timeout=5)
    append_capture(_capture(pane))  # pre

    for line in lines:
        pane.send_keys(line, enter=True, suppress_history=False)

    time.sleep(0.3)
    _wait_idle(pane, timeout=MAX_WAIT)
    append_capture(_capture(pane))  # post


def main():
    raw = sys.stdin.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        sys.exit(0)

    tool_name = data.get("tool_name", "")
    tool_input = data.get("tool_input", {})

    if not tool_name:
        sys.exit(0)

    pane = _get_pane()

    if tool_name == "Read":
        file_path = tool_input.get("file_path", "")
        if file_path:
            _send_and_capture(pane, [f"cat {file_path}"])

    elif tool_name == "Write":
        file_path = tool_input.get("file_path", "")
        content = tool_input.get("content", "")
        if file_path and content:
            # Write content to a temp file, then cat heredoc to target
            fd, tmp = tempfile.mkstemp(prefix="stage0_write_", suffix=".txt")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            _send_and_capture(pane, [f"cat {tmp} > {file_path} && cat {file_path}"])
            try:
                os.unlink(tmp)
            except OSError:
                pass

    elif tool_name == "Edit":
        file_path = tool_input.get("file_path", "")
        old_string = tool_input.get("old_string", "")
        new_string = tool_input.get("new_string", "")
        if file_path:
            # Write a Python one-liner to do the replacement, then show diff
            fd, tmp = tempfile.mkstemp(prefix="stage0_edit_", suffix=".py")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(
                    "import sys\n"
                    f"p = {file_path!r}\n"
                    "t = open(p).read()\n"
                    f"t = t.replace({old_string!r}, {new_string!r}, 1)\n"
                    "open(p, 'w').write(t)\n"
                )
            _send_and_capture(pane, [f"python3 {tmp} && head -20 {file_path}"])
            try:
                os.unlink(tmp)
            except OSError:
                pass


if __name__ == "__main__":
    main()
