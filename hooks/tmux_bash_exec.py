#!/usr/bin/env python3
"""
tmux_bash_exec.py: Execute a bash command inside a tmux session via libtmux.

Sends the command line by line to a tmux pane, waits for completion,
captures the pane output, and prints it to stdout.

Writes screen captures directly to a (num_capture,) symbolic tensor
for Stage 0 cold-start data harvesting.

Usage: python3 tmux_bash_exec.py <command_file>
"""

import os
import sys
import time

import libtmux

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage0_tensor import append_capture


SESSION_NAME = "claude_bash_stage0"
POLL_INTERVAL = 0.3
MAX_WAIT = 120  # seconds


def ensure_session(server: libtmux.Server) -> libtmux.Pane:
    """Ensure tmux session exists, return the active pane."""
    for s in server.sessions:
        if s.session_name == SESSION_NAME:
            return s.active_window.active_pane

    session = server.new_session(
        session_name=SESSION_NAME,
        x=200, y=50,
    )
    time.sleep(0.5)  # wait for shell init
    return session.active_window.active_pane


def capture_pane(pane: libtmux.Pane) -> str:
    """Capture pane content as a single string."""
    lines = pane.capture_pane()
    return "\n".join(lines)


def has_idle_prompt(pane: libtmux.Pane) -> bool:
    """Check if the pane shows an idle prompt (no command after prompt marker)."""
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


def wait_for_idle(pane: libtmux.Pane, timeout: float = MAX_WAIT) -> bool:
    """Wait until the pane shows an idle prompt."""
    start = time.time()
    while time.time() - start < timeout:
        if has_idle_prompt(pane):
            return True
        time.sleep(POLL_INTERVAL)
    return False


def main():
    if len(sys.argv) < 2:
        print("Usage: tmux_bash_exec.py <command_file>", file=sys.stderr)
        sys.exit(1)

    cmd_file = sys.argv[1]
    with open(cmd_file, "r", encoding="utf-8") as f:
        command = f.read()

    if not command.strip():
        sys.exit(0)

    server = libtmux.Server()
    pane = ensure_session(server)

    # Wait for idle before starting
    if not wait_for_idle(pane, timeout=5):
        pane.send_keys("C-c", enter=False, suppress_history=False)
        time.sleep(0.5)

    # Capture screen BEFORE — write to tensor
    pre_capture = capture_pane(pane)
    append_capture(pre_capture)

    # Send command line by line via libtmux
    lines = command.split("\n")
    for line in lines:
        pane.send_keys(line, enter=True, suppress_history=False)

    # Wait for command to complete
    time.sleep(0.3)
    completed = wait_for_idle(pane, timeout=MAX_WAIT)

    # Capture screen AFTER — write to tensor
    post_capture = capture_pane(pane)
    append_capture(post_capture)

    # Print to stdout for Claude Code
    print(post_capture)

    if not completed:
        print("\n[tmux_bash_exec: command may not have completed within timeout]",
              file=sys.stderr)
        sys.exit(1)

    # Clean up temp file
    try:
        os.unlink(cmd_file)
    except OSError:
        pass


if __name__ == "__main__":
    main()
