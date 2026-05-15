#!/usr/bin/env python3
"""
pre_bash_tmux.py: PreToolUse hook for Bash tool.

Writes the command to a temp file, then rewrites the Bash tool's
command to call tmux_bash_exec.py which sends it line by line
via libtmux.

Input (stdin): JSON with tool_input.command
Output (stdout): JSON with updatedInput
"""

import json
import os
import sys
import tempfile

HOOK_DIR = os.path.dirname(os.path.abspath(__file__))
EXEC_SCRIPT = os.path.join(HOOK_DIR, "tmux_bash_exec.py")


def main():
    raw = sys.stdin.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        sys.exit(0)

    command = (data.get("tool_input") or {}).get("command", "")
    if not command.strip():
        sys.exit(0)

    # Write command to temp file (avoids all escaping issues)
    fd, tmpfile = tempfile.mkstemp(prefix="claude_bash_cmd_", suffix=".sh", dir="/tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(command)

    # Rewrite the Bash tool to call our Python executor
    wrapped = f"python3 {EXEC_SCRIPT} {tmpfile}"

    output = {
        "hookSpecificOutput": {
            "updatedInput": {"command": wrapped}
        }
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
