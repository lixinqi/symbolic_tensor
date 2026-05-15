#!/usr/bin/env python3
"""
test_stage0_hook.py: End-to-end test for Stage 0 tmux capture hook.

Verifies screen captures are written directly to a (num_capture,)
symbolic tensor on disk.
"""

import json
import os
import shutil
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
HOOK_DIR = os.path.join(REPO_ROOT, "hooks")
PRE_HOOK = os.path.join(HOOK_DIR, "pre_bash_tmux.py")
POST_HOOK = os.path.join(HOOK_DIR, "post_tool_log.py")

# Import stage0_tensor utilities
sys.path.insert(0, HOOK_DIR)
from stage0_tensor import TENSOR_DIR, TENSOR_UID, num_captures, read_capture, _tensor_root


def run_hook_chain(command: str) -> str:
    """Simulate Claude Code: PreToolUse hook → Bash execution."""
    hook_input = json.dumps({"tool_input": {"command": command}})
    result = subprocess.run(
        ["python3", PRE_HOOK],
        input=hook_input, capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        print(f"  Hook failed: {result.stderr}", file=sys.stderr)
        return ""

    hook_output = json.loads(result.stdout)
    rewritten_cmd = hook_output["hookSpecificOutput"]["updatedInput"]["command"]

    result = subprocess.run(
        ["bash", "-c", rewritten_cmd],
        capture_output=True, text=True, timeout=120,
    )
    return result.stdout


def run_post_hook(tool_name: str, tool_input: dict, tool_result: str = ""):
    """Simulate Claude Code PostToolUse."""
    hook_input = json.dumps({
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_result": tool_result,
    })
    subprocess.run(
        ["python3", POST_HOOK],
        input=hook_input, capture_output=True, text=True, timeout=10,
    )


print("Running Stage 0 hook tests...\n")

passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  \u2713 {name}")
    else:
        failed += 1
        print(f"  \u2717 {name}")


# Clean tensor for fresh test
tensor_root = _tensor_root()
if os.path.isdir(tensor_root):
    shutil.rmtree(tensor_root)

# Test 1: Bash command → 2 captures (pre + post)
print("Test 1: Bash command captures screen to tensor")
before = num_captures()
output = run_hook_chain("echo stage0_tensor_test")
after = num_captures()
check("2 new captures (pre + post)", after == before + 2)
check("output contains result", "stage0_tensor_test" in output)
check("post capture has result", "stage0_tensor_test" in read_capture(after - 1))

# Test 2: Read tool → 2 captures (pre + post, real tmux)
print("\nTest 2: Read tool sent as cat to tmux")
# Create a file to read
subprocess.run(["bash", "-c", "echo 'print(42)' > /tmp/stage0_test_read.py"], timeout=5)
before = num_captures()
run_post_hook("Read", {"file_path": "/tmp/stage0_test_read.py"}, "print(42)")
after = num_captures()
check("2 new captures (pre + post)", after == before + 2)
content = read_capture(after - 1)
check("post capture has file content", "print(42)" in content)

# Test 3: Write tool → 2 captures (pre + post, real tmux)
print("\nTest 3: Write tool sent as cat > file to tmux")
before = num_captures()
run_post_hook("Write", {"file_path": "/tmp/stage0_test_write.py", "content": "x = 99"})
after = num_captures()
check("2 new captures (pre + post)", after == before + 2)
content = read_capture(after - 1)
check("post capture has written content", "x = 99" in content)

# Test 4: Edit tool → 2 captures (pre + post, real tmux)
print("\nTest 4: Edit tool sent as python replace to tmux")
subprocess.run(["bash", "-c", "echo 'val = 99' > /tmp/stage0_test_edit.py"], timeout=5)
before = num_captures()
run_post_hook("Edit", {"file_path": "/tmp/stage0_test_edit.py", "old_string": "99", "new_string": "42"})
after = num_captures()
check("2 new captures (pre + post)", after == before + 2)
content = read_capture(after - 1)
check("post capture has new value", "42" in content)

# Test 5: Tensor on disk has correct shape (2+2+2+2 = 8 captures)
print("\nTest 5: Tensor shape on disk")
shape_path = os.path.join(tensor_root, "shape")
check("shape file exists", os.path.isfile(shape_path))
with open(shape_path) as f:
    shape = json.loads(f.read())
total = num_captures()
check(f"shape is [{total}]", shape == [total])

# Test 6: All storage files exist
print("\nTest 6: All storage files readable")
all_readable = all(read_capture(i) != "" for i in range(total))
check(f"all {total} captures non-empty", all_readable)

print(f"\nPassed: {passed}, Failed: {failed}")
if failed > 0:
    sys.exit(1)
print("\nAll Stage 0 hook tests passed.")
