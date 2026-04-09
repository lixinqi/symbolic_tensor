# Code Auto-Encoder User Guide

This document explains how to use `test_baseline.py` to run baseline tests for the code auto-encoder.

## Quick Start

```bash
# Non-interactive mode (recommended for batch processing)
python experience/example/code_auto_encoder/test_baseline.py \
    --llm-method tmux_cc

# Interactive mode (visual observation of tmux_cc execution)
python experience/example/code_auto_encoder/test_baseline.py \
    --llm-method tmux_cc \
    --interactive \
    --tmux-session manual_tmux_cc
```

## Architecture Overview

### Design Philosophy

The `tmux_cc` method uses a **file-based input/output approach** instead of passing data through command-line arguments or tmux send-keys. This design solves several problems:

1. **Long prompt handling**: Avoids command-line length limits and tmux buffer issues
2. **Special character escaping**: No need to escape quotes, newlines, or other special characters
3. **Debugging support**: Input/output files are preserved for inspection
4. **Task isolation**: Each task runs in its own workspace directory

### Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                        tmux_cc Workflow                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Create workspace directory                                  │
│     ~/.tmux_cc_tmp/task_YYYYMMDD_HHMMSS_idx/                   │
│                                                                 │
│  2. Write input files                                           │
│     ├── input/prompt.txt          (task description)           │
│     ├── input/packed_workspace.txt (codebase content)          │
│     └── input/task_info.txt       (metadata)                   │
│                                                                 │
│  3. Start ducc with short prompt                                │
│     "Read from ./input/, write to ./output/result.txt"         │
│                                                                 │
│  4. ducc reads files and processes task                         │
│     └── Writes result to output/result.txt                     │
│                                                                 │
│  5. Copy output back to original target file                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `TMUX_CC_WORKSPACE_ROOT` | `~/.tmux_cc_tmp/` | Root directory for task workspaces |
| `TMUX_CC_BIN` | (auto-detect) | Path to ducc binary |

**Binary auto-detection order:**
1. `TMUX_CC_BIN` environment variable
2. `ducc` in PATH
3. `~/.comate/extensions/baidu.baidu-cc-*/resources/native-binary/bin/ducc`

### Workspace Structure

Each task creates an isolated workspace:

```
~/.tmux_cc_tmp/
└── task_20260408_143025_0_0/
    ├── input/
    │   ├── prompt.txt           # Task description/prompt
    │   ├── packed_workspace.txt # Packed codebase (repomix format)
    │   └── task_info.txt        # Metadata (original paths, timestamp)
    └── output/
        └── result.txt           # Output from ducc (created by agent)
```

## Basic Usage

```bash
python experience/example/code_auto_encoder/test_baseline.py [OPTIONS]
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--total-batch-size` | int | 16 | Batch size |
| `--num-iterations` | int | 1 | Number of iterations |
| `--llm-method` | str | raw_llm_api | LLM method: `raw_llm_api`, `coding_agent`, `tmux_cc` |
| `--workspace-dir` | str | None | Working directory (uses temp directory by default) |
| `--interactive` | flag | False | Enable tmux interactive mode (only for tmux_cc) |
| `--no-auto-confirm` | flag | False | Disable auto-confirm (manual operation in interactive mode) |
| `--tmux-session` | str | None | Custom tmux session name |

## LLM Methods

### 1. raw_llm_api (default)

Direct LLM API calls, suitable for quick testing.

```bash
python experience/example/code_auto_encoder/test_baseline.py \
    --llm-method raw_llm_api
```

### 2. coding_agent

Uses coding agent mode.

```bash
python experience/example/code_auto_encoder/test_baseline.py \
    --llm-method coding_agent
```

### 3. tmux_cc

Uses tmux_cc CLI tool, supports two running modes:

#### Non-interactive Mode (default)

Runs tmux_cc via subprocess, no visual interface.

```bash
python experience/example/code_auto_encoder/test_baseline.py \
    --llm-method tmux_cc
```

#### Interactive Mode (tmux visualization)

Runs tmux_cc in a tmux session for real-time agent observation.

```bash
python experience/example/code_auto_encoder/test_baseline.py \
    --llm-method tmux_cc \
    --interactive
```

## Interactive Mode Details

### Session Reuse Optimization

**New in latest version:** Interactive mode now reuses a single ducc session across all tasks in a batch, significantly reducing startup overhead.

**Before (old behavior):**
```
task1 → [start ducc] → execute → [stop] → task2 → [start ducc] → execute → [stop] ...
```

**After (new behavior):**
```
[start ducc once] → task1 → task2 → task3 ... → [session preserved]
```

**Benefits:**
- Eliminates ducc startup time for each task (typically 3-5 seconds per task)
- Single tmux session to monitor (easier to observe)
- Faster overall execution for multi-task batches

### File-Based Input Approach

tmux_cc uses a file-based approach to pass input data to the agent. Instead of sending long prompts through tmux, it:

1. **Creates a workspace directory** under `~/.tmux_cc_tmp/`
2. **Writes input files** containing the prompt and packed codebase
3. **Sends a short instruction** telling ducc to read from the files
4. **Reads output** from `output/result.txt` after completion

This approach is more reliable than tmux send-keys for:
- Long prompts (thousands of characters)
- Content with special characters (quotes, newlines, backticks)
- Multi-file codebases packed into a single prompt

### Workspace Configuration

**Environment Variables:**

```bash
# Set custom workspace root (default: ~/.tmux_cc_tmp/)
export TMUX_CC_WORKSPACE_ROOT=/path/to/custom/workspace

# Set custom ducc binary path (optional, auto-detected by default)
export TMUX_CC_BIN=/path/to/ducc
```

**Workspace Lifecycle:**
- Created: When a task starts
- Preserved: After task completes (for debugging)
- Cleanup: Manual (see FAQ section)

### How It Works

Interactive mode will:
1. Create a workspace directory under `~/.tmux_cc_tmp/`
2. Write input data (prompt, packed codebase) to `input/` directory
3. Start tmux_cc with a short prompt that points to the input files
4. tmux_cc reads from files and writes output to `output/result.txt`
5. Copy output back to the original target file
6. **Keep tmux session alive** for user observation

### Enable Interactive Mode

Add the `--interactive` flag to enable tmux interactive mode:

```bash
python experience/example/code_auto_encoder/test_baseline.py \
    --num-iterations 1 \
    --total-batch-size 2 \
    --llm-method tmux_cc \
    --interactive
```

### Observe tmux_cc Execution

After starting the program, in **another terminal**:

```bash
# List running tmux sessions
tmux ls

# Attach to tmux_cc session for real-time output
tmux attach -t tmux_cc_interactive_0_0

# Detach from session (won't close it)
# Press Ctrl+B then D
```

### Session Naming Convention

In interactive mode, a single shared session is used for all tasks:
- Default: `tmux_cc_interactive_batch`
- Custom name: use `--tmux-session my_name` argument

The session processes all tasks sequentially and is preserved after completion.

### Cleanup tmux Sessions

Sessions are preserved in interactive mode for observation, manual cleanup required:

```bash
# Kill the batch session
tmux kill-session -t tmux_cc_interactive_batch

# Kill custom named session
tmux kill-session -t my_custom_session

# Kill all tmux_cc sessions
tmux ls | grep tmux_cc | cut -d: -f1 | xargs -I{} tmux kill-session -t {}

# Kill all tmux sessions (use with caution)
tmux kill-server
```

### Auto-confirm Feature

Interactive mode enables auto-confirm by default, automatically handling these prompts:

| Prompt | Auto Action |
|--------|-------------|
| "Do you want to proceed" | Press Enter |
| "Yes, I trust this folder" | Press Enter |
| "allow all edits during this session" | Press Down + Enter |
| "Press Enter to continue" | Press Enter |
| "Yes, I accept" | Press Down + Enter |
| "No, exit" | Press Down + Enter |

### Disable Auto-confirm

To manually control each tmux_cc confirmation step:

```bash
python experience/example/code_auto_encoder/test_baseline.py \
    --llm-method tmux_cc \
    --interactive \
    --no-auto-confirm
```

Then attach to tmux in another terminal for manual operation.

### Custom tmux Session Name

```bash
python experience/example/code_auto_encoder/test_baseline.py \
    --llm-method tmux_cc \
    --interactive \
    --tmux-session my_tmux_cc_session
```

Then use `tmux attach -t my_tmux_cc_session` to connect.

## Complete Examples

### Example 1: Quick Test (small batch)

```bash
python experience/example/code_auto_encoder/test_baseline.py \
    --total-batch-size 2 \
    --num-iterations 1 \
    --llm-method raw_llm_api
```

### Example 2: tmux_cc Interactive Mode Observation

```bash
# Terminal 1: Start test
python experience/example/code_auto_encoder/test_baseline.py \
    --total-batch-size 2 \
    --llm-method tmux_cc \
    --interactive

# Terminal 2: Observe tmux_cc execution (single session for all tasks)
tmux attach -t tmux_cc_interactive_batch
```

### Example 3: Multiple Iterations + Custom Workspace

```bash
python experience/example/code_auto_encoder/test_baseline.py \
    --total-batch-size 8 \
    --num-iterations 3 \
    --llm-method tmux_cc \
    --workspace-dir /tmp/my_workspace
```

### Example 4: Manual tmux_cc Control (auto-confirm disabled)

```bash
# Terminal 1: Start test
python experience/example/code_auto_encoder/test_baseline.py \
    --llm-method tmux_cc \
    --interactive \
    --no-auto-confirm \
    --tmux-session manual_tmux_cc

# Terminal 2: Manual operation
tmux attach -t manual_tmux_cc
# Manually handle all confirmation prompts in tmux
```

## Python API Usage

```python
from experience.example.code_auto_encoder.test_baseline import test_baseline

# Non-interactive mode
test_baseline(
    total_batch_size=16,
    num_iterations=1,
    llm_method="tmux_cc",
    interactive=False,
)

# Interactive mode
test_baseline(
    total_batch_size=16,
    num_iterations=1,
    llm_method="tmux_cc",
    interactive=True,
    auto_confirm=True,
    tmux_session="my_session",
)
```

## FAQ

### Q: tmux command not found

Install tmux:

```bash
# macOS
brew install tmux

# Ubuntu/Debian
sudo apt install tmux
```

### Q: tmux_cc never finishes in interactive mode

The program detects tmux_cc completion by monitoring screen content changes. If tmux_cc gets stuck:

1. Attach to tmux session to check status: `tmux attach -t <session>`
2. Manually complete the task
3. Or press `Ctrl+C` to terminate the program

### Q: How to list all tmux sessions

```bash
tmux ls
```

### Q: How to cleanup tmux_cc sessions

```bash
# Kill the batch session
tmux kill-session -t tmux_cc_interactive_batch

# Kill all tmux_cc sessions
tmux ls | grep tmux_cc | cut -d: -f1 | xargs -I{} tmux kill-session -t {}
```

### Q: How to cleanup workspace directories

```bash
# View workspace directories
ls -la ~/.tmux_cc_tmp/

# Remove all workspace directories
rm -rf ~/.tmux_cc_tmp/*

# Or set a custom workspace root
export TMUX_CC_WORKSPACE_ROOT=/tmp/my_tmux_cc_workspace
```

### Q: How to debug tmux_cc issues

Check the workspace directory for input/output files:

```bash
# Find recent workspace
ls -lt ~/.tmux_cc_tmp/ | head -5

# Check input files
cat ~/.tmux_cc_tmp/task_*/input/prompt.txt
cat ~/.tmux_cc_tmp/task_*/input/task_info.txt

# Check output
cat ~/.tmux_cc_tmp/task_*/output/result.txt
```
