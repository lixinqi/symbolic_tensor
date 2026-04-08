# Code Auto-Encoder User Guide

This document explains how to use `test_baseline.py` to run baseline tests for the code auto-encoder.

## Quick Start

```bash
# Non-interactive mode (recommended for batch processing)
python experience/example/code_auto_encoder/test_baseline.py \
    --llm-method tmux_cc

# Interactive mode (visual observation of ducc execution)
python experience/example/code_auto_encoder/test_baseline.py \
    --llm-method tmux_cc \
    --interactive \
    --tmux-session manual_ducc
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

Uses ducc CLI tool, supports two running modes:

#### Non-interactive Mode (default)

Runs ducc via subprocess, no visual interface.

```bash
python experience/example/code_auto_encoder/test_baseline.py \
    --llm-method tmux_cc
```

#### Interactive Mode (tmux visualization)

Runs ducc in a tmux session for real-time agent observation.

```bash
python experience/example/code_auto_encoder/test_baseline.py \
    --llm-method tmux_cc \
    --interactive
```

## Interactive Mode Details

### How It Works

Interactive mode will:
1. Write prompt to a temp file (avoids command line length limits)
2. Create a wrapper script to call ducc
3. Execute in tmux session, capturing output to file
4. Detect ducc completion and write output to target file
5. **Keep tmux session alive** for user observation

### Enable Interactive Mode

Add the `--interactive` flag to enable tmux interactive mode:

```bash
python experience/example/code_auto_encoder/test_baseline.py \
    --num-iterations 1 \
    --total-batch-size 2 \
    --llm-method tmux_cc \
    --interactive
```

### Observe ducc Execution

After starting the program, in **another terminal**:

```bash
# List running tmux sessions
tmux ls

# Attach to ducc session for real-time output
tmux attach -t ducc_interactive_0_0

# Detach from session (won't close it)
# Press Ctrl+B then D
```

### Session Naming Convention

Each task creates a separate tmux session:
- `ducc_interactive_0_0` - 1st batch, 1st task
- `ducc_interactive_1_0` - 2nd batch, 1st task
- Custom name: use `--tmux-session my_name` argument

### Cleanup tmux Sessions

Sessions are preserved in interactive mode for observation, manual cleanup required:

```bash
# Kill single session
tmux kill-session -t ducc_interactive_0_0

# Kill all ducc sessions
tmux ls | grep ducc | cut -d: -f1 | xargs -I{} tmux kill-session -t {}

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

To manually control each ducc confirmation step:

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
    --tmux-session my_ducc_session
```

Then use `tmux attach -t my_ducc_session` to connect.

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

# Terminal 2: Observe ducc execution
tmux attach -t ducc_interactive_0_0
```

### Example 3: Multiple Iterations + Custom Workspace

```bash
python experience/example/code_auto_encoder/test_baseline.py \
    --total-batch-size 8 \
    --num-iterations 3 \
    --llm-method tmux_cc \
    --workspace-dir /tmp/my_workspace
```

### Example 4: Manual ducc Control (auto-confirm disabled)

```bash
# Terminal 1: Start test
python experience/example/code_auto_encoder/test_baseline.py \
    --llm-method tmux_cc \
    --interactive \
    --no-auto-confirm \
    --tmux-session manual_ducc

# Terminal 2: Manual operation
tmux attach -t manual_ducc
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

### Q: ducc never finishes in interactive mode

The program detects ducc completion by monitoring screen content changes. If ducc gets stuck:

1. Attach to tmux session to check status: `tmux attach -t <session>`
2. Manually complete the task
3. Or press `Ctrl+C` to terminate the program

### Q: How to list all tmux sessions

```bash
tmux ls
```

### Q: How to cleanup ducc tmux sessions

```bash
# Kill single session
tmux kill-session -t ducc_interactive_0_0

# Kill all ducc sessions
tmux ls | grep ducc | cut -d: -f1 | xargs -I{} tmux kill-session -t {}
```
