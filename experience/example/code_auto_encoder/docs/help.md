# Code Auto-Encoder User Guide

This document explains how to use `test_baseline.py` to run baseline tests for the code auto-encoder.

## Quick Start

The system supports three `--llm-method` options. Below are complete example commands for each method:

### 1. raw_llm_api (Direct API Call)

The simplest approach, directly calling the LLM API:

```bash
python experience/example/code_auto_encoder/test_baseline.py \
    --llm-method raw_llm_api \
    --num-iterations 1 \
    --total-batch-size 1 \
    --experiment-dir ./my_experiment \
    --seed 42
```

### 2. coding_agent (Coding Agent Mode)

Uses the coding agent framework:

```bash
python experience/example/code_auto_encoder/test_baseline.py \
    --llm-method coding_agent \
    --num-iterations 1 \
    --total-batch-size 1 \
    --experiment-dir ./my_experiment \
    --seed 42
```

### 3. tmux_cc (Tmux Interactive Mode)

Runs ducc via tmux with visual observation support:

```bash
python experience/example/code_auto_encoder/test_baseline.py \
    --llm-method tmux_cc \
    --interactive \
    --tmux-session manual_ducc \
    --experiment-dir ./my_experiment \
    --seed 42
```

Observe execution in another terminal:
```bash
tmux attach -t manual_ducc
```

### Method Comparison

| Method | Features | Use Cases |
|--------|----------|-----------|
| `raw_llm_api` | Direct API call, fastest | Quick testing, batch evaluation |
| `coding_agent` | With agent framework | Requires file operation capabilities |
| `tmux_cc` | Visual interaction | Debugging, observing agent behavior |

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
| `--interactive` | flag | False | Enable tmux interactive mode (only for tmux_cc) |
| `--no-auto-confirm` | flag | False | Disable auto-confirm (manual operation in interactive mode) |
| `--tmux-session` | str | None | Custom tmux session name |
| `--seed` | int | None | Random seed for dataset generation reproducibility |
| `--experiment-dir` | str | None | Experiment directory for organized results |

## Experiment Directory Structure

When using `--experiment-dir`, results are organized into a structured directory:

```
my_experiment/
├── dataset/                      # Input data cache (auto-managed)
│   ├── seed_42/                  # With seed
│   │   ├── metadata.json
│   │   ├── source_files/
│   │   └── samples/
│   └── no_seed_20260410_143000/  # Without seed (timestamp-based)
│       └── ...
├── runs/
│   ├── raw_llm_api/              # Grouped by llm_method
│   │   └── run_20260410_113000/
│   ├── coding_agent/
│   │   └── run_20260410_120000/
│   └── tmux_cc/
│       └── run_20260410_130000/
│           ├── config.json       # Run parameters
│           ├── results.json      # Core index file
│           ├── output/           # Model outputs
│           │   ├── 0/
│           │   │   ├── prediction.txt    # Model prediction
│           │   │   └── ground_truth.txt  # Expected output
│           │   └── ...
│           └── logs/
│               └── run.log       # Run log
└── latest -> runs/tmux_cc/run_...  # Symlink to latest run
```

### Results JSON Format

The `results.json` file contains:

```json
{
  "run_id": "run_20260410_130000",
  "timestamp": "20260410_130000",
  "llm_method": "coding_agent",
  "config": {
    "llm_method": "coding_agent",
    "total_batch_size": 16,
    "num_iterations": 1,
    "seed": 42,
    "interactive": false
  },
  "status": "completed",
  "dataset": {
    "cache_dir": "../../dataset",
    "seed": 42
  },
  "summary": {
    "batch_size": 16,
    "completed": 16,
    "failed": 0,
    "mean_loss": 0.2156
  },
  "tasks": [
    {
      "index": 0,
      "file_info": "path/to/file.py:10-15",
      "input": {
        "file": "path/to/file.py",
        "lines": "10-15",
        "dataset_ref": "../../dataset/seed_42"
      },
      "ground_truth": {
        "path": "output/0/ground_truth.txt",
        "preview": "def example():\n    return 42"
      },
      "output": {
        "path": "output/0/prediction.txt",
        "preview": "def example():\n    return 42",
        "loss": 0.0
      },
      "status": "completed",
      "completed_at": "2026-04-10T13:00:05.123456"
    }
  ]
}
```

## LLM Methods

### 1. raw_llm_api (default)

Direct LLM API calls, suitable for quick testing. All files are merged and sent to the LLM in a single prompt.

```bash
python experience/example/code_auto_encoder/test_baseline.py \
    --llm-method raw_llm_api \
    --experiment-dir ./my_experiment \
    --seed 42
```

### 2. coding_agent

Uses coding agent mode. The LLM receives the files separately and can read/write files.

```bash
python experience/example/code_auto_encoder/test_baseline.py \
    --llm-method coding_agent \
    --experiment-dir ./my_experiment \
    --seed 42
```

### 3. tmux_cc

Uses tmux_cc CLI tool (ducc), supports two running modes:

#### Non-interactive Mode (default)

Runs tmux_cc via subprocess, no visual interface.

```bash
python experience/example/code_auto_encoder/test_baseline.py \
    --llm-method tmux_cc \
    --experiment-dir ./my_experiment \
    --seed 42
```

#### Interactive Mode (tmux visualization)

Runs tmux_cc in a tmux session for real-time agent observation.

```bash
python experience/example/code_auto_encoder/test_baseline.py \
    --llm-method tmux_cc \
    --interactive \
    --experiment-dir ./my_experiment \
    --seed 42
```

## Interactive Mode Details

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
    --interactive \
    --experiment-dir ./my_experiment
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

Each task creates a separate tmux session:
- `tmux_cc_interactive_0_0` - 1st batch, 1st task
- `tmux_cc_interactive_1_0` - 2nd batch, 1st task
- Custom name: use `--tmux-session my_name` argument

### Cleanup tmux Sessions

Sessions are preserved in interactive mode for observation, manual cleanup required:

```bash
# Kill single session
tmux kill-session -t tmux_cc_interactive_0_0

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
    --no-auto-confirm \
    --experiment-dir ./my_experiment
```

Then attach to tmux in another terminal for manual operation.

### Custom tmux Session Name

```bash
python experience/example/code_auto_encoder/test_baseline.py \
    --llm-method tmux_cc \
    --interactive \
    --tmux-session my_ducc_session \
    --experiment-dir ./my_experiment
```

Then use `tmux attach -t my_ducc_session` to connect.

## Complete Examples

### Example 1: Quick Test (small batch)

```bash
python experience/example/code_auto_encoder/test_baseline.py \
    --total-batch-size 2 \
    --num-iterations 1 \
    --llm-method raw_llm_api \
    --experiment-dir ./quick_test
```

### Example 2: Model Comparison with Fixed Dataset

Use `--experiment-dir` and `--seed` to ensure different models use the same input data:

```bash
# Model A: Generate and cache dataset
python experience/example/code_auto_encoder/test_baseline.py \
    --total-batch-size 16 \
    --llm-method raw_llm_api \
    --experiment-dir ./comparison_exp \
    --seed 42

# Model B: Use the same cached dataset
python experience/example/code_auto_encoder/test_baseline.py \
    --total-batch-size 16 \
    --llm-method tmux_cc \
    --experiment-dir ./comparison_exp \
    --seed 42

# Model C: Use the same cached dataset
python experience/example/code_auto_encoder/test_baseline.py \
    --total-batch-size 16 \
    --llm-method coding_agent \
    --experiment-dir ./comparison_exp \
    --seed 42

# Results organized by method:
# comparison_exp/runs/raw_llm_api/run_.../
# comparison_exp/runs/tmux_cc/run_.../
# comparison_exp/runs/coding_agent/run_.../
```

### Example 3: Incremental Dataset Expansion

```bash
# Start with small batch for quick validation
python experience/example/code_auto_encoder/test_baseline.py \
    --total-batch-size 5 \
    --llm-method raw_llm_api \
    --experiment-dir ./my_experiment \
    --seed 42

# Expand to larger batch (reuses first 5 samples)
python experience/example/code_auto_encoder/test_baseline.py \
    --total-batch-size 50 \
    --llm-method raw_llm_api \
    --experiment-dir ./my_experiment \
    --seed 42
# Output: Cache has 5 samples, need 50, will generate more
```

### Example 4: tmux_cc Interactive Mode Observation

```bash
# Terminal 1: Start test
python experience/example/code_auto_encoder/test_baseline.py \
    --total-batch-size 2 \
    --llm-method tmux_cc \
    --interactive \
    --experiment-dir ./debug_run

# Terminal 2: Observe tmux_cc execution
tmux attach -t tmux_cc_interactive_0_0
```

### Example 5: Manual tmux_cc Control (auto-confirm disabled)

```bash
# Terminal 1: Start test
python experience/example/code_auto_encoder/test_baseline.py \
    --llm-method tmux_cc \
    --interactive \
    --no-auto-confirm \
    --tmux-session manual_ducc \
    --experiment-dir ./manual_test

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
    llm_method="raw_llm_api",
    seed=42,
    experiment_dir="./my_experiment",
)

# Interactive mode
test_baseline(
    total_batch_size=16,
    num_iterations=1,
    llm_method="tmux_cc",
    interactive=True,
    auto_confirm=True,
    tmux_session="my_session",
    seed=42,
    experiment_dir="./my_experiment",
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
# Kill single session
tmux kill-session -t tmux_cc_interactive_0_0

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

### Q: How to view experiment results

```bash
# View latest run results
cat my_experiment/latest/results.json | python -m json.tool

# Compare prediction vs ground truth
diff my_experiment/latest/output/0/prediction.txt \
     my_experiment/latest/output/0/ground_truth.txt

# View all task losses
cat my_experiment/latest/results.json | jq '.tasks[].output.loss'
```

### Q: How to ensure reproducible results

Use the `--seed` parameter:

```bash
# Same seed = same dataset samples
python test_baseline.py --seed 42 --experiment-dir exp1
python test_baseline.py --seed 42 --experiment-dir exp2

# Dataset will be cached in experiment_dir/dataset/seed_42/
```
