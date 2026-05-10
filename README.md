# Experience

A **self-improving harness framework** built as a PyTorch extension. Composable tensor ops + trainable experience + autograd = portable Modules that improve through training.

Each harness Module is a static DAG of lazy tensor ops (`ft_expert`, `ft_switch`, `ft_sequential`, `ft_recurrent`, `ft_tmux_*`). The Module's behavior is determined by its **experience tensors** — QKV stores that serve as learnable weights. The LLM acts as the compute kernel: forward pass reads experience to produce outputs; backward pass computes diffs; optimizer applies patches to experience incrementally.

The result: **portable, self-improving agents** that share the same infrastructure but learn different skills through training. Like Claude Code slash commands, but standardized — typed tensors, autograd, shape contracts, trainable experience.

## Architecture

```
experience/
├── symbolic_tensor/
│   ├── function/          # Autograd Functions: st_moe, st_attention, st_stack, slice_*, merge, fork, copy, loss, etc.
│   ├── tensor_util/       # Symbolic tensor primitives: make, slice, assign, diff, patch, dense/sparse
│   ├── module/            # nn.Module wrappers: StMoeModule, WithDenseView
│   ├── optimizer/         # StSGD: dual-channel (numeric + symbolic patch) optimizer
│   ├── data_loader/       # Batch data loading from files
│   └── test/              # Integration tests
├── future_tensor/
│   ├── future_tensor.py   # FutureTensor factory: lazy async scalar tensor with ft_* attributes
│   ├── status.py          # Status tagged union (confidence, scbf, kContextOverflow, etc.)
│   ├── function/          # Autograd ops: slice, unsqueeze, recurrent, expert, switch, sequential, tmux
│   └── second_derivative/ # 2nd-derivative framework: policies, dispatcher, GradFn wrappers
├── llm_client/            # LLM backends: raw API (OpenAI-compatible) and coding agent (Claude SDK)
├── sparse_util/           # Sparse coordinate operations (transpose, convert)
├── fs_util/               # File system utilities (directory packing, path enumeration, text merger)
├── test/                  # End-to-end tests and benchmarks
└── example/               # Training demos
    ├── naive_symbolic_transform_model/  # Python-to-Viba translation
    └── auto-encoder/                    # Auto-encoder baseline stability tests
```

## Dual-Channel Gradient System

Gradients propagate through two channels simultaneously:

| Channel | What it carries | How it's computed |
|---------|----------------|-------------------|
| **Numeric** (coefficient) | Float values (bfloat16) | Standard autograd / SGD arithmetic |
| **Symbolic** (text) | Unified diffs stored in files | LLM computes `diff -u` between actual and expected |

The `symbolic_grad_registry` (thread-local dictionary) passes symbolic gradient metadata between autograd Function backward calls, since PyTorch autograd strips custom tensor attributes (`st_relative_to`, `st_tensor_uid`) when propagating gradients between Function nodes.

## FutureTensor

A **lazy, async tensor abstraction** built on top of SymbolicTensor. A `FutureTensor` is a scalar `torch.Tensor` (shape `()`, bfloat16, value `1`) monkey-patched with `ft_*` attributes that make it a reference to a pending symbolic computation. Each element is a file on disk, produced on-demand by an async generator (typically an LLM call).

Key properties:
- **Lazy**: elements are not materialized until `ft_forward(prompt_tensor)` is called
- **Async**: all elements materialize concurrently via `asyncio.gather`
- **Status-encoded**: coefficients in `ft_static_tensor` carry `Status` floats (confidence, self_confidence_but_failed, kContextOverflow)

### FutureTensor Operations

| Op | Purpose | 2nd Derivative |
|----|---------|----------------|
| `ft_slice` | Slice with torch semantics; backward scatters grad | `SliceGradFn` |
| `ft_unsqueeze` | Insert dim of size 1 | `UnsqueezeGradFn` |
| `ft_recurrent` | Generate-validate retry loop over last dim | `RecurrentGradFn` |
| `ft_expert` | Expert query + retrieval + LLM translation | `ExpertGradFn` |
| `ft_switch` | Lazy control-flow branch selection | `SwitchGradFn` |
| `ft_sequential` | Lazy sequential evaluation with early-return on error | `SequentialGradFn` |
| `ft_tmux_create_session` | Create tmux terminal session | `TmuxCreateSessionGradFn` |
| `ft_tmux_send_text` | Send text to tmux session | `TmuxSendTextGradFn` |
| `ft_tmux_send_ctrl` | Send Ctrl key to tmux session | `TmuxSendCtrlGradFn` |

## Second Derivative (Meta-Learning)

The framework supports symbolic second derivatives — the LLM reflects on its own reflections:

| Order | Mathematical object | Framework interpretation |
|---|---|---|
| Forward pass | `f(x)` | LLM generates output |
| 1st derivative | `∂L/∂x` | LLM *reflects* on its output |
| 2nd derivative | `∂²L/∂x²` | LLM reflects on the *reflection* |

Implementation uses `need_2nd_derivative(input, anchor)` to create a computational dependency, then `loss.backward(create_graph=True)` builds the 2nd-derivative graph, and `anchor.grad.backward()` traverses it through GradFn wrappers that dispatch to a configurable Policy.

```python
# 2nd-derivative flow
second_derivative_start = torch.ones((), dtype=torch.bfloat16, requires_grad=True)
anchored = need_2nd_derivative(input_ft, second_derivative_start)
output = model(anchored)
loss = ft_mean(output)

loss.backward(create_graph=True)

records = []
with dispatch_policy(TracePolicy(records)):
    second_derivative_start.grad.backward()
# records: list of ReflectionRecord(fn, inputs, output, timestamp)
```

Policies:
- **`TracePolicy`** (default): records all 2nd-derivative dispatches without running any LLM
- **Custom policies**: subclass `Policy` and implement `dispatch(fn, arg_name2inputs)` to execute LLM meta-reflections

## LLM Backends

Two backends are supported:

### `raw_llm_api` (default)
OpenAI-compatible API (`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`). Packs directory contents into a prompt, finds files containing the TODO placeholder, and replaces them with LLM responses. Lightweight, no tool access.

### `coding_agent`
Claude Agent SDK with `Read`, `Edit`, `Write` tool access. The agent can directly read and modify files in the workspace. Best for complex tasks requiring file system interaction.

Both are dispatched through `TaskHandler`, which takes `AgentTask` objects and runs them concurrently via `asyncio.gather`.

## ExperienceTensor

An **ExperienceTensor** is a symbolic tensor of shape `[N, 3]` where each row is a `(query, key, value)` triple:
- **Query** (position 0): Semantic keywords (one per line) used for Jaccard similarity retrieval
- **Key** (position 1): Source domain content (e.g., Python code)
- **Value** (position 2): Target domain content (e.g., code in another language)

It acts as the learnable "weight" of the model — starts empty and is populated during training. The backward pass computes diffs against expected output, and the optimizer applies patches to experience entries via `patch`.

## Autograd Functions

### Core (`function/`)

| Function | Purpose |
|----------|---------|
| **`StMoe`** (`st_moe`) | Mixture-of-Experts: query gen → retrieval → LLM translation → copy-back |
| **`st_attention`** | Composes `slice_attention` + `merge` into a full attention operation |
| **`slice_attention`** | Attention-style forward/backward with causal masks |
| **`st_stack`** | Stack symbolic tensors along a new axis (like `torch.stack`) |
| **`slice_view`** | Autograd-aware slice creating symlinked views (shared storage) |
| **`slice_tensor`** | Autograd-aware slice creating independent copies |
| **`merge`** | Merges symbolic tensor elements along an axis using `TextMerger` |
| **`ForkTensor`** | Replicates input into N identical views; backward merges all grads |
| **`Copy`** | Independent copy with autograd support |
| **`GetEditDistanceRatio`** | Loss function: Levenshtein edit distance ratio |
| **`dense_to_sparse`** / **`sparse_to_dense`** | Sparse/dense symbolic tensor conversion pair |
| **`with_dense_view`** | Temporarily provides a dense view of sparse tensors |

### Sparse/Dense Conversions

`dense_to_sparse` extracts nonzero elements (by `torch.nonzero`) into a 1D sparse tensor with coordinate indexes. `sparse_to_dense` reconstructs a dense tensor from sparse representation. They form a forward/backward pair as `torch.autograd.Function` — each one's backward calls the other.

### StMoe Detail

Forward pass:
1. **Query Generation**: LLM extracts semantic keywords from each input element
2. **Experience Retrieval**: Jaccard similarity (with Gaussian noise for exploration) selects top-k relevant experience entries. Cold-start: returns random indexes when all queries are empty.
3. **Context Assembly**: Dump symlink views of experience and input
4. **Task Dispatch**: `TaskHandler` dispatches `AgentTask` objects to the LLM backend
5. **Copy-back**: Results propagate through symlinks to parent storage

Backward pass computes gradients for both **input** and **experience** through numeric + symbolic channels:

- **Grad Input**: Numeric pass-through. Symbolically, the LLM reads grad_output diffs alongside original input/output/experience and writes improved input files.
- **Grad Experience**: Forward index list transposed to group by experience entry. Cold-start padding randomly samples empty entries so they still receive gradients. Backward runs twice — once for key, once for value — with domain-specific prompts.

### Slicing with Autograd

Both `slice_view` and `slice_tensor` support autograd:

- **`slice_view`**: Creates views via symlinks (shared storage). Backward scatters gradients back to original positions.
- **`slice_tensor`**: Creates independent copies. Backward still tracks which input positions contributed to each output.

Both support standard Python indexing syntax:
```python
# Via tensor attributes
row = t.st_view_slicer[0, :]      # Symlinked view
col = t.st_value_slicer[:, 1]     # Independent copy

# Via function calls
from experience.symbolic_tensor.function.slice_view import slice_view
sub = slice_view(t, [0, slice(None)])
```

## Tensor Utilities (`tensor_util/`)

| Utility | Purpose |
|---------|---------|
| `make_tensor` | Create symbolic tensor from nested lists of strings/Paths |
| `make_none_tensor` | Create zero-filled tensor with st_relative_to and st_tensor_uid |
| `none_tensor_like` | Create None-filled tensor matching input shape |
| `empty_tensor_like` | Create ""-filled tensor matching input shape |
| `todo_tensor_like` | Create TODO-filled tensor matching input shape |
| `slice_view` | Create symbolic tensor view via symlinks (shared storage) |
| `slice_tensor` | Create independent copy (not symlinked) |
| `assign_tensor` | Assign content from one tensor to another (copy) |
| `assign_view` | Assign via symlinks (view) |
| `get_diff_tensor` | Element-wise `diff -u` between two symbolic tensors |
| `patch_tensor` | Apply unified diffs via `patch` CLI (fuzz=3). Cold-start: extracts `+` lines when target empty. |
| `dump_tensor` / `dump_view` | Serialize tensor content to directory |
| `load_tensor` | Deserialize tensor from directory |
| `pack_tensor` | Pack tensor into nested list of file contents |
| `st_patched` | Check if a tensor has been patched |
| `dense_to_sparse` / `sparse_to_dense` | Sparse/dense conversion implementations |
| `register_tensor_ops` | Monkey-patches `torch.Tensor` with symbolic tensor methods |

### Registered Tensor Methods

`register_tensor_ops` adds the following methods to `torch.Tensor`:

| Method | Purpose |
|--------|---------|
| `st_pack()` | Pack tensor into nested list of file contents |
| `st_assign(rvalue)` | Copy assignment |
| `st_assign_view(rvalue)` | Symlink assignment (view) |
| `st_get_diff(rvalue)` | Compute unified diff |
| `st_patch(rvalue)` | Apply patches |
| `st_file_paths()` | List all storage paths |
| `st_fork(n)` | Fork into N views |
| `st_view_slicer[...]` | Pythonic slicing with symlink views |
| `st_value_slicer[...]` | Pythonic slicing with independent copies |

## Optimizer (`StSGD`)

Two-channel update per step:
- **Numeric**: `param.data = (1 - lr) * param.data + lr * grad.data`
- **Symbolic**: Applies unified diff patches from grad storage to param storage via `patch_tensor` (uses the `patch` CLI with fuzz=3). Only patches elements where `grad.data != 0` (key+value dims).
- **Query auto-update**: After patching key+value, derives query content by running `get_query_tensor` on the updated kv, merging LLM-generated keywords, sorting and deduplicating.

## Storage Layout

```
{relative_to}/{tensor_uid}/
├── shape                    # JSON: [2, 3]
├── storage/
│   ├── 0/data               # Element at flat index 0
│   ├── 1/data               # Element at flat index 1
│   └── 1/1/data             # Multi-digit index 11
```

## Demo: LLM Coder Simulator (Portable Harness Module)

This example demonstrates the core thesis: **composable ops + trainable experience + autograd = self-improving harness models**. An LLM-powered coder is composed entirely from reusable infrastructure ops — no custom logic except a termination validator. The same Module structure with different experience tensors becomes a different skill.

### Why This Matters

Traditional agent frameworks hard-code action parsing, state management, and orchestration logic. Each new capability requires new glue code. This framework replaces all of that with a **static DAG of composable tensor ops**:

- **Expert 1 decides WHAT** (action type + hint): `"text:输入shell命令"` or `"ctrl:按回车执行"`
- **Expert 2 knows HOW** (hint → real command): `"echo hello world"` or `"Enter"`
- **New action types** don't require changing Expert 2 — just add experience entries to Expert 1
- **Training improves both experts** via autograd — experience tensors are the learnable weights

### Architecture

```
workspace_ft[1]             — tmux instance ID
capture_op[1, 30]          = ft_tmux_capture_pane(ft_expand(workspace_ft, [1, 30]))
decision_expert[1, 30]     = ft_expert(capture_op, decision_exp)   → "text:hint"|"ctrl:hint"
cmd_expert[1, 30]          = ft_expert(decision_expert, cmd_exp)   → "echo hello"|"Enter"
send_text_op[1, 30]        = ft_tmux_send_text(cmd_expert, workspace_expanded)
send_ctrl_op[1, 30]        = ft_tmux_send_ctrl(cmd_expert, workspace_expanded)
switched_send[1, 30]       = ft_switch(decision_expert, [("text",send_text_op),("ctrl",send_ctrl_op)])
validator[1, 30]           = ft_coder_validator(ft_sequential(switched_send, sleep, capture))
output[1]                  = ft_recurrent(validator)
```

Every op except `ft_coder_validator` is shared infrastructure reusable across all harness models.

### Key Insight: Chain Experts, Don't Parse

Instead of one monolithic expert + custom parsers for each action type:

```python
# BAD: custom parsing, not reusable, not trainable
decision = parse_action(expert_output)  # brittle regex/string matching
if decision.type == "text":
    cmd = extract_command(expert_output)  # another custom parser
```

Chain two experts — each with its own trainable experience:

```python
# GOOD: composable, trainable, extensible
decision_expert = ft_expert(capture_op, decision_experience, topk=2)   # WHAT
cmd_expert = ft_expert(decision_expert, cmd_experience, topk=2)         # HOW
```

Expert 1's output feeds directly as Expert 2's input. Write-through ensures no double LLM calls.

### Experience Tensors (Learnable Weights)

```python
# Decision expert: terminal observation → action_type:hint
decision_experience = make_tensor([
    ["提示符后面没有其他文字", "命令行为空", "text:输入shell命令"],
    ["提示符后面有echo命令",   "命令行已有内容", "ctrl:按回车执行"],
], tmpdir)

# Cmd expert: hint → real command
cmd_experience = make_tensor([
    ["text:输入shell命令\n任务是列出目录",    "需要ls",   "ls"],
    ["text:输入shell命令\n任务是输出hello",   "需要echo", "echo hello world"],
    ["ctrl:按回车执行",                       "按Enter",  "Enter"],
], tmpdir)
```

These are `[N, 3]` QKV tensors — trainable via backward + `StSGD`. More experience entries = better generalization.

### Pipeline Composition

```python
# All shared ops — same for ANY harness model
workspace_expanded = ft_expand(workspace_ft, [1, MAX_ITERS])
capture_op = ft_tmux_capture_pane(workspace_expanded)
decision_expert = ft_expert(capture_op, decision_experience, task_prompt="...", topk=2)
cmd_expert = ft_expert(decision_expert, cmd_experience, task_prompt="...", topk=2)
send_text_op = ft_tmux_send_text(cmd_expert, workspace_expanded)
send_ctrl_op = ft_tmux_send_ctrl(cmd_expert, workspace_expanded)
switched_send = ft_switch(decision_expert, [
    ("text", "type", "send text to terminal", send_text_op),
    ("ctrl", "ctrl", "send control key",      send_ctrl_op),
])
sleep_op = ft_sleep(workspace_expanded, 0.5)
iteration_body = ft_sequential(switched_send, sleep_op, capture_op)

# Only this is harness-specific
validator = ft_coder_validator(iteration_body, max_iters=MAX_ITERS)
output = ft_recurrent(validator)

# ONE forward call — the prompt IS the task
output.ft_forward(make_tensor(["在终端中输出问候语hello world"], tmpdir))
```

### Portable Modules

A harness Module is defined by:
1. **Experience tensors** — the learned skill (trainable via autograd)
2. **Validator** — the termination condition (harness-specific)
3. **Pipeline structure** — identical across all modules (shared infrastructure)

Same Module structure + different experience = different behavior:

| Module | Experience 1 | Experience 2 | Validator |
|--------|-------------|-------------|-----------|
| Coder | terminal → action_type | hint → shell cmd | prompt returns to idle |
| Debugger | error → strategy | strategy → debug cmd | tests pass |
| Deployer | status → action | action → deploy cmd | service healthy |

### Run

```bash
python -m experience.future_tensor.test.test_llm_coder_simulator
```

### Design Principles

- **Static DAG, not imperative orchestration** — compose a graph of lazy tensors, then pull from the output
- **No shared mutable state** — coordination through tensor coordinates, not lists/dicts
- **Chain experts, don't parse** — two chained `ft_expert` ops replace custom parsers
- **Observe the world directly** — `ft_tmux_capture_pane` reads live terminal state, no shadow copies
- **Shape-first** — all inputs shaped `[1, max_iters]` or broadcastable
- **Self-improving** — experience tensors accumulate knowledge via backward + optimizer step

## Training Demo: Python to Viba Translation

The `example/naive_symbolic_transform_model/` directory demonstrates training from scratch — translating Python into **Viba**, a novel DSL that doesn't exist in any LLM's training corpus. Experience starts empty and accumulates correct mappings over 5 iterations (loss: 0.66 → 0.42, 36.5% reduction).

```bash
python -m experience.example.naive_symbolic_transform_model.train
```

## Auto-Encoder Example

The `example/auto-encoder/` directory contains a baseline stability test that runs the auto-encoder experiment multiple times to measure variance:

```bash
python -m example.auto-encoder.loop_run
```

This runs 10 experiments and reports mean, min, max, and standard deviation of the loss, useful for establishing baseline metrics and debugging reproducibility issues.

## Tests

```bash
# Unit tests (tensor_util inline tests)
python -m experience.symbolic_tensor.tensor_util.make_tensor
python -m experience.symbolic_tensor.tensor_util.slice_view
python -m experience.symbolic_tensor.tensor_util.patch_tensor

# Integration tests
python -m experience.test.test_gain_st_sgd
python -m experience.test.test_attention_vs_traditional
python -m experience.test.test_st_attention_followed_by_st_moe

# Individual function tests
python -m experience.symbolic_tensor.function.slice_attention_backward
python -m experience.symbolic_tensor.function.st_copy
python -m experience.symbolic_tensor.function.fork_tensor
python -m experience.symbolic_tensor.function.get_edit_distance_ratio
python -m experience.symbolic_tensor.function.st_stack
python -m experience.symbolic_tensor.function.slice_view
python -m experience.symbolic_tensor.function.slice_tensor

# FutureTensor function tests
python -m experience.future_tensor.function.ft_sequential
python -m experience.future_tensor.function.ft_switch

# FutureTensor 2nd-derivative tests
python -m experience.future_tensor.second_derivative.test.test_recurrent_2nd
python -m experience.future_tensor.second_derivative.test.test_sequential_2nd
python -m experience.future_tensor.second_derivative.test.test_switch_2nd
python -m experience.future_tensor.second_derivative.test.test_tmux_2nd

# Full training demo
python -m experience.example.naive_symbolic_transform_model.train
```

## Key Design Decisions

- **LLM as compute kernel**: Replaces matrix multiplication with semantic reasoning
- **Patch-based Optimizer**: `diff`/`patch` for efficient incremental experience updates
- **Two LLM backends**: `raw_llm_api` (default, lightweight) and `coding_agent` (tool access)
- **Symlinks for views, copies for mutations**: Shared storage for context, independent copies for LLM writes
- **Experience starts empty**: Learned entirely at runtime, not pre-seeded
- **Cold-start support**: Random retrieval and direct `+`-line extraction handle empty experience
- **Append-only experience**: Zeros out gradient for non-empty rows so optimizer skips them
- **Pythonic slicing**: `st_view_slicer` and `st_value_slicer` provide NumPy-like indexing syntax
- **Lazy async tensors**: FutureTensor defers LLM calls until materialization, enabling lazy graph construction
- **Status-encoded coefficients**: Success/failure/overflow encoded as floats in tensor data for autograd compatibility
- **Symbolic 2nd derivatives**: Meta-learning via dispatch policies — LLM reflects on its own reflections
- **Scalar anchor for 2nd derivative**: No Hessian matrix; each element's 2nd derivative is independent and parallel
- **GradFn owns reconstruction logic**: Each `*GradFn.forward` handles attribute reconstruction stripped by autograd, keeping `Ft*.backward` thin

## Dependencies

- Python 3.13+
- PyTorch
- `openai` (default LLM backend)
- `claude-agent-sdk` (alternative LLM backend)
- `Levenshtein` (edit distance loss)

## Installation

```bash
pip install torch openai claude-agent-sdk Levenshtein
```

## Quick Start

```python
from experience.symbolic_tensor import tensor, none

# Create a symbolic tensor
t = tensor(["hello world", "bonjour le monde"], "/tmp/my_tensors")

print(t.shape)           # torch.Size([2])
print(t.data)            # tensor([1., 1.], dtype=torch.bfloat16)
print(t.st_relative_to)  # '/tmp/my_tensors'
print(t.st_tensor_uid)   # 'a3f2...'

# Read text content
import os
path = os.path.join(t.st_relative_to, t.st_tensor_uid, "storage", "0", "data")
with open(path) as f:
    print(f.read())      # "hello world"

# Tensor ops (registered on torch.Tensor)
diff = t.st_get_diff(expected)     # unified diff
t.st_patch(grad)                   # apply patch
paths = t.st_file_paths()          # list all storage paths
t.st_assign(new_value)             # copy assignment
t.st_assign_view(new_value)        # symlink assignment
forks = t.st_fork(num_outputs=3)   # fork into N views

# Pythonic slicing
row_view = t.st_view_slicer[0, :]    # symlinked view
row_copy = t.st_value_slicer[0, :]   # independent copy

# Stack tensors
from experience.symbolic_tensor.function.st_stack import st_stack
stacked = st_stack([t1, t2, t3], dim=0)
```
