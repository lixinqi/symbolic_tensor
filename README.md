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

## Internals

Two LLM backends: `raw_llm_api` (OpenAI-compatible, lightweight) and `coding_agent` (Claude Agent SDK with file tools), dispatched concurrently via `asyncio.gather`. Autograd functions in `symbolic_tensor/function/` implement `StMoe` (query → retrieval → LLM translate), attention, slicing (symlink views vs copies), stack, fork, merge, and edit-distance loss. Tensor utilities (`tensor_util/`) provide `make_tensor`, `get_diff_tensor`, `patch_tensor` (unified diffs, cold-start support), and Pythonic methods registered on `torch.Tensor` (`st_pack`, `st_patch`, `st_get_diff`, `st_view_slicer[...]`, etc.).

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

