# Experience

A **self-improving harness framework** built as a PyTorch extension. Composable tensor ops + trainable experience + autograd = portable Modules that improve through training.

The result: **portable, self-improving agents** that share the same infrastructure but learn different skills through training. Like Claude Code slash commands, but standardized — typed tensors, autograd, shape contracts, trainable experience.

## Example: LLM Coder Simulator

An LLM-powered coder composed entirely from reusable ops — no custom logic except a termination validator. Two chained experts collaborate: Expert 1 decides WHAT to do, Expert 2 knows HOW to do it.

```
# [1, 30] live terminal text
capture_op       = ft_tmux_capture_pane(workspace)
# [1, 30] → "text:hint"|"ctrl:hint"
decision_expert  = ft_expert(capture_op, decision_exp)
# [1, 30] → "echo hello"|"Enter"
cmd_expert       = ft_expert(decision_expert, cmd_exp)
# [1, 30] routes by prefix
switched_send    = ft_switch(decision_expert, [("text",…),("ctrl",…)])
# [1, 30]
validator        = ft_coder_validator(ft_sequential(switched_send, sleep, capture))
# [1] reduced
output           = ft_recurrent(validator)
```

Every op except `ft_coder_validator` is shared infrastructure reusable across all harness models.

### Chain Experts, Don't Parse

Instead of one monolithic expert + custom parsers:

```python
# BAD: custom parsing, not reusable, not trainable
decision = parse_action(expert_output)  # brittle regex
cmd = extract_command(expert_output)    # another custom parser
```

Chain two experts — each with its own trainable experience:

```python
# GOOD: composable, trainable, extensible
decision_expert = ft_expert(capture_op, decision_experience, topk=2)   # WHAT
cmd_expert = ft_expert(decision_expert, cmd_experience, topk=2)         # HOW
```

Expert 1's output feeds directly as Expert 2's input. New action types don't require changing Expert 2 — just add experience entries to Expert 1.

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

These are `[N, 3]` QKV tensors — trainable via backward + `StSGD`. More experience = better generalization.

### Portable Modules

A harness Module is defined by:
1. **Experience tensors** — the learned skill (trainable via autograd)
2. **Validator** — the termination condition (harness-specific)
3. **Pipeline structure** — identical across all modules (shared ops)

Same structure + different experience = different behavior:

| Module | Experience 1 | Experience 2 | Validator |
|--------|-------------|-------------|-----------|
| Coder | terminal → action_type | hint → shell cmd | prompt returns to idle |
| Debugger | error → strategy | strategy → debug cmd | tests pass |
| Deployer | status → action | action → deploy cmd | service healthy |

```bash
python -m experience.future_tensor.test.test_llm_coder_simulator
```

## Why a Compute Graph?

Programs are sequential, branch, and loop. We compact all three into typed tensor ops:

| Control flow | Op | What it does |
|---|---|---|
| Sequential | `ft_sequential` | Do A then B |
| Branch | `ft_switch` | If X then A else B |
| Loop | `ft_recurrent` | Repeat until validator passes |

This makes agent generation trivial for LLMs. Instead of writing correct imperative code (hard — shared state, error handling, async coordination), the LLM composes a graph from ~10 typed ops (easy — slot-filling with shape verification). Three properties make this uniquely LLM-friendly:

1. **Structure vs behavior separation.** The LLM generates the graph (structure) once. Behavior is determined by experience tensors, which self-improve through training. The LLM doesn't need to get behavior right at generation time.
2. **Shape contracts = static verification.** If shapes don't match, the composition is wrong — detectable before execution. LLMs make fewer structural errors than behavioral errors.
3. **Fault-tolerant generation.** Even poor seed experience gets corrected by training. The graph just needs to be structurally correct.

## How It Works

### Tensor Elements Are Files

Each tensor element is a text file on disk. Numeric coefficients flow through standard PyTorch autograd while symbolic content (code, translations, commands) lives in files.

```
{relative_to}/{tensor_uid}/
├── shape                    # JSON: [2, 3]
├── storage/
│   ├── 0/data               # Element at flat index 0
│   ├── 1/data               # Element at flat index 1
│   └── 1/1/data             # Multi-digit index 11
```

### ExperienceTensor

An `[N, 3]` tensor where each row is `(query, key, value)`:
- **Query**: semantic keywords for Jaccard similarity retrieval
- **Key**: source domain content
- **Value**: target domain content

Starts empty, populated during training. The backward pass computes diffs, the optimizer patches entries.

### FutureTensor (Lazy Async Ops)

A `FutureTensor` is a scalar `torch.Tensor` monkey-patched with `ft_*` attributes — a reference to a pending computation. Elements materialize on-demand via LLM calls.

| Op | Purpose |
|----|---------|
| `ft_expert` | Query + retrieval + LLM translation (chainable) |
| `ft_switch` | Lazy control-flow branch selection |
| `ft_sequential` | Sequential evaluation with early-return on error |
| `ft_recurrent` | Generate-validate retry loop (reduces last dim) |
| `ft_tmux_*` | Terminal session ops (create, send_text, send_ctrl, capture) |
| `ft_slice` / `ft_unsqueeze` | Shape manipulation with autograd |

### Dual-Channel Gradients

| Channel | Carries | Computed by |
|---------|---------|-------------|
| **Numeric** | Float coefficients (bfloat16) | Standard autograd |
| **Symbolic** | Unified diffs in files | LLM compares actual vs expected |

Optimizer (`StSGD`) applies both: numeric SGD update + `patch` CLI applies diffs to experience files.

### Second Derivative (Meta-Learning)

The LLM reflects on its own reflections:

```python
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

## Design Principles

- **Static DAG, not imperative orchestration** — compose a graph of lazy tensors, then pull from output
- **Chain experts, don't parse** — two chained `ft_expert` ops replace custom parsers
- **No shared mutable state** — coordination through tensor coordinates
- **Observe the world directly** — read live state, no shadow copies
- **Self-improving** — experience accumulates knowledge via backward + optimizer step
- **Experience starts empty** — learned entirely at runtime, cold-start supported
- **LLM as compute kernel** — replaces matrix multiplication with semantic reasoning

## Roadmap

1. **Adapter mechanism for existing coding agents.** Two directions:
   - **Agent → Harness Module**: wrap Claude Code / OpenCode / OpenClaw / Hermes as a Harness Module — their tool interfaces become `ft_*` ops in the compute graph, gaining autograd and self-improvement for free.
   - **Harness Module → Agent skill**: export a trained Harness Module as a coding agent tool/skill — any agent can invoke it as a composable capability without knowing the internals.
2. **Tmux screen capture streams hub.** A shared capture stream that records ground-truth terminal interactions, producing standardized experience tensors. Multiple harness agents train from the same hub — enabling experience sharing and transfer learning across agents.

## Internals

Two LLM backends: `raw_llm_api` (OpenAI-compatible, lightweight) and `coding_agent` (Claude Agent SDK with file tools), dispatched concurrently via `asyncio.gather`. Autograd functions in `symbolic_tensor/function/` implement `StMoe` (query → retrieval → LLM translate), attention, slicing (symlink views vs copies), stack, fork, merge, and edit-distance loss. Tensor utilities (`tensor_util/`) provide `make_tensor`, `get_diff_tensor`, `patch_tensor` (unified diffs, cold-start support), and Pythonic methods registered on `torch.Tensor` (`st_pack`, `st_patch`, `st_get_diff`, `st_view_slicer[...]`, etc.).

## Installation

```bash
pip install torch openai claude-agent-sdk Levenshtein
```

Requires Python 3.13+.

## Architecture

```
experience/
├── symbolic_tensor/
│   ├── function/          # Autograd Functions: st_moe, st_attention, st_stack, slice_*, merge, fork, copy, loss
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
```
