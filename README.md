# Experience

A PyTorch extension that replaces numeric matrix multiplication with **LLM-powered text computation** inside neural network training loops. Each tensor element stores arbitrary text (code, translations, etc.) in files on disk, while numeric coefficients flow through standard autograd. An ExperienceTensor — a key-value store shaped as `[N, 3]` — serves as the learnable "weight" of the model, starting empty and being built up entirely by a patch-based optimizer across training iterations. The LLM acts as the compute kernel: during forward pass it reads experience entries to produce outputs; during backward pass it computes diffs; during the optimizer step `patch` applies diffs to experience entries incrementally.

The result is a model that can **learn to perform tasks it was never trained on** by accumulating experience at runtime, demonstrated by translating Python into Viba — a novel DSL that does not exist in the training corpus of any existing LLM.

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

## Demo: Python to Viba Translation

This example trains a model from scratch to translate Python code into **Viba** — a novel domain-specific language invented for this project that does not exist in the training corpus of any existing LLM. The LLM must learn to generate syntactically and semantically correct code in a language it has never seen, purely from the experience entries built up during training.

### Why Viba?

Existing LLMs have been trained on billions of lines of Python, JavaScript, Haskell, etc. Translating between these tests whether the model can *recall* patterns from pre-training. Viba eliminates this confound — any correct translation demonstrates genuine *generalization* driven by the experience mechanism.

Viba is an algebraic data type (ADT) definition language with these core constructs:

```viba
name := <type_expr>                     # type/function definition
<type_expr> := <binding> ...            # sequence of bindings
$var <type>                             # typed variable declaration
<- $var <type>                          # input binding (argument)
<- $expr                                # return binding (result)
# inline                                # inline expansion hint
```

Type expressions use four ADT combinators:

| Combinator | Syntax | Meaning | Example |
|-----------|--------|---------|---------|
| Sum | `\|` | Tagged union (branching) | `\| 'positive' \| 'zero' \| 'negative'` |
| Product | adjacency / `*` | Tuple composition | `str * int` |
| Exponent | `<-` | Function type | `int <- int` (unary), `str <- int <- float` (curried) |
| Tag | `` ` `` | Type-level tag/annotation | `` `JSON`` |

Additional constructs:

- **Match**: `Match[$x > 0 -> 'positive', $x == 0 -> 'zero', _ -> 'negative']`
- **Generics**: `list[$elem int]`
- **Dict literal**: `dict['first' = $a, 'second' = $b]`
- **Function type value**: `(int <- int)`
- **Literals**: `INT`, `FLOAT`, `BOOLEAN`, `STRING ("...")`, `SINGLE_STRING ('...')`, `TRIPLE_STRING ('''...''')`, `CODE_BLOCK ({...})`
- **Special types**: `void` / `()` (unit), `never` (bottom), `...` (ellipsis)
- **Tuple syntax**: `(A, B)` desugars to product type
- **Import**: `Import[path/to/file.viba]` references another Viba definition

### 1. Dataset

12 Python-to-Viba translation pairs covering fundamental patterns:

| Pattern | Python | Viba |
|---------|--------|------|
| Sequential | `seq.py` | `seq.viba` |
| Branching | `branch.py` | `branch.viba` |
| Loop | `loop.py` | `loop.viba` |
| Recursion | `recursion.py` | `recursion.viba` |
| Higher-order | `higher_order.py` | `higher_order.viba` |
| Data structures | `data_struct.py` | `data_struct.viba` |
| Default args | `default_arg.py` | `default_arg.viba` |
| List comprehension | `list_comp.py` | `list_comp.viba` |
| String formatting | `format_str.py` | `format_str.viba` |
| Guard clauses | `guard.py` | `guard.viba` |
| Accumulator | `accumulator.py` | `accumulator.viba` |
| Closure | `closure.py` | `closure.viba` |

### 2. Setup

```python
import tempfile
from pathlib import Path

from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.function.get_edit_distance_ratio import get_edit_distance_ratio
from experience.symbolic_tensor.optimizer.st_sgd import StSGD
from experience.example.naive_symbolic_transform_model.model import NaiveModel

DATASET_PAIRS = [
    "seq", "branch", "loop",
    "recursion", "higher_order", "data_struct",
    "default_arg", "list_comp", "format_str",
    "guard", "accumulator", "closure",
]
```

### 3. Build Tensors

```python
with tempfile.TemporaryDirectory() as tmpdir:
    # Input: symlinks to Python source files
    py_paths = [Path("dataset") / f"{name}.py" for name in DATASET_PAIRS]
    input_tensor = make_tensor(py_paths, tmpdir, symlink=True)

    # Expected: Viba code as strings
    viba_contents = [(Path("dataset") / f"{name}.viba").read_text() for name in DATASET_PAIRS]
    expected_tensor = make_tensor(viba_contents, tmpdir)

    # Experience: starts EMPTY — learned during training
    n = len(DATASET_PAIRS)
    experience_tensor = make_tensor([[""] * 3 for _ in range(n)], tmpdir)
```

### 4. Training Loop

```python
    model = NaiveModel(task_prompt="Translate Python To Viba", topk=1)
    model.load_experience(experience_tensor)
    optimizer = StSGD(model.parameters(), lr=1.0)

    for iteration in range(1, 6):
        optimizer.zero_grad()

        # Forward: LLM translates each Python file to Viba using experience
        output, selected_indexes = model(input_tensor)

        # Loss: Levenshtein edit distance ratio
        loss = get_edit_distance_ratio(output, expected_tensor)
        mean_loss = loss.mean().item()
        print(f"Iteration {iteration} loss: {mean_loss:.4f}")

        # Backward: computes symbolic gradients (diffs) via autograd
        loss.mean().backward()

        # Optimizer step: applies patches to experience via patch
        optimizer.step()
```

### 5. Run

```bash
python -m experience.example.naive_symbolic_transform_model.train
```

### 6. Results (5 iterations, 12 pairs)

Loss trajectory: `['0.6641', '0.5469', '0.4668', '0.4473', '0.4219']` — **converges** (36.5% reduction).

```
Dataset: 12 pairs
Experience: [24, 3]   # 24 entries (2x dataset size for exploration headroom)

Iteration 1/5  Mean loss: 0.6641
  output[0]: 'fun greet(name: str) -> str:\n    let greeting = "Hello"...'        # random language
  output[1]: 'func classify(x: int) -> str:\n    if x > 0:...'                  # Python-like
  Patches: applied=18 rejected=0 fuzzed=0 skipped=0

Iteration 3/5  Mean loss: 0.4668
  output[0]: 'greet :=\n    $message str\n    <- $name str\n    ...'              # Viba syntax!
  output[5]: 'make_pair :=\n    dict\n    <- $a str\n    <- $b str\n    ...'    # Viba syntax!
  Patches: applied=6 rejected=0 fuzzed=0 skipped=0

Iteration 5/5  Mean loss: 0.4219
  output[1]: "classify :=\n    | 'positive'\n    | 'zero'\n    | 'negative'..."  # exact Viba
  output[3]: 'factorial :=\n    <- $n int\n    # recursive\n    <- Match[...]'   # exact Viba
  output[11]: 'make_adder :=\n    $adder (int <- int)\n    <- $x int\n    ...'    # exact Viba

Patch stats (all 5 iterations): 34/34 applied, 0 rejected, 100% success rate.
```

Key observations:
- The model **starts with no Viba knowledge** — iteration 1 outputs are in random languages (Go, Rust, TypeScript-like).
- By iteration 3, the LLM begins using Viba syntax (`<-`, `$var`, `:=`) for some outputs.
- By iteration 5, several outputs match expected Viba code exactly (loss < 0.01 for branch, closure, guard).
- Experience entries accumulate correct Python-to-Viba mappings during training — e.g., entry `[22,23]` stores `classify(x) -> classify := | 'positive' | ...`.
- All 34 patches applied cleanly with 0% rejection rate across 5 iterations.

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
