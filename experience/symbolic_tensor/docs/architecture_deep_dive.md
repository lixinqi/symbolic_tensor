# Symbolic Tensor Architecture Deep Dive

## 1. Core Design Philosophy

The essence of Symbolic Tensor is: **extending PyTorch's tensor abstraction from a "numerical container" to a "text/code content container" while maintaining the complete autograd mechanism**.

### 1.1 Dual-Channel Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Symbolic Tensor                          │
├─────────────────────────────────────────────────────────────┤
│  Numerical Channel                                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ tensor.data = [1.0, 1.0, 0.0, 1.0]                   │  │
│  │ - Stored in GPU/CPU memory                           │  │
│  │ - Coefficient meaning: 1.0 = has content, 0.0 = empty/masked │
│  │ - Participates in PyTorch autograd computation graph │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  Symbolic Channel                                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ {st_relative_to}/{st_tensor_uid}/storage/            │  │
│  │ ├── 0/data  →  "Hello world\n"                       │  │
│  │ ├── 1/data  →  "def foo():\n    return 42\n"         │  │
│  │ ├── 2/data  →  (does not exist, corresponds to 0.0)  │  │
│  │ └── 3/data  →  "SELECT * FROM users\n"               │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Why Two Channels?

| Requirement | Numerical Channel Solves | Symbolic Channel Solves |
|-------------|--------------------------|-------------------------|
| Gradient propagation | ✓ PyTorch autograd | - |
| Store arbitrary text | - | ✓ Disk files |
| Sparsity representation | ✓ 0.0/1.0 coefficients | - |
| Incremental updates | - | ✓ unified diff |
| LLM processing | - | ✓ Text format |

---

## 2. Storage Mechanism Details

### 2.1 Flat Index to Path Mapping

Multi-dimensional tensors use row-major (C-order) flattening, then each digit becomes a directory level:

```python
# Tensor shape: [2, 3]
# Logical coordinates (1, 2) → flat index = 1*3 + 2 = 5 → path "5/data"
# Logical coordinates (0, 0) → flat index = 0 → path "0/data"

# Flat index 12 → path "1/2/data" (one directory per digit)
# Flat index 123 → path "1/2/3/data"
```

Benefits of this design:
1. **Avoid too many files in a single directory**: Large tensors won't create millions of files in one directory
2. **Fast lookup**: O(log10(N)) directory traversal
3. **Easy debugging**: Human-readable path structure

### 2.2 make_tensor Workflow

```python
def make_tensor(nested_data, relative_to):
    # 1. Infer shape
    shape = _get_shape(nested_data)  # e.g., [2, 3]

    # 2. Create zero-filled PyTorch tensor
    tensor = make_none_tensor(shape, relative_to)
    # tensor.data all 0.0
    # tensor.st_relative_to = relative_to
    # tensor.st_tensor_uid = uuid4()

    # 3. Flatten and write to disk
    for i, content in enumerate(flatten(nested_data)):
        if content is not None:
            path = f"{relative_to}/{tensor_uid}/storage/{digits(i)}/data"
            write(path, content)
            tensor.data[coords(i)] = 1.0  # Mark as having content

    return tensor
```

---

## 3. Autograd Integration: torch.autograd.Function Pattern

### 3.1 Why Can't We Use Standard PyTorch Operations?

Standard PyTorch operations only handle numerical values, they cannot handle the symbolic channel. Symbolic Tensor requires:

1. **Forward**: Copy/link files + update coefficients
2. **Backward**: Compute diff + propagate symbolic gradients

### 3.2 SliceTensor Example

```python
class SliceTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, slice_list):
        # Symbolic channel: copy files to new directory
        output = _raw_slice_tensor(input, slice_list)

        # Save information needed for backward
        ctx.save_for_backward(input, output)
        _save_st_attrs(ctx, input=input, output=output)
        ctx.per_dim = _build_per_dim(slice_list, input.size())

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        _restore_st_attrs(ctx, input=input, output=output)

        # Symbolic gradient processing
        grad_output = _resolve_grad_output(output, grad_output)
        grad_input = slice_backward(grad_output, input, output, ctx.per_dim)

        # Register to global registry (because autograd strips st_* attributes)
        if grad_input is not None:
            symbolic_grad_registry.register(input.st_tensor_uid, grad_input)

        return grad_input, None
```

### 3.3 Key Detail: ctx.save_for_backward Strips Custom Attributes

```python
# PyTorch's save_for_backward only saves tensor.data
# st_relative_to, st_tensor_uid and other attributes will be lost!

# Solution: manually save
ctx.st_attrs = {}
for name, tensor in [("input", input), ("output", output)]:
    for attr in ("st_relative_to", "st_tensor_uid"):
        if hasattr(tensor, attr):
            ctx.st_attrs[name][attr] = getattr(tensor, attr)
```

---

## 4. Symbolic Grad Registry: Solving the Attribute Stripping Problem

### 4.1 Problem

When gradients pass between multiple `torch.autograd.Function` instances, PyTorch will:
1. Create new tensors to store gradients
2. Discard all custom attributes (`st_*`) from the original tensor

### 4.2 Solution: Thread-Local Registry

```python
# symbolic_grad_registry.py
_local = threading.local()

def register(key, symbolic_tensor):
    """Producer backward registers symbolic gradient"""
    _get_store()[key] = symbolic_tensor

def pop(key):
    """Consumer backward or optimizer retrieves and removes"""
    return _get_store().pop(key, None)
```

### 4.3 Usage Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Function A  │ →  │ Function B  │ →  │  Optimizer  │
│  backward   │    │  backward   │    │    step     │
└─────────────┘    └─────────────┘    └─────────────┘
      │                  │                   │
      │ register(        │ pop(a_uid)        │ pop(b_uid)
      │  a_out_uid,      │ register(         │
      │  grad)           │  b_out_uid,       │
      ↓                  │  grad)            ↓
┌─────────────────────────────────────────────────────┐
│              symbolic_grad_registry                 │
│  { "a_out_uid": grad_a, "b_out_uid": grad_b }      │
└─────────────────────────────────────────────────────┘
```

---

## 5. st_attention: Symbolic Attention Mechanism

### 5.1 Pipeline

```
input: (batch, seq_len)
mask:  (batch, seq_len, seq_len)
              │
              ▼
    slice_attention
    Select tokens each query can see based on mask
              │
              ▼
sliced: (batch, seq_len, seq_len)
    Each [b, q, :] contains token content query q can see
              │
              ▼
    merge(axis=-1)
    Merge tokens in the last dimension into single text
              │
              ▼
output: (batch, seq_len)
    Each element is the merged context
```

### 5.2 Causal Mask Example

```python
# input = [["a", "b", "c"]]
# mask = lower triangular (causal)
#        [[1, 0, 0],
#         [1, 1, 0],
#         [1, 1, 1]]

# sliced content:
#   [0, 0] = "a"
#   [0, 1] = "a", "b"
#   [0, 2] = "a", "b", "c"

# merged output:
#   [0, 0] = merge(["a"]) = frame(0, "a")
#   [0, 1] = merge(["a", "b"]) = frame(0, "a") + frame(1, "b")
#   [0, 2] = merge(["a", "b", "c"]) = frame(0, "a") + frame(1, "b") + frame(2, "c")
```

### 5.3 TextMerger Format

```
===ST-MERGED-CONTENT===ST-MERGED-CONTENT===ST-MERGED-CONTENT===
index: 0
coefficient: 1.0
content:

  Hello world
===ST-MERGED-CONTENT===ST-MERGED-CONTENT===ST-MERGED-CONTENT===
index: 1
coefficient: 1.0
content:

  def foo():
      return 42
```

---

## 6. st_moe: Symbolic Mixture of Experts

### 6.1 Concept

st_moe replaces traditional MoE's "expert weights" with "experience entries", where each entry is a (query, key, value) triplet:

```python
experience = make_tensor([
    # [query, key, value]
    ["greeting\nhello\nworld", "Hello world", "Bonjour le monde"],
    ["farewell\ngoodbye",      "Goodbye",     "Au revoir"],
], tmpdir)
# shape: (num_experts, 3)
```

### 6.2 Forward Flow

```
input: "Hello world in English"
              │
              ▼
    get_query_tensor()
    LLM extracts keywords from input
              │
              ▼
query: "hello\nworld\nenglish"
              │
              ▼
    select_qkv_indexes()
    Compute similarity between query and experience[:,0]
    Select top-k experiences
              │
              ▼
selected: [(0, sim_score), ...]
              │
              ▼
    Construct prompt + LLM call
    "Given experience: {key} → {value}, translate: {input}"
              │
              ▼
output: "Bonjour le monde en français"
```

### 6.3 Backward: Gradient Distribution

```
grad_output: "Should use formal French: 'Bonjour le monde' → 'Bonjour au monde'"
              │
              ▼
    st_moe_backward()
              │
    ┌─────────┴─────────┐
    ▼                   ▼
grad_input          grad_experience
(suggestions        (diff for related
 to modify input)    entries in experience)
```

---

## 7. StSGD Optimizer

### 7.1 Dual-Channel Update

```python
def step(self):
    for param in self.params:
        grad = symbolic_grad_registry.pop(param.st_tensor_uid)

        # 1. Numerical channel: exponential moving average
        param.data = (1 - lr) * param.data + lr * grad.data

        # 2. Symbolic channel: apply diff patch
        stats = patch_tensor(param, grad)
        # stats = {"applied": 2, "rejected": 0, "fuzzed": 0, "skipped": 1}
```

### 7.2 Automatic Query Update

For experience tensors, when key/value is updated, automatically extract keywords from new content to update query:

```python
# Original experience[0] = ["hello\nworld", "Hello", "Bonjour"]
# Update value: "Bonjour" → "Bonjour tout le monde"
# Auto-update query: "hello\nworld" → "hello\nworld\nbonjour\ntout\nle\nmonde"
```

---

## 8. Gradient Representation: Unified Diff

### 8.1 Why Diff Instead of Directly Storing New Values?

| Approach | Pros | Cons |
|----------|------|------|
| Store new value | Simple | Cannot do incremental updates, cannot combine |
| Unified Diff | Incremental, composable, auditable | Requires patch tool |

### 8.2 get_diff_tensor Implementation

```python
def get_diff_tensor(lvalue, rvalue):
    """Compute element-wise unified diff"""
    for coords in all_coordinates(lvalue.shape):
        lpath = get_storage_path(lvalue, coords)
        rpath = get_storage_path(rvalue, coords)

        # Call system diff tool
        diff = subprocess.run(["diff", "-u", lpath, rpath])

        diffs.append(diff.stdout)

    return make_tensor(diffs, lvalue.st_relative_to)
```

### 8.3 Diff Format Example

```diff
--- data
+++ data
@@ -1,3 +1,3 @@
 def foo():
-    return 42
+    return 43
```

---

## 9. Design Patterns Summary

### 9.1 View vs Copy

| Operation | View (symlink) | Copy |
|-----------|----------------|------|
| slice_view | ✓ | - |
| slice_tensor | - | ✓ |
| Use case | Read-only, shared memory | Needs independent modification |

### 9.2 Forward/Backward Symmetry

```
Forward:  input → operation → output
          save ctx.st_attrs

Backward: grad_output → inverse_operation → grad_input
          restore ctx.st_attrs
          register to symbolic_grad_registry
```

### 9.3 Error Handling

- When file doesn't exist, return `None`, corresponding coefficient is 0.0
- When diff has no changes, return empty string or `None`
- When patch fails, record to stats, don't throw exception

---

## 10. Extension Points

1. **Custom TextMerger**: Implement `pack()/unpack()` interface
2. **New Autograd Function**: Inherit from `torch.autograd.Function`
3. **Custom Retrieval Method**: Replace similarity computation in `select_qkv_indexes`
4. **New LLM Backend**: Configure via `llm_method` and `llm_env` parameters
