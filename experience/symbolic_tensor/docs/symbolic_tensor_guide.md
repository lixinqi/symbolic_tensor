# Symbolic Tensor Module Documentation

## 1. Core Concept: What is Symbolic Tensor?

**Symbolic Tensor** is a mechanism that extends PyTorch tensor storage from memory to disk. Unlike regular PyTorch tensors that directly store numerical values, symbolic tensors store each element's content as files on disk.

### 1.1 Dual Representation

Each symbolic tensor element has two channels:

| Channel | Storage Location | Purpose |
|---------|------------------|---------|
| **Numerical Channel** | `tensor.data` | Stores coefficients (0.0 or 1.0), used for gradient computation |
| **Symbolic Channel** | `{st_relative_to}/{st_tensor_uid}/storage/` | Stores actual file content |

### 1.2 Key Attributes

```python
tensor.st_relative_to  # Storage root directory
tensor.st_tensor_uid   # Unique identifier
```

Example directory structure:
```
/tmp/xxx/
└── abc123/                  # st_tensor_uid
    ├── shape                # Shape info in JSON format
    └── storage/
        ├── 0/data          # Content at index 0
        ├── 1/data          # Content at index 1
        └── 1/2/data        # Content at index 12 (multi-digit)
```

---

## 2. Directory Structure

```
experience/symbolic_tensor/
├── __init__.py                    # Main entry point
├── tensor_util/                    # Core utilities (~20 files)
│   ├── make_tensor.py             # Create symbolic tensor from nested list
│   ├── make_none_tensor.py        # Create zero-filled symbolic tensor
│   ├── slice_view.py              # Symbolic view slicing (symlink-based)
│   ├── get_diff_tensor.py         # Compute unified diff between two tensors
│   ├── patch_tensor.py            # Apply unified diff patch
│   └── register_tensor_ops.py     # Register custom tensor operations
├── function/                       # Autograd Functions (~20 files)
│   ├── symbolic_grad_registry.py   # Thread-local gradient registry
│   ├── slice_view.py              # Differentiable slice_view
│   ├── st_moe.py                  # Sparse Token MoE
│   └── get_query_tensor.py        # LLM keyword generation
├── module/                        # PyTorch nn.Module wrappers
│   └── st_moe.py                  # StMoeModule
└── optimizer/                     # Custom optimizers
    └── st_sgd.py                  # Symbolic SGD
```

---

## 3. Core API

### 3.1 `make_tensor(nested_data, relative_to, symlink=False)`

Create symbolic tensor from nested list:

```python
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor

# 2D tensor: [num_experience, 3(query,key,value)]
experience = make_tensor([
    ["greeting\nhello\nworld", "Hello world", "Bonjour le monde"],
    ["farewell\ngoodbye", "Goodbye", "Au revoir"],
], tmpdir)
```

**Features**:
- Automatically persists content to disk
- Returns tensor with all coefficients set to 1.0
- Supports Path type and symlink mode

### 3.2 `slice_view(input, slice_tensors)`

Create symbolic view (symlink-based):

```python
from experience.symbolic_tensor.tensor_util.slice_view import slice_view

# Select row 0, columns [1, 2]
kv_view = slice_view(experience, [0, slice(1, 3)])
```

### 3.3 `get_diff_tensor(lvalue, rvalue)` → unified diff

Compute unified diff between two tensors:

```python
diff = get_diff_tensor(output_tensor, expected_tensor)
# diff[i] contains the difference between lvalue[i] and rvalue[i]
```

Output format:
```
--- data
+++ data
@@ -1 +1 @@
-old content
+new content
```

### 3.4 `patch_tensor(lvalue, rvalue)` → stats

Apply unified diff patch to target files:

```python
stats = patch_tensor(experience, diff_grad)
# stats: {"applied": 2, "rejected": 0, "fuzzed": 0, "skipped": 1}
```

---

## 4. Symbolic Gradient Registry

PyTorch's autograd strips custom attributes (`st_*`) during backpropagation. `symbolic_grad_registry` solves this problem:

```python
from experience.symbolic_tensor.function import symbolic_grad_registry

# Register in producer backward
symbolic_grad_registry.register(output.st_tensor_uid, symbolic_grad)

# Retrieve in consumer backward or optimizer
grad = symbolic_grad_registry.pop(output.st_tensor_uid)
```

This is a **thread-local** storage, ensuring safety in multi-threaded environments.

---

## 5. StSGD Optimizer

`StSGD` is a custom SGD optimizer that supports dual-channel updates:

### 5.1 Numerical Channel
```python
param.data = (1 - lr) * param.data + lr * grad.data
```

### 5.2 Symbolic Channel
```python
# Only apply patch to elements where grad.data != 0
stats = patch_tensor(kv_param, kv_grad)
```

### 5.3 Automatic Query Update
```python
# Extract keywords from key+value content, update query
query_content = extract_keywords(key_text) + extract_keywords(value_text)
write_to_query_file(query_content)
```

---

## 6. End-to-End Example: From Training to Inference

```python
import tempfile
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.optimizer.st_sgd import StSGD
from experience.example.naive_symbolic_transform_model.model import NaiveModel

# 1. Create dataset
input_tensor = make_tensor(["Python code..."], tmpdir)
expected_tensor = make_tensor(["Viba code..."], tmpdir)

# 2. Initialize experience (query, key, value)
experience_tensor = make_tensor([
    ["python\nviba\ntranslate", "Python code", "Viba code"],
], tmpdir)

# 3. Create model
model = NaiveModel(task_prompt="Translate Python To Viba", topk=1)
model.load_experience(experience_tensor)

# 4. Create optimizer
optimizer = StSGD(model.parameters(), lr=1.0)

# 5. Training loop
for iteration in range(5):
    optimizer.zero_grad()

    # Forward: LLM generates output based on experience
    output, selected = model(input_tensor)

    # Loss: compute edit distance ratio
    loss = get_edit_distance_ratio(output, expected_tensor)

    # Backward: compute gradients
    loss.mean().backward()

    # Optimizer step: update coefficients + apply diff patch + update query
    optimizer.step()
```

---

## 7. Symbolic Gradients vs Regular Gradients

| Aspect | Regular PyTorch Tensor | Symbolic Tensor |
|--------|------------------------|-----------------|
| **Storage** | GPU/CPU memory | Disk files |
| **Content Type** | Numerical values | Arbitrary text/code |
| **Gradient Form** | Numerical gradients | Unified diff |
| **Update Method** | `data -= lr * grad` | `patch_tensor(param, grad_diff)` |
| **Use Case** | Standard ML | LLM-enhanced ML / Experience replay |

---

## 8. Design Motivation

Symbolic Tensor design goals:

1. **Store arbitrary-sized content**: Not limited by GPU memory
2. **Support LLM-driven content generation**: Text/code as learning targets
3. **Unified diff as gradients**: Efficient incremental updates
4. **Experience replay**: Retrieve historical experience via semantic similarity
