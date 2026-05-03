# FutureTensor Second Derivative

## Conceptual grounding

In this framework, derivatives are symbolic rather than numeric:

| Order | Mathematical object | Framework interpretation |
|---|---|---|
| Forward pass | `f(x)` | LLM generates output |
| 1st derivative (`backward`) | `∂L/∂x` | LLM *reflects* on its output — "how should this change to reduce loss?" |
| 2nd derivative | `∂²L/∂x²` | LLM reflects on the *reflection* — "how should the reflection itself change?" |

Because `FutureTensor` is a 0D tensor (scalar), there is no Hessian matrix. The second
derivative is a scalar-to-scalar map: a single symbolic correction applied to the
gradient text, not a matrix of cross-partials.

The first derivative (`recurrent_backward`, `st_moe_backward`) produces a
`SymbolicTensor` whose elements are LLM-generated text diffs — the reflections. The
second derivative runs a backward pass *through those reflections*, producing a second
layer of corrective text. This enables meta-learning: learning how to reflect better.

---

## Usage pattern

### 1. Before the forward pass — mark the 2nd derivative entry point

```python
import torch
from experience.future_tensor.second_derivative import need_2nd_derivative

second_derivative_start = torch.nn.Parameter(torch.ones(()))  # scalar anchor

input = need_2nd_derivative(input, second_derivative_start)
```

`need_2nd_derivative` inserts a transparent edge in the autograd graph that routes the
1st-derivative results (LLM reflections) back through `second_derivative_start`. When
`second_derivative_start.grad.backward()` is later called, autograd propagates through
the reflection tensors.

### 2. Run the harness model normally

```python
output = model(input)
loss = criterion(output, target)
loss.backward()
# After this: second_derivative_start.grad holds the accumulated
# symbolic gradient from all 1st-derivative backward ops.
```

### 3. After the first backward — introspect or compute the 2nd derivative

```python
from experience.future_tensor.second_derivative import dispatch_policy, TracePolicy

llm_reflections = []
with dispatch_policy(TracePolicy(llm_reflections)):
    second_derivative_start.grad.backward()

# llm_reflections now contains one entry per backward op that fired:
# [
#   ReflectionRecord(
#       fn="recurrent_backward",
#       inputs={"grad_output": ..., "input": ..., "output": ..., "prompt_tensor": ...},
#       output=<placeholder scalar tensor, value=1>,
#   ),
#   ReflectionRecord(fn="moe_backward", inputs={...}, output=...),
#   ...
# ]
```

`TracePolicy` is non-destructive — it records every 2nd-derivative op call without
actually running any LLM. Use it to inspect *which* reflections would be differentiated
and *what* their inputs are, before committing to a full 2nd-derivative run.

To actually execute the 2nd derivative (reflection of reflection):

```python
with dispatch_policy('default'):
    second_derivative_start.grad.backward()
```

---

## API reference

### `need_2nd_derivative(input, second_derivative_start)`

```python
need_2nd_derivative(
    input: torch.Tensor,                    # FutureTensor or SymbolicTensor
    second_derivative_start: nn.Parameter,  # scalar anchor (shape=(), value=1)
) -> torch.Tensor                           # input, unchanged
```

Returns `input` unchanged. Registers `second_derivative_start` in module-level
thread-local state so that every subsequent `get_2nd_dispatcher` call within the
same thread accumulates its placeholder scalar output into
`second_derivative_start.grad` when `second_derivative_start.grad.backward()` is
later called. No graph edge is inserted and the autograd graph of the harness
model is not modified.

### `get_2nd_dispatcher(function_name)`

```python
from experience.future_tensor.second_derivative import get_2nd_dispatcher

dispatch = get_2nd_dispatcher(__function__)  # __function__ = current backward fn name
dispatch(arg_name2inputs)                    # fires the active policy
```

Called inside each 2nd-derivative backward function. Looks up the dispatcher registered
for `function_name` in the currently active `dispatch_policy`. If no policy is active,
raises `NoPolicyError`.

Every 2nd-derivative op function returns a **placeholder scalar tensor** (`value=1`)
rather than a meaningful tensor — consistent with FutureTensor being 0D. The actual
derivative content is produced asynchronously by the policy.

### `dispatch_policy(policy)` — context manager

```python
with dispatch_policy(policy):
    ...
```

Sets the thread-local active policy for all `get_2nd_dispatcher` calls inside the block.

`policy` is one of:
- The string `'default'` — uses `DefaultPolicy`
- A `Policy` instance — e.g. `TracePolicy(collector)`, or a custom subclass

Policies are **not** reentrant by default. Nesting two `dispatch_policy` blocks raises
`PolicyConflictError` unless the inner policy explicitly allows nesting.

---

## Dispatch policies

### `DefaultPolicy`

Runs the actual 2nd derivative: for each backward op, invokes an LLM to reflect on the
reflection. The LLM receives:

- The 1st-derivative output (the reflection text) as the "gradient to differentiate"
- The original inputs to the 1st-derivative backward (forward output, input, prompt)
- A system prompt explaining that it is reflecting on a reflection

Produces a `SymbolicTensor` element containing "how should this reflection change?"

```python
with dispatch_policy('default'):
    second_derivative_start.grad.backward()
```

### `TracePolicy(collector: list)`

Non-destructive. Records every dispatch call into `collector` without running any LLM.
Each record is a `ReflectionRecord`:

```python
@dataclass
class ReflectionRecord:
    fn: str                        # backward function name, e.g. "recurrent_backward"
    inputs: dict[str, Any]         # arg_name -> tensor/value passed to dispatch
    output: torch.Tensor           # placeholder scalar (value=1)
    timestamp: float               # time.monotonic() at dispatch time
```

Use `TracePolicy` to:
- Audit which backward functions fired and with what arguments
- Dry-run before committing to an expensive 2nd-derivative LLM pass
- Build custom schedulers that selectively promote trace records to full execution

```python
records = []
with dispatch_policy(TracePolicy(records)):
    second_derivative_start.grad.backward()

for r in records:
    print(r.fn, list(r.inputs.keys()))
```

### Custom policy

Subclass `Policy` and implement `dispatch`:

```python
from experience.future_tensor.second_derivative.policy import Policy, ReflectionRecord

class SelectivePolicy(Policy):
    """Only run 2nd derivative for recurrent backward ops; trace the rest."""

    def __init__(self, trace_collector):
        self._trace = trace_collector
        self._default = DefaultPolicy()

    def dispatch(self, fn: str, arg_name2inputs: dict) -> torch.Tensor:
        if fn == "recurrent_backward":
            return self._default.dispatch(fn, arg_name2inputs)
        record = ReflectionRecord(fn=fn, inputs=arg_name2inputs,
                                  output=torch.ones(()))
        self._trace.append(record)
        return record.output

with dispatch_policy(SelectivePolicy(trace)):
    second_derivative_start.grad.backward()
```

---

## How 2nd-derivative op functions are structured

Each backward function in `future_tensor/function/` that participates in 2nd
differentiation has a corresponding thin wrapper in `second_derivative/function/`.
The wrapper follows a fixed three-step pattern:

```python
# second_derivative/function/recurrent_2nd.py

def recurrent_2nd_backward(grad_output, input, output, prompt_tensor, **kwargs):
    from experience.future_tensor.second_derivative import get_2nd_dispatcher

    # 1. Get the dispatcher for this backward function
    dispatch = get_2nd_dispatcher("recurrent_backward")

    # 2. Dispatch with named arguments
    dispatch({
        "grad_output":   grad_output,
        "input":         input,
        "output":        output,
        "prompt_tensor": prompt_tensor,
        **kwargs,
    })

    # 3. Return placeholder scalar — the policy handles actual content
    return torch.ones(())
```

The `arg_name2inputs` dict mirrors the argument names of the *1st*-derivative backward
function exactly, so `TracePolicy` records are directly readable alongside the 1st
backward's source.

---

## Module layout

```
second_derivative/
├── README.md
├── __init__.py              # exports: need_2nd_derivative, get_2nd_dispatcher,
│                            #          dispatch_policy, TracePolicy, DefaultPolicy
├── policy.py                # Policy base class, ReflectionRecord, PolicyConflictError
├── context.py               # dispatch_policy context manager + thread-local state
├── dispatcher.py            # get_2nd_dispatcher; per-function dispatcher registry
├── need_2nd_derivative.py   # need_2nd_derivative: graph edge insertion
├── default_policy.py        # DefaultPolicy: runs LLM reflection-of-reflection
├── trace_policy.py          # TracePolicy: non-destructive collector
└── function/
    ├── recurrent_2nd.py     # 2nd derivative wrapper for recurrent_backward
    └── moe_2nd.py           # 2nd derivative wrapper for moe_backward
```

---

## Design notes

**Why a scalar anchor?**
`second_derivative_start` has shape `()` — matching FutureTensor's scalar shape. This
keeps the 2nd derivative graph homogeneous: every node is a scalar, every edge carries
a scalar gradient. The "value" of the gradient is not a float but a `SymbolicTensor`
element (a text diff), dispatched by the policy.

**Why `__function__` as the dispatcher key?**
Each 2nd-derivative wrapper is defined inside a specific backward function file, so
`__function__` (the module-level `__name__`) uniquely identifies which backward op is
being differentiated. Policies can branch on this name to apply different strategies
per op — for example, running LLM for `recurrent_backward` but tracing `moe_backward`.

**No Hessian.**
Because `FutureTensor` is 0D, the second derivative is a scalar functional — there are
no cross-partial terms between different elements. Each element's 2nd derivative is
computed independently, in parallel, by the policy. This is what makes the mechanism
tractable at scale.

**`need_2nd_derivative` is a pass-through.**
It returns `input` unchanged and does not touch the autograd graph. The connection
between each dispatched op and `second_derivative_start` is established by the
dispatcher: each 2nd-derivative op appends its placeholder scalar to a list stored
under `second_derivative_start` in thread-local state. When
`second_derivative_start.grad.backward()` is called, autograd traverses that list
and fires each registered 2nd-derivative op. This keeps the harness-model graph
fully unmodified.

**Policy composability.**
`TracePolicy` + manual promotion is the recommended pattern for production use:

```python
# Phase 1: trace cheaply
records = []
with dispatch_policy(TracePolicy(records)):
    second_derivative_start.grad.backward()

# Phase 2: selectively execute expensive records
important = [r for r in records if should_run_2nd(r)]
for r in important:
    DefaultPolicy().dispatch(r.fn, r.inputs)
```

This separates graph traversal (cheap) from LLM calls (expensive) and gives full
control over budget, priority, and batching.
