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

### 1. Before the forward pass — enable gradient flow through the input

```python
import torch
from experience.future_tensor.second_derivative import need_2nd_derivative

second_derivative_start = torch.nn.Parameter(torch.ones(()))  # scalar anchor

input = need_2nd_derivative(input, second_derivative_start)
```

`need_2nd_derivative` returns `input` with `requires_grad=True`. Because `input` now
participates in the autograd graph, gradients computed during `loss.backward()` flow
through it and accumulate into `second_derivative_start.grad` via the placeholder
scalars returned by each 2nd-derivative op.

### 2. Run the harness model normally

```python
output = model(input)
loss = criterion(output, target)
loss.backward()
# After this: second_derivative_start.grad holds the accumulated
# scalar gradient from all 1st-derivative backward ops.
```

### 3. After the first backward — introspect the LLM reflections

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

`TracePolicy` is the default policy. It records every 2nd-derivative op call without
running any LLM — use it to inspect which reflections fired and what their inputs are.

---

## API reference

### `need_2nd_derivative(input, second_derivative_start)`

```python
need_2nd_derivative(
    input: torch.Tensor,                    # FutureTensor or SymbolicTensor
    second_derivative_start: nn.Parameter,  # scalar anchor (shape=(), value=1)
) -> torch.Tensor                           # input with requires_grad=True
```

Returns `input` with `requires_grad` set to `True`. Has no side effects beyond that —
no thread-local registration, no graph edge insertion, no dispatcher interaction.
Setting `requires_grad=True` ensures the autograd graph connects `input` to
`second_derivative_start` through the placeholder scalars that each 2nd-derivative
op returns, so `second_derivative_start.grad.backward()` can traverse them.

### `get_2nd_dispatcher(function_name)`

```python
from experience.future_tensor.second_derivative import get_2nd_dispatcher

dispatch = get_2nd_dispatcher(__function__)  # __function__ = current backward fn name
dispatch(arg_name2inputs)                    # fires the active policy
```

Called inside each 2nd-derivative backward function. Looks up the dispatcher registered
for `function_name` in the currently active `dispatch_policy`. If no policy is active,
uses `TracePolicy` with a module-level default collector.

Every 2nd-derivative op function returns a **placeholder scalar tensor** (`value=1`)
rather than a meaningful tensor — consistent with FutureTensor being 0D. The actual
derivative content is recorded by the policy.

### `dispatch_policy(policy)` — context manager

```python
with dispatch_policy(policy):
    ...
```

Sets the thread-local active policy for all `get_2nd_dispatcher` calls inside the block.

`policy` is a `Policy` instance — e.g. `TracePolicy(collector)`, or a custom subclass.
`TracePolicy` is the default when no `dispatch_policy` block is active.

Policies are **not** reentrant by default. Nesting two `dispatch_policy` blocks raises
`PolicyConflictError` unless the inner policy explicitly allows nesting.

---

## Dispatch policies

### `TracePolicy(collector: list)` — default

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
- Inspect the LLM reflections (1st-derivative outputs) before deciding what to do
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

class ExecutePolicy(Policy):
    """Run an LLM to reflect on each reflection."""

    def dispatch(self, fn: str, arg_name2inputs: dict) -> torch.Tensor:
        # call LLM with arg_name2inputs["grad_output"] (the 1st-derivative text)
        # and the original forward inputs to produce the 2nd derivative
        ...
        return torch.ones(())

with dispatch_policy(ExecutePolicy()):
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
│                            #          dispatch_policy, TracePolicy
├── policy.py                # Policy base class, ReflectionRecord, PolicyConflictError
├── context.py               # dispatch_policy context manager + thread-local state
├── dispatcher.py            # get_2nd_dispatcher; per-function dispatcher registry
├── need_2nd_derivative.py   # need_2nd_derivative: set requires_grad=True
├── trace_policy.py          # TracePolicy: non-destructive collector (default)
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
element (a text diff), recorded by the policy.

**Why `__function__` as the dispatcher key?**
Each 2nd-derivative wrapper is defined inside a specific backward function file, so
`__function__` (the module-level `__name__`) uniquely identifies which backward op is
being differentiated. Policies can branch on this name to apply different strategies
per op.

**No Hessian.**
Because `FutureTensor` is 0D, the second derivative is a scalar functional — there are
no cross-partial terms between different elements. Each element's 2nd derivative is
computed independently, in parallel, by the policy. This is what makes the mechanism
tractable at scale.

**`need_2nd_derivative` is minimal.**
It only sets `requires_grad=True` on `input` and returns it. The second derivative
machinery is entirely in the dispatch layer — `need_2nd_derivative` carries no
side effects and does not know about policies, dispatchers, or `second_derivative_start`
beyond receiving it as a parameter (currently unused beyond serving as a typed signal
to the caller that a 2nd-derivative pass is intended).

**`TracePolicy` as default.**
Making `TracePolicy` the default means a bare `second_derivative_start.grad.backward()`
without any `dispatch_policy` block is safe and useful: it collects all reflection
records into the module-level default collector. No LLM is ever invoked unexpectedly.
