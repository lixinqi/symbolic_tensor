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

### Natural PyTorch flow — end-to-end demo

```python
import torch
import torch.nn as nn

from experience.future_tensor.second_derivative import (
    need_2nd_derivative,
    dispatch_policy,
    TracePolicy,
)
from experience.future_tensor.function.ft_slice import ft_slice
from experience.future_tensor.function.ft_unsqueeze import ft_unsqueeze
from experience.future_tensor.function.ft_recurrent import ft_recurrent
from experience.future_tensor.function.ft_mean import ft_mean


class ToyHarnessModel(nn.Module):
    """Minimal harness: ft_unsqueeze → ft_slice → ft_recurrent."""

    def forward(self, input_ft):
        x = ft_unsqueeze(input_ft, dim=1)                     # (2, 1, 2)
        x = ft_slice(x, [slice(None), 0, slice(None)])        # (2, 2)
        output, _ = ft_recurrent(x, task_prompt="toy model")  # (2,)
        return output


# ── 1. Create the scalar anchor OUTSIDE the model ──
second_derivative_start = torch.ones(
    (), dtype=torch.bfloat16, requires_grad=True
)

# ── 2. Forward pass ──
model = ToyHarnessModel()
input_ft = ...  # a FutureTensor with shape (2, 2)
anchored = need_2nd_derivative(input_ft, second_derivative_start)
output = model(anchored)
loss = ft_mean(output)

# ── 3. First backward (builds the graph for 2nd derivative) ──
loss.backward(create_graph=True)

# After this: second_derivative_start.grad holds the accumulated
# scalar gradient from all 1st-derivative backward ops, AND its
# grad_fn chain contains the full backward graph.

# ── 4. Second backward — introspect LLM reflections ──
llm_reflections = []
with dispatch_policy(TracePolicy(llm_reflections)):
    second_derivative_start.grad.backward()

# llm_reflections now contains one entry per backward op that fired,
# in backward-traversal order (closest to second_derivative_start first):
# [
#   ReflectionRecord(fn=unsqueeze_forward,   inputs={"dim": 1, ...},               output=tensor(1.)),
#   ReflectionRecord(fn=slice_backward,      inputs={"original_shape": [2,1,2], ...}, output=tensor(1.)),
#   ReflectionRecord(fn=recurrent_backward,  inputs={"task_prompt": "toy model", ...}, output=tensor(1.)),
# ]
```

`TracePolicy` is the default policy. It records every 2nd-derivative op call without
running any LLM — use it to inspect which reflections fired and what their inputs are.

### `create_graph=True` is required

Without `create_graph=True`, PyTorch discards the intermediate backward graph after
computing `second_derivative_start.grad`. The subsequent
`second_derivative_start.grad.backward()` would have no graph to traverse and would
produce no 2nd-derivative records.

---

## API reference

### `need_2nd_derivative(input, second_derivative_start)`

```python
need_2nd_derivative(
    input: torch.Tensor,              # FutureTensor or SymbolicTensor, must be scalar
    second_derivative_start: torch.Tensor,  # scalar anchor with requires_grad=True
) -> torch.Tensor                         # input with computational dependency on anchor
```

Multiplies `input` by `second_derivative_start` to create a computational dependency.
During `loss.backward(create_graph=True)`, the gradient for `second_derivative_start`
includes the entire model backward graph. Calling
`second_derivative_start.grad.backward()` then naturally traverses that graph and
triggers the 2nd-derivative GradFn backward methods.

If `input` carries FutureTensor monkey-patched attributes, they are copied to the
result so downstream ops still see a valid FutureTensor.

`second_derivative_start` must be scalar (`shape == ()`) and have
`requires_grad=True`. Use:

```python
second_derivative_start = torch.ones((), dtype=torch.bfloat16, requires_grad=True)
```

### `get_2nd_dispatcher(function_name)`

```python
from experience.future_tensor.second_derivative import get_2nd_dispatcher

dispatch = get_2nd_dispatcher(recurrent_backward)  # pass the 1st-derivative fn object
dispatch(arg_name2inputs)                           # fires the active policy
```

Called inside each 2nd-derivative backward function. Receives the actual 1st-derivative
backward function object as the key. Looks up the dispatcher in the currently active
`dispatch_policy`. If no policy is active, uses `TracePolicy` with a module-level
default collector.

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
    fn: Callable                   # the 1st-derivative backward function object
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
    print(r.fn.__name__, list(r.inputs.keys()))
```

### Custom policy

Subclass `Policy` and implement `dispatch`:

```python
from experience.future_tensor.second_derivative.policy import Policy, ReflectionRecord

class ExecutePolicy(Policy):
    """Run an LLM to reflect on each reflection."""

    def dispatch(self, fn: Callable, arg_name2inputs: dict) -> torch.Tensor:
        # call LLM with arg_name2inputs["grad_output"] (the 1st-derivative text)
        # and the original forward inputs to produce the 2nd derivative
        ...
        return torch.ones(())

with dispatch_policy(ExecutePolicy()):
    second_derivative_start.grad.backward()
```

---

## How 2nd-derivative GradFns are structured

Each FutureTensor op whose 1st derivative participates in 2nd differentiation wraps
that 1st derivative inside a `torch.autograd.Function` (a *GradFn*). The GradFn's
`forward` IS the 1st derivative; its `backward` dispatches to the active Policy.

```python
# future_tensor/function/recurrent_2nd.py

import torch
from experience.future_tensor.second_derivative.dispatcher import get_2nd_dispatcher

class RecurrentGradFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, input, output, prompt_tensor, **kwargs):
        from experience.future_tensor.function.ft_recurrent_backward import recurrent_backward

        ctx.save_for_backward(grad_output, input, output, prompt_tensor)
        ctx._recurrent_backward_fn = recurrent_backward

        grad_input = recurrent_backward(grad_output, input, output, prompt_tensor, **kwargs)
        ctx._grad_input = grad_input
        return grad_input

    @staticmethod
    def backward(ctx, grad_grad_input):
        grad_output, input, output, prompt_tensor = ctx.saved_tensors

        # 1. Get the dispatcher keyed on the 1st-derivative backward function object
        dispatch = get_2nd_dispatcher(ctx._recurrent_backward_fn)

        # 2. Dispatch with named arguments
        dispatch({
            "grad_output":   grad_output,
            "input":         input,
            "output":        output,
            "prompt_tensor": prompt_tensor,
            "grad_input":    ctx._grad_input,
        })

        # 3. Return placeholder gradient for (grad_output, input, output, prompt_tensor, ...)
        return None, None, None, None, None, None, None, None, None
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
│                            #          dispatch_policy, TracePolicy, Policy,
│                            #          ReflectionRecord, PolicyConflictError
├── policy.py                # Policy base class, ReflectionRecord, PolicyConflictError
├── context.py               # dispatch_policy context manager + thread-local state
├── dispatcher.py            # get_2nd_dispatcher; per-function dispatcher registry
├── need_2nd_derivative.py   # need_2nd_derivative: computational dependency anchor
└── trace_policy.py          # TracePolicy: non-destructive collector (default)

future_tensor/function/      # GradFn wrappers live next to their 1st-derivative ops
├── recurrent_2nd.py         # RecurrentGradFn  (wraps recurrent_backward)
├── moe_2nd.py               # MoeGradFn        (wraps st_moe_backward)
├── slice_2nd.py             # SliceGradFn      (wraps slice_backward)
└── unsqueeze_2nd.py         # UnsqueezeGradFn  (wraps unsqueeze squeeze-via-slice)
```

---

## Design notes

**Why a scalar anchor?**
`second_derivative_start` has shape `()` — matching FutureTensor's scalar shape. This
keeps the 2nd derivative graph homogeneous: every node is a scalar, every edge carries
a scalar gradient. The "value" of the gradient is not a float but a `SymbolicTensor`
element (a text diff), recorded by the policy.

**Why the 1st-derivative function object as the dispatcher key?**
Passing the actual function object (e.g. `recurrent_backward`) rather than a string
means policies can branch on identity (`fn is recurrent_backward`) or inspect the
function's own attributes (`fn.__name__`, `fn.__module__`) without relying on
string conventions. It also avoids name collisions across modules and makes
refactoring safe — renaming the function updates the key automatically.

**No Hessian.**
Because `FutureTensor` is 0D, the second derivative is a scalar functional — there are
no cross-partial terms between different elements. Each element's 2nd derivative is
computed independently, in parallel, by the policy. This is what makes the mechanism
tractable at scale.

**`need_2nd_derivative` creates a computational dependency.**
It multiplies `input * second_derivative_start` so that during
`loss.backward(create_graph=True)` the gradient computation for
`second_derivative_start` includes the entire backward graph of the model.
FutureTensor monkey-patched attributes are copied to the result so downstream
ops still see a valid FutureTensor. The second derivative machinery itself is
entirely in the dispatch layer — `need_2nd_derivative` does not know about
policies or dispatchers.

**`TracePolicy` as default.**
Making `TracePolicy` the default means a bare `second_derivative_start.grad.backward()`
without any `dispatch_policy` block is safe and useful: it collects all reflection
records into the module-level default collector. No LLM is ever invoked unexpectedly.
