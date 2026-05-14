# FutureTensor Backward Dispatch

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

from experience.future_tensor.backward_dispatch import (
    need_reflection,
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
sds = torch.ones((), dtype=torch.bfloat16, requires_grad=True)

# ── 2. Forward pass ──
model = ToyHarnessModel()
input_ft = ...  # a FutureTensor with shape (2, 2)
anchored = need_reflection(input_ft, sds)
output = model(anchored)
loss = ft_mean(output)

# ── 3. First backward (builds the graph for 2nd derivative) ──
# Use dispatch_policy to trace 1st-derivative ops
records_1st = []
with dispatch_policy(TracePolicy(records_1st)):
    loss.backward(create_graph=True)

# ── 4. Second backward — introspect LLM reflections ──
records_2nd = []
with dispatch_policy(TracePolicy(records_2nd)):
    sds.grad.backward()

# records_2nd now contains one entry per backward op that fired,
# in backward-traversal order:
# [
#   ReflectionRecord(fn=unsqueeze_forward,   inputs={"dim": 1, ...},               output=tensor(1.)),
#   ReflectionRecord(fn=slice_backward,      inputs={"original_shape": [2,1,2], ...}, output=tensor(1.)),
#   ReflectionRecord(fn=recurrent_backward,  inputs={"task_prompt": "toy model", ...}, output=tensor(1.)),
# ]
```

`TracePolicy` records every dispatch call without running any LLM — use it to inspect
which ops fired and what their inputs are.

`dispatch_policy` is the same context manager for both 1st and 2nd derivative tracing.
They are never active simultaneously: 1st runs during `loss.backward(create_graph=True)`,
2nd runs during `sds.grad.backward()`.

### `create_graph=True` is required

Without `create_graph=True`, PyTorch discards the intermediate backward graph after
computing `sds.grad`. The subsequent `sds.grad.backward()` would have no graph to
traverse and would produce no 2nd-derivative records.

---

## API reference

### `need_reflection(input, sds)`

```python
need_reflection(
    input: torch.Tensor,              # FutureTensor or SymbolicTensor, must be scalar
    sds: torch.Tensor,                # scalar anchor with requires_grad=True
) -> torch.Tensor                     # input with computational dependency on anchor
```

Multiplies `input` by `sds` to create a computational dependency.
During `loss.backward(create_graph=True)`, the gradient for `sds`
includes the entire model backward graph. Calling
`sds.grad.backward()` then naturally traverses that graph and
triggers the 2nd-derivative GradFn backward methods.

If `input` carries FutureTensor monkey-patched attributes, they are copied to the
result so downstream ops still see a valid FutureTensor.

`sds` must be scalar (`shape == ()`) and have `requires_grad=True`. Use:

```python
sds = torch.ones((), dtype=torch.bfloat16, requires_grad=True)
```

### `get_backward_dispatcher(fn)`

```python
from experience.future_tensor.backward_dispatch import get_backward_dispatcher

dispatch = get_backward_dispatcher(recurrent_backward)  # pass the backward fn object
dispatch(arg_name2inputs)                                # fires the active policy
```

Called inside each backward function. Receives the backward function object as the key.
Looks up the active `dispatch_policy`. Returns `True` if a policy handled the dispatch,
`False` if no policy is active (caller runs default backward).

Every dispatched op returns a **placeholder scalar tensor** (`value=1`)
rather than a meaningful tensor — consistent with FutureTensor being 0D. The actual
derivative content is recorded by the policy.

### `dispatch_policy(policy)` — context manager

```python
with dispatch_policy(policy):
    ...
```

Sets the thread-local active policy for all `get_backward_dispatcher` calls inside the block.
Used for both 1st and 2nd derivative tracing.

`policy` is a `Policy` instance — e.g. `TracePolicy(collector)`, or a custom subclass.
When no `dispatch_policy` block is active, dispatch is a no-op (returns `False`).

Policies are **not** reentrant. Nesting two `dispatch_policy` blocks raises
`PolicyConflictError`.

---

## Dispatch policies

### `TracePolicy(collector: list)`

Non-destructive. Records every dispatch call into `collector` without running any LLM.
Each record is a `ReflectionRecord`:

```python
@dataclass
class ReflectionRecord:
    fn: Callable                   # the backward function object
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
    sds.grad.backward()

for r in records:
    print(r.fn.__name__, list(r.inputs.keys()))
```

### Custom policy

Subclass `Policy` and implement `dispatch`:

```python
from experience.future_tensor.backward_dispatch.policy import Policy, ReflectionRecord

class ExecutePolicy(Policy):
    """Run an LLM to reflect on each reflection."""

    def dispatch(self, fn: Callable, arg_name2inputs: dict) -> torch.Tensor:
        # call LLM with arg_name2inputs["grad_output"] (the 1st-derivative text)
        # and the original forward inputs to produce the 2nd derivative
        ...
        return torch.ones(())

with dispatch_policy(ExecutePolicy()):
    sds.grad.backward()
```

---

## How 2nd-derivative GradFns are structured

Each FutureTensor op whose 1st derivative participates in 2nd differentiation wraps
that 1st derivative inside a `torch.autograd.Function` (a *GradFn*). The GradFn's
`forward` IS the 1st derivative; its `backward` dispatches to the active Policy.

```python
# future_tensor/function/recurrent_2nd.py

import torch
from experience.future_tensor.backward_dispatch.backward_dispatcher import get_backward_dispatcher

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
        dispatch = get_backward_dispatcher(ctx._recurrent_backward_fn)

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
backward_dispatch/
├── README.md
├── __init__.py              # exports: need_reflection, get_backward_dispatcher,
│                            #          dispatch_policy, TracePolicy, Policy,
│                            #          ReflectionRecord, PolicyConflictError
├── policy.py                # Policy base class, ReflectionRecord, PolicyConflictError
├── context.py               # dispatch_policy context manager + thread-local state
├── backward_dispatcher.py   # get_backward_dispatcher; checks active policy
├── need_reflection.py       # need_reflection: computational dependency anchor
└── trace_policy.py          # TracePolicy: non-destructive collector

future_tensor/function/      # GradFn wrappers live next to their 1st-derivative ops
├── recurrent_2nd.py         # RecurrentGradFn  (wraps recurrent_backward)
├── expert_2nd.py            # ExpertGradFn     (wraps st_moe_backward)
├── slice_2nd.py             # SliceGradFn      (wraps slice_backward)
├── unsqueeze_2nd.py         # UnsqueezeGradFn  (wraps unsqueeze squeeze-via-slice)
├── sequential_2nd.py        # SequentialGradFn (wraps sequential_backward)
├── tmux_create_session_2nd.py  # TmuxCreateSessionGradFn
├── tmux_send_text_2nd.py    # TmuxSendTextGradFn
├── tmux_send_ctrl_2nd.py    # TmuxSendCtrlGradFn
├── tmux_capture_pane_2nd.py # TmuxCapturePaneGradFn
├── read_file_2nd.py         # ReadFileGradFn
├── sleep_2nd.py             # SleepGradFn
├── switch_2nd.py            # SwitchGradFn
└── expand_2nd.py            # ExpandGradFn
```

---

## Design notes

**Why a scalar anchor?**
`sds` has shape `()` — matching FutureTensor's scalar shape. This
keeps the 2nd derivative graph homogeneous: every node is a scalar, every edge carries
a scalar gradient. The "value" of the gradient is not a float but a `SymbolicTensor`
element (a text diff), recorded by the policy.

**Why the backward function object as the dispatcher key?**
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

**`need_reflection` creates a computational dependency.**
It multiplies `input * sds` so that during `loss.backward(create_graph=True)` the
gradient computation for `sds` includes the entire backward graph of the model.
FutureTensor monkey-patched attributes are copied to the result so downstream
ops still see a valid FutureTensor. `need_reflection` does not know about
policies or dispatchers.

**One `dispatch_policy` for both derivatives.**
1st and 2nd derivative tracing are never active simultaneously. During
`loss.backward(create_graph=True)` only 1st-derivative ops fire; during
`sds.grad.backward()` only 2nd-derivative GradFn backwards fire. One thread-local
slot, one context manager — no need to distinguish.
