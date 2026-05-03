# FutureTensor

A **lazy, async tensor abstraction** for symbolic computation over text — each element
is a file on disk, produced on-demand by an async generator (typically an LLM call).

---

## What is a FutureTensor?

A `FutureTensor` is a **scalar `torch.Tensor`** — shape `()`, dtype `bfloat16`, value
always `1` — monkey-patched with `ft_*` attributes that make it a **reference** to a
pending symbolic computation.

```python
ft = FutureTensor(relative_to, ft_async_get, ft_shape_schema)
# ft is a torch.Tensor((), bfloat16, value=1)
# ft.ft_static_tensor  -> SymbolicTensor of shape ft_capacity_shape
# ft.ft_forwarded      -> False until ft_forward() is called
```

It is **not a subclass of SymbolicTensor**. The scalar `1` carries no semantic meaning
beyond "this reference exists." All status and coefficient semantics live in
`ft_static_tensor` (a `SymbolicTensor`).

---

## Interface

```
FutureTensor :=
    torch.Tensor[(), bfloat16, value=1]
    * ft_static_tensor          SymbolicTensor[...]
    * ft_incremental_concated_tensors  list[(SymbolicTensor, concat_axis: int)]
    * ft_shape_schema           list[sympy.Symbol | sympy.Integer]
    * ft_capacity_shape         list[int]
    * ft_forwarded              bool
    * ft_async_get              (coordinates: list[int], prompt: str) -> Awaitable[(str, Status)]
    * ft_forward                (prompt_tensor: SymbolicTensor) -> void
    * ft_get_materialized_value (coordinates: list[int]) -> (coefficient: float, filepath: str)
    * ft_reset_materialized_value (coordinates, coefficient, filepath, symlink=False) -> void
```

### Fields

| Field | Type | Description |
|---|---|---|
| `ft_static_tensor` | `SymbolicTensor[...]` | Base referent. Always valid — initialized to `make_none_tensor` at construction; populated by `ft_forward`. Holds Status floats as coefficients. |
| `ft_incremental_concated_tensors` | `list[(SymbolicTensor, int)]` | Tensors appended in dynamic cases. Logical view = `concat(ft_static_tensor, *ft_incremental_concated_tensors)`. Empty in the static case. |
| `ft_shape_schema` | `list[sympy.Symbol]` | Declared logical shape schema — elements may be symbolic (`sympy.Symbol`) or concrete (`sympy.Integer`). |
| `ft_capacity_shape` | `list[int]` | Concrete shape of the logical view tensor. Must match `concat(ft_static_tensor, ...).shape`. |
| `ft_forwarded` | `bool` | `True` once `ft_forward` has been called and all elements are materialized. |
| `ft_async_get` | callable | Async element generator: `(coordinates, prompt) -> (str, Status)`. |

### Constructor

```python
FutureTensor(
    relative_to: str,                              # storage root directory
    ft_async_get: Callable,                        # async (coords, prompt) -> (str, Status)
    ft_shape_schema: list[sympy.Symbol | int],     # logical shape schema
) -> torch.Tensor  # scalar bfloat16, value=1, with ft_* attrs
```

### `ft_forward(prompt_tensor)`

Materializes all elements concurrently:

1. For each coordinate in `ft_capacity_shape`, reads the prompt from `prompt_tensor`.
2. Calls `ft_async_get(coords, prompt)` for all elements via `asyncio.gather`.
3. Writes Status floats into `ft_static_tensor.data` (coefficients).
4. Writes content strings to disk storage under `ft_static_tensor`.
5. Sets `ft_forwarded = True`. Idempotent — subsequent calls are no-ops.

### `ft_get_materialized_value(coordinates)`

Returns `(coefficient: float, filepath: str)` for the element at `coordinates` in the
logical view tensor. `coefficient` is the bfloat16 value in `ft_static_tensor.data`.

### `ft_reset_materialized_value(coordinates, coefficient, filepath, symlink=False)`

Overwrites the element at `coordinates`. Copies the file (or symlinks if
`symlink=True`) and updates the coefficient in `ft_static_tensor.data`.

---

## Status encoding

`Status` is a tagged union encoded as a float in `ft_static_tensor` coefficients:

| Variant | Float encoding | Meaning |
|---|---|---|
| `confidence(v)` | `+v` (0.0–1.0) | Success — element is valid |
| `self_confidence_but_failed(v)` | `-v` (−1.0–0.0) | Failed but self-assessed confidence known |
| `kConfidenceNotBounded` | `−2.0` | Error: confidence out of range |
| `kContextOverflow` | `−3.0` | Error: context too long |

The `FutureTensor` scalar itself is always `1` — it is the `ft_static_tensor`
coefficients that carry Status.

---

## Module layout

```
future_tensor/
├── future_tensor.py          # FutureTensor factory function + helpers
├── status.py                 # Status tagged union
└── function/
    ├── slice_forward.py      # slice_forward: FutureTensor -> sliced FutureTensor
    ├── slice_backward.py     # slice_backward: scatter grad back to original shape
    ├── unsqueeze_forward.py  # unsqueeze_forward: insert dim of size 1
    ├── ft_slice.py           # FtSlice: autograd Function wrapping slice_forward/backward
    ├── ft_unsqueeze.py       # FtUnsqueeze: autograd Function wrapping unsqueeze_forward
    ├── ft_recurrent_forward.py  # recurrent_forward: retry loop encoded in last dim
    ├── ft_recurrent_backward.py # recurrent_backward: LLM reflection on failed iterations
    ├── ft_recurrent.py       # FtRecurrent: autograd Function for recurrent loop
    ├── ft_moe_forward.py     # ft_moe_forward: MoE query+translate per element
    └── ft_moe.py             # FtMoe: autograd Function for MoE with backward
```

---

## Key operations

### `ft_slice(ft, slices)` — same semantics as torch slice

```python
from experience.future_tensor.function.ft_slice import ft_slice

r = ft_slice(ft, [slice(1, 4), slice(None)])  # shape [3, N]
r = ft_slice(ft, [2])                          # int index collapses dim -> shape [N]
```

Backward scatters grad back to original positions (zeros for non-sliced positions).

### `ft_unsqueeze(ft, dim)` — same semantics as `torch.unsqueeze`

```python
from experience.future_tensor.function.ft_unsqueeze import ft_unsqueeze

r = ft_unsqueeze(ft, 0)   # insert dim at 0
r = ft_unsqueeze(ft, -1)  # insert dim at last position
```

### `ft_recurrent(ft, ...)` — generate-validate retry loop

```python
from experience.future_tensor.function.ft_recurrent import ft_recurrent

# input shape: (*prefix_dims, recurrent_dim)
# output shape: (*prefix_dims)
output, prompt_tensor = ft_recurrent(
    input_ft,
    topk_self_confidence_but_failed=8,
    task_prompt="...",
    llm_method="raw_llm_api",
    accumulate_output=None,   # or Callable[[acc, cur], acc] for context accumulation
)
```

For each prefix coordinate, iterates `i` in `range(recurrent_dim)`:
- `confidence` → return immediately (success).
- `self_confidence_but_failed` → record best, try next iteration.
- All fail → return best `scbf` result.

`accumulate_output` enables context accumulation across iterations (e.g. concatenating
clean tool-call results in the `HarnessModel` context-gather stage).

### `ft_moe(ft, experience, ...)` — Mixture-of-Experts translation

```python
from experience.future_tensor.function.ft_moe import ft_moe

output, prompt_tensor, indexes_map = ft_moe(
    input_ft, experience_tensor,
    topk=16, task_prompt="...", llm_method="raw_llm_api",
)
```

For each element: query `experience` via `select_qkv_indexes`, retrieve top-k entries,
call LLM to translate. Backward reflects on failure cases to update `experience`.

---

## Storage layout

Each `SymbolicTensor` element is a file:

```
{st_relative_to}/{st_tensor_uid}/storage/{d0}/{d1}/.../data
```

where `d0, d1, ...` are the digits of the flat index. For example, flat index `42` →
`storage/4/2/data`.

`ft_static_tensor` and the tensors in `ft_incremental_concated_tensors` share the same
`st_relative_to` root but have distinct `st_tensor_uid`s.

---

## Usage example

```python
import sympy
from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor

async def my_generator(coords, prompt):
    # call LLM or any async source
    result = await some_llm_call(prompt)
    return (result, Status.confidence(0.9))

ft = FutureTensor(
    relative_to="/tmp/mystore",
    ft_async_get=my_generator,
    ft_shape_schema=[sympy.Integer(4), sympy.Integer(3)],
)
# ft.ft_capacity_shape == [4, 3]
# ft.ft_forwarded == False

prompt_tensor = make_tensor([["p"]*3]*4, "/tmp/mystore")
ft.ft_forward(prompt_tensor)
# ft.ft_forwarded == True
# ft.ft_static_tensor has content at all 12 positions
```
