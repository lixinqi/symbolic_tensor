import os
import itertools
import torch
from typing import List, Optional
from experience.symbolic_tensor.tensor_util.make_none_tensor import make_none_tensor
from experience.symbolic_tensor.tensor_util.todo_tensor_like import todo_tensor_like
from experience.symbolic_tensor.tensor_util.st_patched import st_patched
from experience.symbolic_tensor.tensor_util.get_diff_tensor import get_diff_tensor
from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor
from experience.symbolic_tensor.tensor_util.slice_view import slice_view
from experience.symbolic_tensor.function import symbolic_grad_registry
def _get_storage_path(tensor: torch.Tensor, flat_index: int) -> str:
    digits = list(str(flat_index))
    return os.path.join(
        tensor.st_relative_to,
        tensor.st_tensor_uid,
        "storage",
        os.path.join(*digits),
        "data",
    )
def _read_storage(tensor: torch.Tensor, flat_index: int) -> Optional[str]:
    path = os.path.realpath(_get_storage_path(tensor, flat_index))
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return f.read()
def _write_storage(tensor: torch.Tensor, flat_index: int, content: str) -> None:
    path = _get_storage_path(tensor, flat_index)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
def _coords_to_flat(coords: List[int], strides: tuple) -> int:
    return sum(c * s for c, s in zip(coords, strides))
def st_stack_forward(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """Stack symbolic tensors along a new axis, like torch.stack.
    All input tensors must have the same shape. A new dimension of size n
    is inserted at position dim.
    Args:
        tensors: List of symbolic tensors with identical shapes.
        dim: Dimension at which to insert the new axis (default 0).
    Returns:
        Symbolic tensor of shape (*input_shape[:dim], n, *input_shape[dim:]).
    """
    n = len(tensors)
    assert n > 0, "st_stack requires at least one tensor"
    shape = list(tensors[0].shape)
    ndim = len(shape)
    for i, t in enumerate(tensors):
        assert list(t.shape) == shape, (
            f"Shape mismatch: tensor[0] {shape} != tensor[{i}] {list(t.shape)}"
        )
    if dim < 0:
        dim = ndim + 1 + dim
    assert 0 <= dim <= ndim, f"dim {dim} out of range for {ndim}D input"
    output_shape = shape[:dim] + [n] + shape[dim:]
    output = make_none_tensor(output_shape, tensors[0].st_relative_to)
    if shape:
        input_ranges = [range(s) for s in shape]
    else:
        input_ranges = []
    input_strides = tensors[0].stride()
    output_strides = output.stride()
    for in_coords in itertools.product(*input_ranges) if input_ranges else [()]:
        in_coords = list(in_coords)
        in_flat = _coords_to_flat(in_coords, input_strides) if in_coords else 0
        for k in range(n):
            out_coords = in_coords[:dim] + [k] + in_coords[dim:]
            out_flat = _coords_to_flat(out_coords, output_strides)
            content = _read_storage(tensors[k], in_flat)
            if content is not None:
                _write_storage(output, out_flat, content)
                coefficient = tensors[k].data.flatten()[in_flat].item()
                output.data.flatten()[out_flat] = coefficient
    return output
def st_stack_backward(
    grad_output: torch.Tensor,
    inputs: List[torch.Tensor],
    output: torch.Tensor,
    dim: int = 0,
) -> List[Optional[torch.Tensor]]:
    """Backward for st_stack: split grad along stacked axis, compute per-input diffs.
    Args:
        grad_output: Gradient w.r.t. forward output.
        inputs: Original input tensors saved from forward.
        output: Original output saved from forward.
        dim: The dimension that was inserted during forward.
    Returns:
        List of grad tensors (or None for inputs not requiring grad).
    """
    n = len(inputs)
    shape = list(inputs[0].shape)
    better_version_output = st_patched(output, grad_output)
    grad_inputs: List[Optional[torch.Tensor]] = []
    if shape:
        input_ranges = [range(s) for s in shape]
    else:
        input_ranges = []
    output_strides = output.stride()
    for k in range(n):
        if not inputs[k].requires_grad:
            grad_inputs.append(None)
            continue
        grad_input_k = todo_tensor_like(inputs[k])
        grad_input_k.data.zero_()
        input_strides_k = inputs[k].stride()
        for in_coords in itertools.product(*input_ranges) if input_ranges else [()]:
            in_coords = list(in_coords)
            out_coords = in_coords[:dim] + [k] + in_coords[dim:]
            out_flat = _coords_to_flat(out_coords, output_strides)
            orig_content = _read_storage(output, out_flat)
            improved_content = _read_storage(better_version_output, out_flat)
            if orig_content is None or improved_content is None:
                continue
            orig_view = slice_view(output, out_coords)
            improved_view = slice_view(better_version_output, out_coords)
            diff = get_diff_tensor(orig_view, improved_view)
            grad_view = slice_view(grad_input_k, in_coords)
            assign_tensor(grad_view, diff)
            in_flat = _coords_to_flat(in_coords, input_strides_k) if in_coords else 0
            improved_coeff = better_version_output.data.flatten()[out_flat].item()
            grad_input_k.data.flatten()[in_flat] = improved_coeff
        grad_inputs.append(grad_input_k)
    return grad_inputs
class StStack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dim: int, *tensors: torch.Tensor) -> torch.Tensor:
        output = st_stack_forward(list(tensors), dim)
        ctx.save_for_backward(*tensors, output)
        ctx.n_inputs = len(tensors)
        ctx.dim = dim
        ctx.st_attrs = {}
        for i, t in enumerate(tensors):
            attrs = {}
            for attr in ("st_relative_to", "st_tensor_uid"):
                if hasattr(t, attr):
                    attrs[attr] = getattr(t, attr)
            ctx.st_attrs[f"input_{i}"] = attrs
        out_attrs = {}
        for attr in ("st_relative_to", "st_tensor_uid"):
            if hasattr(output, attr):
                out_attrs[attr] = getattr(output, attr)
        ctx.st_attrs["output"] = out_attrs
        return output
    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        n = ctx.n_inputs
        inputs = list(saved[:n])
        output = saved[n]
        for i, t in enumerate(inputs):
            for attr, val in ctx.st_attrs[f"input_{i}"].items():
                setattr(t, attr, val)
        for attr, val in ctx.st_attrs["output"].items():
            setattr(output, attr, val)
        symbolic_grad = symbolic_grad_registry.pop(output.st_tensor_uid)
        if symbolic_grad is not None:
            grad_output = symbolic_grad
        elif not hasattr(grad_output, "st_relative_to"):
            symbolic_grad_output = todo_tensor_like(output)
            symbolic_grad_output.data.copy_(grad_output.data)
            grad_output = symbolic_grad_output
        grad_inputs = st_stack_backward(grad_output, inputs, output, ctx.dim)
        for i, gi in enumerate(grad_inputs):
            if gi is not None:
                symbolic_grad_registry.register(inputs[i].st_tensor_uid, gi)
        return (None,) + tuple(grad_inputs)
def st_stack(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """Stack symbolic tensors along a new axis, like torch.stack.
    Args:
        tensors: List of symbolic tensors with identical shapes.
        dim: Dimension at which to insert the new axis (default 0).
    Returns:
        Symbolic tensor with a new dimension of size len(tensors) at dim.
    """
    return StStack.apply(dim, *tensors)
if __name__ == "__main__":
    import tempfile
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
    print("Running st_stack tests...\n")
    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"  \u2713 {name}")
        else:
            print(f"  \u2717 {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")
    def read_out(tensor, flat_index):
        path = _get_storage_path(tensor, flat_index)
        path = os.path.realpath(path)
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return f.read()
    print("Test 1: dim=0, two 1D (3,) → (2, 3)")
    with tempfile.TemporaryDirectory() as tmpdir:
        a = make_tensor(["a0", "a1", "a2"], tmpdir)
        b = make_tensor(["b0", "b1", "b2"], tmpdir)
        output = st_stack_forward([a, b], dim=0)
        run_test("shape [2, 3]", list(output.shape) == [2, 3])
        run_test("[0,0]='a0'", read_out(output, 0) == "a0")
        run_test("[0,1]='a1'", read_out(output, 1) == "a1")
        run_test("[0,2]='a2'", read_out(output, 2) == "a2")
        run_test("[1,0]='b0'", read_out(output, 3) == "b0")
        run_test("[1,1]='b1'", read_out(output, 4) == "b1")
        run_test("[1,2]='b2'", read_out(output, 5) == "b2")
    print("Test 2: dim=-1, two 1D (3,) → (3, 2)")
    with tempfile.TemporaryDirectory() as tmpdir:
        a = make_tensor(["a0", "a1", "a2"], tmpdir)
        b = make_tensor(["b0", "b1", "b2"], tmpdir)
        output = st_stack_forward([a, b], dim=-1)
        run_test("shape [3, 2]", list(output.shape) == [3, 2])
        run_test("[0,0]='a0'", read_out(output, 0) == "a0")
        run_test("[0,1]='b0'", read_out(output, 1) == "b0")
        run_test("[1,0]='a1'", read_out(output, 2) == "a1")
        run_test("[1,1]='b1'", read_out(output, 3) == "b1")
        run_test("[2,0]='a2'", read_out(output, 4) == "a2")
        run_test("[2,1]='b2'", read_out(output, 5) == "b2")
    print("Test 3: dim=1, two 2D (2,3) → (2, 2, 3)")
    with tempfile.TemporaryDirectory() as tmpdir:
        a = make_tensor([["a00", "a01", "a02"], ["a10", "a11", "a12"]], tmpdir)
        b = make_tensor([["b00", "b01", "b02"], ["b10", "b11", "b12"]], tmpdir)
        output = st_stack_forward([a, b], dim=1)
        run_test("shape [2, 2, 3]", list(output.shape) == [2, 2, 3])
        run_test("[0,0,0]='a00'", read_out(output, 0) == "a00")
        run_test("[0,0,1]='a01'", read_out(output, 1) == "a01")
        run_test("[0,0,2]='a02'", read_out(output, 2) == "a02")
        run_test("[0,1,0]='b00'", read_out(output, 3) == "b00")
        run_test("[0,1,1]='b01'", read_out(output, 4) == "b01")
        run_test("[0,1,2]='b02'", read_out(output, 5) == "b02")
        run_test("[1,0,0]='a10'", read_out(output, 6) == "a10")
        run_test("[1,1,0]='b10'", read_out(output, 9) == "b10")
    print("Test 4: dim=-1, two 2D (2,3) → (2, 3, 2)")
    with tempfile.TemporaryDirectory() as tmpdir:
        a = make_tensor([["a00", "a01", "a02"], ["a10", "a11", "a12"]], tmpdir)
        b = make_tensor([["b00", "b01", "b02"], ["b10", "b11", "b12"]], tmpdir)
        output = st_stack_forward([a, b], dim=-1)
        run_test("shape [2, 3, 2]", list(output.shape) == [2, 3, 2])
        run_test("[0,0,0]='a00'", read_out(output, 0) == "a00")
        run_test("[0,0,1]='b00'", read_out(output, 1) == "b00")
        run_test("[1,2,0]='a12'", read_out(output, 10) == "a12")
        run_test("[1,2,1]='b12'", read_out(output, 11) == "b12")
    print("Test 5: dim=0, three 1D (2,) → (3, 2)")
    with tempfile.TemporaryDirectory() as tmpdir:
        a = make_tensor(["x", "y"], tmpdir)
        b = make_tensor(["p", "q"], tmpdir)
        c = make_tensor(["m", "n"], tmpdir)
        output = st_stack_forward([a, b, c], dim=0)
        run_test("shape [3, 2]", list(output.shape) == [3, 2])
        run_test("[0,0]='x'", read_out(output, 0) == "x")
        run_test("[0,1]='y'", read_out(output, 1) == "y")
        run_test("[1,0]='p'", read_out(output, 2) == "p")
        run_test("[2,0]='m'", read_out(output, 4) == "m")
    print("Test 6: Scalar tensors dim=0 → (2,)")
    with tempfile.TemporaryDirectory() as tmpdir:
        a = make_tensor("hello", tmpdir)
        b = make_tensor("world", tmpdir)
        output = st_stack_forward([a, b], dim=0)
        run_test("shape [2]", list(output.shape) == [2])
        run_test("[0]='hello'", read_out(output, 0) == "hello")
        run_test("[1]='world'", read_out(output, 1) == "world")
    print("Test 7: Coefficient preservation")
    with tempfile.TemporaryDirectory() as tmpdir:
        a = make_tensor(["x", "y"], tmpdir)
        a.data[0] = 0.5
        a.data[1] = 0.8
        b = make_tensor(["p", "q"], tmpdir)
        b.data[0] = 0.3
        b.data[1] = 0.9
        output = st_stack_forward([a, b], dim=0)
        run_test("coeff[0,0]=0.5", abs(output.data[0, 0].item() - 0.5) < 0.01)
        run_test("coeff[0,1]=0.8", abs(output.data[0, 1].item() - 0.8) < 0.01)
        run_test("coeff[1,0]=0.3", abs(output.data[1, 0].item() - 0.3) < 0.01)
        run_test("coeff[1,1]=0.9", abs(output.data[1, 1].item() - 0.9) < 0.01)
    print("Test 8: Shape mismatch")
    with tempfile.TemporaryDirectory() as tmpdir:
        a = make_tensor(["x", "y"], tmpdir)
        b = make_tensor(["p", "q", "r"], tmpdir)
        try:
            st_stack_forward([a, b])
            run_test("assertion raised", False)
        except AssertionError:
            run_test("assertion raised", True)
    print("Test 9: st_stack wrapper dim=0")
    with tempfile.TemporaryDirectory() as tmpdir:
        a = make_tensor(["hello\n", "world\n"], tmpdir)
        a.requires_grad_(True)
        b = make_tensor(["foo\n", "bar\n"], tmpdir)
        b.requires_grad_(True)
        output = st_stack([a, b], dim=0)
        run_test("shape [2, 2]", list(output.shape) == [2, 2])
        run_test("has st_relative_to", hasattr(output, "st_relative_to"))
        run_test("[0,0]='hello\\n'", read_out(output, 0) == "hello\n")
        run_test("[0,1]='world\\n'", read_out(output, 1) == "world\n")
        run_test("[1,0]='foo\\n'", read_out(output, 2) == "foo\n")
        run_test("[1,1]='bar\\n'", read_out(output, 3) == "bar\n")
    print("Test 10: st_stack wrapper dim=-1")
    with tempfile.TemporaryDirectory() as tmpdir:
        a = make_tensor(["hello\n", "world\n"], tmpdir)
        b = make_tensor(["foo\n", "bar\n"], tmpdir)
        output = st_stack([a, b], dim=-1)
        run_test("shape [2, 2]", list(output.shape) == [2, 2])
        run_test("[0,0]='hello\\n'", read_out(output, 0) == "hello\n")
        run_test("[0,1]='foo\\n'", read_out(output, 1) == "foo\n")
        run_test("[1,0]='world\\n'", read_out(output, 2) == "world\n")
        run_test("[1,1]='bar\\n'", read_out(output, 3) == "bar\n")
    print("Test 11: Backward content diff (dim=0)")
    with tempfile.TemporaryDirectory() as tmpdir:
        a = make_tensor(["alpha\n", "beta\n"], tmpdir)
        a.requires_grad_(True)
        b = make_tensor(["gamma\n", "delta\n"], tmpdir)
        b.requires_grad_(True)
        output = st_stack_forward([a, b], dim=0)
        out_00 = read_out(output, 0)
        modified_00 = out_00.replace("alpha", "ALPHA")
        modified_data = [
            [modified_00, read_out(output, 1)],
            [read_out(output, 2), read_out(output, 3)],
        ]
        modified_output = make_tensor(modified_data, tmpdir)
        grad_out = get_diff_tensor(output, modified_output)
        grad_inputs = st_stack_backward(grad_out, [a, b], output, dim=0)
        run_test("grad_a not None", grad_inputs[0] is not None)
        run_test("grad_b not None", grad_inputs[1] is not None)
        run_test("grad_a shape [2]", list(grad_inputs[0].shape) == [2])
        gi_a0 = read_out(grad_inputs[0], 0)
        run_test("grad_a[0] has diff", gi_a0 is not None and "---" in gi_a0)
        run_test("grad_a[0] has +ALPHA", "+ALPHA" in gi_a0 if gi_a0 else False)
        gi_a1 = read_out(grad_inputs[0], 1)
        run_test("grad_a[1] empty diff", gi_a1 is not None and "---" not in gi_a1)
        gi_b0 = read_out(grad_inputs[1], 0)
        run_test("grad_b[0] empty diff", gi_b0 is not None and "---" not in gi_b0)
    print("Test 12: No grad when not required")
    with tempfile.TemporaryDirectory() as tmpdir:
        a = make_tensor(["x\n"], tmpdir)
        b = make_tensor(["y\n"], tmpdir)
        b.requires_grad_(True)
        output = st_stack_forward([a, b], dim=0)
        modified_data = [[read_out(output, 0)], [read_out(output, 1)]]
        modified_output = make_tensor(modified_data, tmpdir)
        grad_out = get_diff_tensor(output, modified_output)
        grad_inputs = st_stack_backward(grad_out, [a, b], output, dim=0)
        run_test("grad_a is None", grad_inputs[0] is None)
        run_test("grad_b not None", grad_inputs[1] is not None)
    print("Test 13: Parity with torch.stack shapes")
    for dim in [-2, -1, 0, 1]:
        shape = [2, 3]
        expected = list(torch.stack([torch.zeros(shape), torch.zeros(shape)], dim=dim).shape)
        with tempfile.TemporaryDirectory() as tmpdir:
            a = make_tensor([["a", "b", "c"], ["d", "e", "f"]], tmpdir)
            b = make_tensor([["g", "h", "i"], ["j", "k", "l"]], tmpdir)
            output = st_stack_forward([a, b], dim=dim)
            run_test(f"dim={dim}: {expected}", list(output.shape) == expected)
    print("\nAll tests completed.")