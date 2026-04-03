import os
import shutil
import torch
from typing import List
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.tensor_util.dense_to_sparse import dense_to_sparse
def _make_empty_dense(shape: List[int], relative_to: str) -> torch.Tensor:
    """Create a dense symbolic tensor with empty storage files and zero data."""
    total = 1
    for s in shape:
        total *= s
    def _reshape(flat, shp):
        if len(shp) == 1:
            return flat[:shp[0]]
        chunk = 1
        for s in shp[1:]:
            chunk *= s
        return [_reshape(flat[i * chunk:(i + 1) * chunk], shp[1:]) for i in range(shp[0])]
    nested = [""] * total
    if len(shape) == 0:
        nested_data = ""
    elif len(shape) == 1:
        nested_data = nested
    else:
        nested_data = _reshape(nested, shape)
    t = make_tensor(nested_data, relative_to)
    t.data.zero_()
    return t
def _get_storage_path(tensor: torch.Tensor, flat_index: int) -> str:
    """Get the real storage file path for a given flat index."""
    digits = list(str(flat_index))
    path = os.path.join(
        tensor.st_relative_to,
        tensor.st_tensor_uid,
        "storage",
        os.path.join(*digits),
        "data",
    )
    return os.path.realpath(path)
def _sparse_to_dense_impl(
    input: torch.Tensor,
    indexes: List[torch.Tensor],
    shape: List[int],
) -> torch.Tensor:
    """Core implementation: reconstruct dense from sparse."""
    output = _make_empty_dense(shape, input.st_relative_to)
    if indexes[0].numel() == 0:
        return output
    stride = output.stride()
    dense_flat = sum(idx * s for idx, s in zip(indexes, stride))
    for i in range(input.numel()):
        src_path = _get_storage_path(input, i)
        dst_path = _get_storage_path(output, int(dense_flat[i]))
        shutil.copy2(src_path, dst_path)
    output.data[tuple(indexes)] = input.data
    return output
class SparseToDense(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, *index_tensors_and_shape):
        shape_tensor = index_tensors_and_shape[-1]
        indexes = list(index_tensors_and_shape[:-1])
        shape = shape_tensor.tolist()
        output = _sparse_to_dense_impl(input, indexes, shape)
        ctx.save_for_backward(input, *indexes, shape_tensor)
        ctx.st_attrs = {}
        for name, tensor in [("input", input), ("output", output)]:
            attrs = {}
            for attr in ("st_relative_to", "st_tensor_uid"):
                if hasattr(tensor, attr):
                    attrs[attr] = getattr(tensor, attr)
            ctx.st_attrs[name] = attrs
        return output
    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        input_tensor = saved[0]
        shape_tensor = saved[-1]
        indexes = list(saved[1:-1])
        for attr, val in ctx.st_attrs["input"].items():
            setattr(input_tensor, attr, val)
        if hasattr(grad_output, "st_relative_to"):
            grad_input, _, _ = dense_to_sparse(grad_output, view=False)
            pass
        grad_input_data = grad_output.data[tuple(indexes)]
        if hasattr(grad_output, "st_relative_to"):
            from experience.symbolic_tensor.tensor_util.dense_to_sparse import (
                _get_storage_path as _sparse_get_path,
            )
            stride = grad_output.stride()
            dense_flat = sum(idx * s for idx, s in zip(indexes, stride))
            paths = [_sparse_get_path(grad_output, int(fi)) for fi in dense_flat]
            from pathlib import Path
            grad_sparse = make_tensor(
                [Path(p) for p in paths],
                input_tensor.st_relative_to,
                symlink=False,
            )
            grad_sparse.data = grad_input_data
        else:
            grad_sparse = grad_input_data
        return (grad_sparse,) + (None,) * len(indexes) + (None,)
def sparse_to_dense(
    input: torch.Tensor,
    indexes: List[torch.Tensor],
    shape: List[int],
) -> torch.Tensor:
    """Reconstruct a dense symbolic tensor from sparse representation.
    Wrapped as torch.autograd.Function for gradient flow.
    Forward: place sparse elements into a dense tensor at indexed positions.
    Backward: extract grad at indexed positions back to sparse form.
    Args:
        input: 1D sparse symbolic tensor containing the nonzero elements.
        indexes: list[torch.Tensor[int]], one per dimension of the target
            dense tensor, containing the coordinates where elements go.
        shape: Target dense shape as list[int].
    Returns:
        A dense symbolic tensor with the sparse elements placed at
        the given index positions. Other positions have empty content.
    """
    shape_tensor = torch.tensor(shape, dtype=torch.long)
    return SparseToDense.apply(input, *indexes, shape_tensor)
if __name__ == "__main__":
    import tempfile
    from experience.symbolic_tensor.tensor_util.dense_to_sparse import dense_to_sparse
    print("Running sparse_to_dense tests...\n")
    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"  \u2713 {name}")
        else:
            print(f"  \u2717 {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")
    def read_storage(tensor, flat_index):
        digits = list(str(flat_index))
        path = os.path.join(
            tensor.st_relative_to,
            tensor.st_tensor_uid,
            "storage",
            os.path.join(*digits),
            "data",
        )
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return f.read()
    print("Test 1: Basic reconstruction")
    with tempfile.TemporaryDirectory() as tmpdir:
        sparse = make_tensor(["a", "b"], tmpdir)
        indexes = [torch.tensor([0, 2])]
        dense = sparse_to_dense(sparse, indexes, [3])
        run_test("shape is [3]", list(dense.shape) == [3])
        run_test("dense[0] == 'a'", read_storage(dense, 0) == "a")
        run_test("dense[2] == 'b'", read_storage(dense, 2) == "b")
        run_test("dense.data[0] == 1.0", dense.data[0].item() == 1.0)
        run_test("dense.data[1] == 0.0", dense.data[1].item() == 0.0)
        run_test("dense.data[2] == 1.0", dense.data[2].item() == 1.0)
    print("Test 2: 2D reconstruction")
    with tempfile.TemporaryDirectory() as tmpdir:
        sparse = make_tensor(["x", "y"], tmpdir)
        indexes = [torch.tensor([0, 1]), torch.tensor([1, 0])]
        dense = sparse_to_dense(sparse, indexes, [2, 2])
        run_test("shape is [2, 2]", list(dense.shape) == [2, 2])
        run_test("dense[0,1] == 'x'", read_storage(dense, 1) == "x")
        run_test("dense[1,0] == 'y'", read_storage(dense, 2) == "y")
        run_test("data[0,1] nonzero", dense.data[0, 1].item() != 0.0)
        run_test("data[1,0] nonzero", dense.data[1, 0].item() != 0.0)
        run_test("data[0,0] zero", dense.data[0, 0].item() == 0.0)
        run_test("data[1,1] zero", dense.data[1, 1].item() == 0.0)
    print("Test 3: Empty sparse input")
    with tempfile.TemporaryDirectory() as tmpdir:
        sparse = make_tensor(["dummy"], tmpdir)
        indexes = [torch.tensor([], dtype=torch.long)]
        dense = sparse_to_dense(sparse, indexes, [3])
        run_test("shape is [3]", list(dense.shape) == [3])
        run_test("all zeros", dense.data.sum().item() == 0.0)
    print("Test 4: Roundtrip 1D")
    with tempfile.TemporaryDirectory() as tmpdir:
        original = make_tensor(["alpha", "beta", "gamma"], tmpdir)
        original.data[1] = 0.0
        sparse, indexes, shape = dense_to_sparse(original)
        reconstructed = sparse_to_dense(sparse, indexes, shape)
        run_test("shape matches", list(reconstructed.shape) == [3])
        run_test("reconstructed[0] == 'alpha'", read_storage(reconstructed, 0) == "alpha")
        run_test("reconstructed[2] == 'gamma'", read_storage(reconstructed, 2) == "gamma")
        run_test("data[0] nonzero", reconstructed.data[0].item() != 0.0)
        run_test("data[1] zero", reconstructed.data[1].item() == 0.0)
        run_test("data[2] nonzero", reconstructed.data[2].item() != 0.0)
    print("Test 5: Roundtrip 2D")
    with tempfile.TemporaryDirectory() as tmpdir:
        original = make_tensor([["a", "b"], ["c", "d"]], tmpdir)
        original.data[0, 1] = 0.0
        original.data[1, 0] = 0.0
        sparse, indexes, shape = dense_to_sparse(original)
        run_test("sparse has 2 elements", sparse.numel() == 2)
        reconstructed = sparse_to_dense(sparse, indexes, shape)
        run_test("shape matches", list(reconstructed.shape) == [2, 2])
        run_test("reconstructed[0,0] == 'a'", read_storage(reconstructed, 0) == "a")
        run_test("reconstructed[1,1] == 'd'", read_storage(reconstructed, 3) == "d")
        run_test("data[0,0] nonzero", reconstructed.data[0, 0].item() != 0.0)
        run_test("data[0,1] zero", reconstructed.data[0, 1].item() == 0.0)
        run_test("data[1,0] zero", reconstructed.data[1, 0].item() == 0.0)
        run_test("data[1,1] nonzero", reconstructed.data[1, 1].item() != 0.0)
    print("Test 6: Roundtrip all-nonzero")
    with tempfile.TemporaryDirectory() as tmpdir:
        original = make_tensor(["p", "q", "r"], tmpdir)
        sparse, indexes, shape = dense_to_sparse(original)
        run_test("all 3 in sparse", sparse.numel() == 3)
        reconstructed = sparse_to_dense(sparse, indexes, shape)
        run_test("reconstructed[0] == 'p'", read_storage(reconstructed, 0) == "p")
        run_test("reconstructed[1] == 'q'", read_storage(reconstructed, 1) == "q")
        run_test("reconstructed[2] == 'r'", read_storage(reconstructed, 2) == "r")
    print("Test 7: Roundtrip with view=True")
    with tempfile.TemporaryDirectory() as tmpdir:
        original = make_tensor(["m", "n"], tmpdir)
        original.data[0] = 0.0
        sparse, indexes, shape = dense_to_sparse(original, view=True)
        reconstructed = sparse_to_dense(sparse, indexes, shape)
        run_test("reconstructed[1] == 'n'", read_storage(reconstructed, 1) == "n")
        run_test("data[0] zero", reconstructed.data[0].item() == 0.0)
        run_test("data[1] nonzero", reconstructed.data[1].item() != 0.0)
    print("Test 8: Autograd integration")
    with tempfile.TemporaryDirectory() as tmpdir:
        sparse = make_tensor(["a", "b"], tmpdir)
        sparse.requires_grad_(True)
        indexes = [torch.tensor([0, 2])]
        dense = sparse_to_dense(sparse, indexes, [3])
        run_test("output requires_grad", dense.requires_grad)
        run_test("has grad_fn", dense.grad_fn is not None)
    print("\nAll tests completed.")