import torch
from typing import Callable, List
from experience.symbolic_tensor.tensor_util.sparse_to_dense import (
    sparse_to_dense,
    _sparse_to_dense_impl,
)
from experience.symbolic_tensor.tensor_util.dense_to_sparse import (
    dense_to_sparse,
    _dense_to_sparse_impl,
)
from experience.symbolic_tensor.tensor_util.todo_tensor_like import todo_tensor_like
from experience.symbolic_tensor.function import symbolic_grad_registry
class WithDenseViewFunction(torch.autograd.Function):
    """Autograd Function for sparse→dense→handler→dense→sparse.
    Wraps the full with_dense_view pipeline so gradients flow through
    the sparse↔dense boundaries. Uses torch.enable_grad() inside forward
    to build an inner autograd graph for the dense_handler.
    """
    @staticmethod
    def forward(ctx, input):
        dense_handler = WithDenseViewFunction._dense_handler_ref
        sparse_data, in_indexes, in_shape = dense_to_sparse(input, view=True)
        sparse_data.requires_grad_(True)
        with torch.enable_grad():
            dense = sparse_to_dense(sparse_data, in_indexes, in_shape)
            result_dense = dense_handler(dense)
        result_sparse, out_indexes, out_shape = _dense_to_sparse_impl(
            result_dense, view=False,
        )
        ctx.sparse_data = sparse_data
        ctx.result_dense = result_dense
        ctx.in_indexes = in_indexes
        ctx.in_shape = in_shape
        ctx.out_indexes = out_indexes
        ctx.out_shape = out_shape
        ctx.save_for_backward(input)
        ctx.st_attrs = {}
        for name, tensor in [
            ("input", input),
            ("sparse_data", sparse_data),
            ("result_dense", result_dense),
            ("result_sparse", result_sparse),
        ]:
            attrs = {}
            for attr in ("st_relative_to", "st_tensor_uid"):
                if hasattr(tensor, attr):
                    attrs[attr] = getattr(tensor, attr)
            ctx.st_attrs[name] = attrs
        return result_sparse
    @staticmethod
    def backward(ctx, grad_sparse):
        input_tensor = ctx.saved_tensors[0]
        sparse_data = ctx.sparse_data
        result_dense = ctx.result_dense
        out_indexes = ctx.out_indexes
        out_shape = ctx.out_shape
        in_indexes = ctx.in_indexes
        in_shape = ctx.in_shape
        for attr, val in ctx.st_attrs["input"].items():
            setattr(input_tensor, attr, val)
        for attr, val in ctx.st_attrs["sparse_data"].items():
            setattr(sparse_data, attr, val)
        for attr, val in ctx.st_attrs["result_dense"].items():
            setattr(result_dense, attr, val)
        sparse_uid = ctx.st_attrs["result_sparse"].get("st_tensor_uid")
        if sparse_uid:
            symbolic_grad = symbolic_grad_registry.pop(sparse_uid)
            if symbolic_grad is not None:
                grad_sparse = symbolic_grad
        if hasattr(grad_sparse, "st_relative_to") and out_indexes[0].numel() > 0:
            grad_dense = _sparse_to_dense_impl(grad_sparse, out_indexes, out_shape)
        else:
            grad_dense = todo_tensor_like(result_dense)
            if out_indexes[0].numel() > 0:
                grad_dense.data[tuple(out_indexes)] = grad_sparse.data
        result_dense_uid = ctx.st_attrs["result_dense"].get("st_tensor_uid")
        if result_dense_uid and hasattr(grad_dense, "st_relative_to"):
            symbolic_grad_registry.register(result_dense_uid, grad_dense)
        torch.autograd.backward(result_dense, grad_dense.data)
        grad_sparse_data = sparse_data.grad
        if grad_sparse_data is not None and in_indexes[0].numel() > 0:
            sparse_data_uid = ctx.st_attrs["sparse_data"].get("st_tensor_uid")
            symbolic_grad_sparse = None
            if sparse_data_uid:
                symbolic_grad_sparse = symbolic_grad_registry.pop(sparse_data_uid)
            if symbolic_grad_sparse is not None and hasattr(symbolic_grad_sparse, "st_relative_to"):
                grad_input = _sparse_to_dense_impl(
                    symbolic_grad_sparse, in_indexes, in_shape,
                )
            else:
                grad_input = todo_tensor_like(input_tensor)
                grad_input.data[tuple(in_indexes)] = grad_sparse_data.data
            input_uid = ctx.st_attrs["input"].get("st_tensor_uid")
            if input_uid and hasattr(grad_input, "st_relative_to"):
                symbolic_grad_registry.register(input_uid, grad_input)
        else:
            grad_input = None
        ctx.sparse_data = None
        ctx.result_dense = None
        return grad_input
def with_dense_view(
    dense_handler: Callable[[torch.Tensor], torch.Tensor],
    input: torch.Tensor,
) -> torch.Tensor:
    """Apply a dense_handler on a sparse symbolic tensor via dense view.
    Converts sparse input to dense, applies dense_handler, converts back.
    Args:
        dense_handler: Callable that takes a dense symbolic tensor and returns
            a dense symbolic tensor (same or different shape).
        input: A sparse symbolic tensor (1D with associated indexes/shape).
    Returns:
        Sparse symbolic tensor after dense_handler transformation.
    """
    WithDenseViewFunction._dense_handler_ref = dense_handler
    return WithDenseViewFunction.apply(input)
if __name__ == "__main__":
    import os
    import tempfile
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
    from experience.symbolic_tensor.tensor_util.dense_to_sparse import dense_to_sparse as d2s
    print("Running with_dense_view tests...\n")
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
    print("Test 1: Identity handler roundtrip")
    with tempfile.TemporaryDirectory() as tmpdir:
        original = make_tensor(["hello", "world", "foo"], tmpdir)
        original.data[1] = 0.0  # zero out "world"
        result = with_dense_view(lambda x: x, original)
        run_test("result has 2 elements", result.numel() == 2)
        run_test("result[0] == 'hello'", read_storage(result, 0) == "hello")
        run_test("result[1] == 'foo'", read_storage(result, 1) == "foo")
    print("Test 2: Handler zeros out element")
    with tempfile.TemporaryDirectory() as tmpdir:
        original = make_tensor(["a", "b", "c"], tmpdir)
        def zero_middle(dense):
            dense.data[1] = 0.0
            return dense
        result = with_dense_view(zero_middle, original)
        run_test("result has 2 elements", result.numel() == 2)
        run_test("result[0] == 'a'", read_storage(result, 0) == "a")
        run_test("result[1] == 'c'", read_storage(result, 1) == "c")
    print("Test 3: 2D identity roundtrip")
    with tempfile.TemporaryDirectory() as tmpdir:
        original = make_tensor([["x", "y"], ["z", "w"]], tmpdir)
        original.data[0, 1] = 0.0  # zero out "y"
        result = with_dense_view(lambda x: x, original)
        run_test("result has 3 elements", result.numel() == 3)
    print("Test 4: All-nonzero identity")
    with tempfile.TemporaryDirectory() as tmpdir:
        original = make_tensor(["p", "q"], tmpdir)
        result = with_dense_view(lambda x: x, original)
        run_test("result has 2 elements", result.numel() == 2)
        run_test("result[0] == 'p'", read_storage(result, 0) == "p")
        run_test("result[1] == 'q'", read_storage(result, 1) == "q")
    print("Test 5: Content preserved through dense view")
    with tempfile.TemporaryDirectory() as tmpdir:
        original = make_tensor(["a", "b"], tmpdir)
        original.data[0] = 0.0  # zero out "a", only "b" is nonzero
        result = with_dense_view(lambda x: x, original)
        run_test("result has 1 element", result.numel() == 1)
        run_test("result[0] == 'b'", read_storage(result, 0) == "b")
    print("\nAll tests completed.")