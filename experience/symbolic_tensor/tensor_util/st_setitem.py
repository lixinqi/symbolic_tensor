"""
st_setitem :=
    void
    <- $tensor SymbolicTensor
    <- $coordinates list[int]
    <- $content str
    <- $coefficient float # default 1.0
    # inline
    <- { set content into file coordinates }
    <- { tensor[coordinates] = coefficient }
"""

import os
from typing import List

import torch


def _coords_to_flat(coordinates: List[int], shape: List[int]) -> int:
    flat = 0
    stride = 1
    for i in reversed(range(len(shape))):
        flat += coordinates[i] * stride
        stride *= shape[i]
    return flat


def st_setitem(
    tensor: torch.Tensor,
    coordinates: List[int],
    content: str,
    coefficient: float = 1.0,
) -> None:
    """Write content and coefficient at the given coordinates of a SymbolicTensor.

    Args:
        tensor: A SymbolicTensor with st_relative_to and st_tensor_uid.
        coordinates: Multi-dimensional index.
        content: String content to write to the storage file.
        coefficient: Numerical value to set in tensor.data at that position.
    """
    shape = list(tensor.shape)
    flat_index = _coords_to_flat(coordinates, shape)
    digits = list(str(flat_index))

    file_path = os.path.join(
        tensor.st_relative_to,
        tensor.st_tensor_uid,
        "storage",
        os.path.join(*digits),
        "data",
    )
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    tensor.data.flatten()[flat_index] = coefficient


if __name__ == "__main__":
    import tempfile
    from experience.symbolic_tensor.tensor_util.make_none_tensor import make_none_tensor

    print("Running tests for st_setitem...\n")

    def run_test(name, cond, expected=None, actual=None):
        if cond:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")
            if expected is not None:
                print(f"    expected: {expected}, actual: {actual}")

    def read_storage(tensor, flat_index):
        digits = list(str(flat_index))
        path = os.path.join(
            tensor.st_relative_to, tensor.st_tensor_uid,
            "storage", os.path.join(*digits), "data",
        )
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return f.read()

    with tempfile.TemporaryDirectory() as tmpdir:
        t = make_none_tensor([3], tmpdir)
        st_setitem(t, [1], "hello", 0.9)
        run_test("content written", read_storage(t, 1) == "hello")
        run_test("coefficient set", abs(t.data.flatten()[1].item() - 0.9) < 0.01)
        run_test("other elem unchanged", t.data.flatten()[0].item() == 0.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        t = make_none_tensor([2, 3], tmpdir)
        st_setitem(t, [1, 2], "world", 0.7)
        run_test("2D content", read_storage(t, 5) == "world")
        run_test("2D coefficient", abs(t.data.flatten()[5].item() - 0.7) < 0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        t = make_none_tensor([4], tmpdir)
        st_setitem(t, [0], "first")
        run_test("default coeff 1.0", abs(t.data.flatten()[0].item() - 1.0) < 0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        t = make_none_tensor([2], tmpdir)
        st_setitem(t, [0], "v1", 0.5)
        st_setitem(t, [0], "v2", 0.8)
        run_test("overwrite content", read_storage(t, 0) == "v2")
        run_test("overwrite coeff", abs(t.data.flatten()[0].item() - 0.8) < 0.01)

    print("\nAll tests completed.")
