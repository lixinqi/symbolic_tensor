"""
expand_forward :=
    FutureTensor
    <- FutureTensor
    <- list[int]
    # inline

# the same behavior with torch expand
"""

import itertools
import os
from typing import List

import sympy
import torch

from experience.future_tensor.future_tensor import FutureTensor


def expand_forward(input: FutureTensor, target_shape: List[int]) -> FutureTensor:
    """Expand a FutureTensor -- broadcast dims of size 1 to target size.

    Same semantics as torch.Tensor.expand(). Dimensions of size 1 in the input
    are broadcast to the corresponding size in target_shape. Passing -1 for a
    dimension means keeping the current size.

    Args:
        input: Source FutureTensor.
        target_shape: Desired output shape. Must be broadcastable from input shape.

    Returns:
        A new FutureTensor with shape == target_shape.
    """
    input_shape = input.ft_capacity_shape

    # Allow target_shape to have more dims than input (prepend size-1 dims)
    ndim_diff = len(target_shape) - len(input_shape)
    if ndim_diff < 0:
        raise ValueError(
            f"expand: target_shape has fewer dims ({len(target_shape)}) "
            f"than input ({len(input_shape)})"
        )
    padded_input_shape = [1] * ndim_diff + input_shape

    # Resolve -1 and validate
    output_shape = []
    for d, (t, i) in enumerate(zip(target_shape, padded_input_shape)):
        if t == -1:
            output_shape.append(i)
        elif i == 1:
            output_shape.append(t)
        elif i == t:
            output_shape.append(t)
        else:
            raise ValueError(
                f"expand: cannot expand dim {d} from size {i} to {t} "
                f"(only size-1 dims can be expanded)"
            )

    # Build coordinate mapping: output_coords -> input_coords
    def map_coords(out_coords: List[int]) -> List[int]:
        in_coords = []
        for d in range(ndim_diff, len(output_shape)):
            in_d = d - ndim_diff
            if input_shape[in_d] == 1:
                in_coords.append(0)
            else:
                in_coords.append(out_coords[d])
        return in_coords

    async def expanded_async_get(coordinates: List[int], prompt: str):
        original_coords = map_coords(coordinates)
        return await input.ft_async_get(original_coords, prompt)

    result = FutureTensor(
        input.ft_static_tensor.st_relative_to,
        expanded_async_get,
        [sympy.Integer(s) for s in output_shape],
    )

    # If input is already forwarded, copy storage directly
    if input.ft_forwarded:
        _copy_expanded_storage(input, result, output_shape, padded_input_shape, ndim_diff)
        result.ft_forwarded = True

    return result


def _copy_expanded_storage(input, output, output_shape, padded_input_shape, ndim_diff):
    """Copy element storage from input to output, broadcasting size-1 dims."""
    import shutil

    input_shape = input.ft_capacity_shape

    for out_coords in itertools.product(*[range(s) for s in output_shape]) if output_shape else [()]:
        out_coords = list(out_coords) if output_shape else []

        # Map to input coords
        in_coords = []
        for d in range(ndim_diff, len(output_shape)):
            in_d = d - ndim_diff
            if input_shape[in_d] == 1:
                in_coords.append(0)
            else:
                in_coords.append(out_coords[d])

        in_flat = _coords_to_flat(in_coords, input_shape) if input_shape else 0
        out_flat = _coords_to_flat(out_coords, output_shape)

        src_path = _storage_path(input, in_flat)
        dst_path = _storage_path(output, out_flat)
        if os.path.isfile(src_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)
            output.ft_static_tensor.data.flatten()[out_flat] = (
                input.ft_static_tensor.data.flatten()[in_flat]
            )


def _coords_to_flat(coordinates: List[int], shape: List[int]) -> int:
    flat = 0
    stride = 1
    for i in reversed(range(len(shape))):
        flat += coordinates[i] * stride
        stride *= shape[i]
    return flat


def _storage_path(ft: FutureTensor, flat_index: int) -> str:
    digits = list(str(flat_index))
    return os.path.join(
        ft.ft_static_tensor.st_relative_to, ft.ft_static_tensor.st_tensor_uid,
        "storage", os.path.join(*digits), "data",
    )


if __name__ == "__main__":
    import tempfile

    from experience.future_tensor.status import Status
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor as st_make_tensor
    from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor

    print("Running tests for expand_forward...\n")

    passed = 0
    failed = 0

    def run_test(name: str, condition: bool, expected=None, actual=None):
        global passed, failed
        if condition:
            passed += 1
        else:
            failed += 1
            print(f"  \u2717 {name}")
            if expected is not None:
                print(f"    expected: {expected}, actual: {actual}")

    def read_ft_element(ft, flat_index):
        path = _storage_path(ft, flat_index)
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return f.read()

    def make_forwarded_ft(shape, data_list, tmpdir):
        async def dummy_get(coords, prompt):
            return ("unused", Status.confidence(1.0))
        ft = FutureTensor(tmpdir, dummy_get, [sympy.Integer(s) for s in shape])
        nested = _unflatten_data(data_list, shape)
        result_tensor = st_make_tensor(nested, tmpdir)
        assign_tensor(ft.ft_static_tensor, result_tensor)
        ft.ft_forwarded = True
        return ft

    def _unflatten_data(flat_list, shape):
        if not shape:
            return flat_list[0] if flat_list else None
        if len(shape) == 1:
            return flat_list
        chunk_size = 1
        for s in shape[1:]:
            chunk_size *= s
        return [
            _unflatten_data(flat_list[i * chunk_size:(i + 1) * chunk_size], shape[1:])
            for i in range(shape[0])
        ]

    # === Group 1: Basic expand (1D -> broadcast) ===
    print("Group 1: Basic expand")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([1, 3], ["a", "b", "c"], tmpdir)
        r = expand_forward(ft, [4, 3])
        run_test("expand [1,3]->[4,3] shape", r.ft_capacity_shape == [4, 3])
        run_test("expand forwarded", r.ft_forwarded is True)
        run_test("[0,0]=a", read_ft_element(r, 0) == "a")
        run_test("[0,2]=c", read_ft_element(r, 2) == "c")
        run_test("[3,0]=a (broadcast)", read_ft_element(r, 9) == "a")
        run_test("[3,2]=c (broadcast)", read_ft_element(r, 11) == "c")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([3, 1], ["x", "y", "z"], tmpdir)
        r = expand_forward(ft, [3, 5])
        run_test("expand [3,1]->[3,5] shape", r.ft_capacity_shape == [3, 5])
        run_test("[0,0]=x", read_ft_element(r, 0) == "x")
        run_test("[0,4]=x (broadcast)", read_ft_element(r, 4) == "x")
        run_test("[2,0]=z", read_ft_element(r, 10) == "z")
        run_test("[2,4]=z (broadcast)", read_ft_element(r, 14) == "z")

    # === Group 2: -1 keeps size ===
    print("\nGroup 2: -1 keeps size")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([1, 4], ["a", "b", "c", "d"], tmpdir)
        r = expand_forward(ft, [3, -1])
        run_test("expand [1,4]->[3,-1] shape", r.ft_capacity_shape == [3, 4])
        run_test("[2,3]=d (broadcast)", read_ft_element(r, 11) == "d")

    # === Group 3: Prepend dims ===
    print("\nGroup 3: Prepend dims")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([3], ["p", "q", "r"], tmpdir)
        r = expand_forward(ft, [2, 3])
        run_test("expand [3]->[2,3] shape", r.ft_capacity_shape == [2, 3])
        run_test("[0,0]=p", read_ft_element(r, 0) == "p")
        run_test("[1,2]=r (broadcast)", read_ft_element(r, 5) == "r")

    # === Group 4: No-op expand (same shape) ===
    print("\nGroup 4: No-op expand")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([2, 3], [f"e{i}" for i in range(6)], tmpdir)
        r = expand_forward(ft, [2, 3])
        run_test("expand [2,3]->[2,3] shape", r.ft_capacity_shape == [2, 3])
        run_test("[0,0]=e0", read_ft_element(r, 0) == "e0")
        run_test("[1,2]=e5", read_ft_element(r, 5) == "e5")

    # === Group 5: Lazy expand ===
    print("\nGroup 5: Lazy expand")

    with tempfile.TemporaryDirectory() as tmpdir:
        calls = []

        async def tracking_get(coords, prompt):
            calls.append(coords)
            return (f"val_{coords}", Status.confidence(1.0))

        ft = FutureTensor(tmpdir, tracking_get, [sympy.Integer(1), sympy.Integer(2)])
        r = expand_forward(ft, [3, 2])
        run_test("lazy expand not forwarded", r.ft_forwarded is False)

        prompt_t = st_make_tensor([["p"] * 2] * 3, tmpdir)
        r.ft_forward(prompt_t)
        run_test("lazy expand forwarded", r.ft_forwarded is True)
        # All rows should delegate to row 0 of input
        run_test("lazy [0,0] calls input [0,0]", read_ft_element(r, 0) == "val_[0, 0]")
        run_test("lazy [2,1] calls input [0,1]", read_ft_element(r, 5) == "val_[0, 1]")

    # === Group 6: Error cases ===
    print("\nGroup 6: Error cases")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([3, 2], ["a"] * 6, tmpdir)
        try:
            expand_forward(ft, [3])
            run_test("fewer dims raises ValueError", False)
        except ValueError:
            run_test("fewer dims raises ValueError", True)

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([3, 2], ["a"] * 6, tmpdir)
        try:
            expand_forward(ft, [4, 2])
            run_test("non-1 dim expand raises ValueError", False)
        except ValueError:
            run_test("non-1 dim expand raises ValueError", True)

    print(f"\n  Passed: {passed}, Failed: {failed}, Total: {passed + failed}")
    if failed == 0:
        print("All expand_forward tests passed.")
    else:
        import sys
        sys.exit(1)
