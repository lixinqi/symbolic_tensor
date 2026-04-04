import os
import itertools
import torch
from pathlib import Path
from typing import List, Tuple, Union
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor, NestedList
SliceElement = Union[torch.Tensor, slice, int]
def _get_storage_path(tensor: torch.Tensor, coordinates: List[int]) -> Path:
    """Get the storage file Path for a given set of coordinates."""
    stride = tensor.stride()
    flat_index = sum(c * s for c, s in zip(coordinates, stride))
    digits = list(str(flat_index))
    return Path(
        tensor.st_relative_to,
        tensor.st_tensor_uid,
        "storage",
        os.path.join(*digits),
        "data",
    )
def _normalize_slice_element(
    elem: SliceElement, dim_size: int
) -> Tuple[List[int], bool]:
    """Normalize a slice element to (list of indices, collapses_dim).
    Returns:
        indices: list of selected indices for this dimension.
        collapses: True if this element collapses the dimension (int or 0-rank tensor).
    """
    if isinstance(elem, int):
        return [elem], True
    elif isinstance(elem, slice):
        return list(range(*elem.indices(dim_size))), False
    elif isinstance(elem, torch.Tensor):
        if elem.dim() == 0:
            return [int(elem.item())], True
        else:
            return elem.tolist(), False
    else:
        raise TypeError(f"Unsupported slice element type: {type(elem)}")
def _build_nested(flat_paths: List[Path], shape: List[int]) -> NestedList:
    """Reshape a flat list of Paths into a nested list according to shape."""
    if not shape:
        return flat_paths[0]
    if len(shape) == 1:
        return flat_paths
    chunk_size = 1
    for s in shape[1:]:
        chunk_size *= s
    result = []
    for i in range(shape[0]):
        start = i * chunk_size
        result.append(_build_nested(flat_paths[start:start + chunk_size], shape[1:]))
    return result
def slice_view(
    input: torch.Tensor,
    slice_tensors: List[SliceElement],
) -> torch.Tensor:
    """
    Create a symbolic tensor view by selecting elements from an input tensor
    using index tensors, slices, or ints (one per dimension), symlinking to
    the source files.
    Dimension collapsing follows NumPy/PyTorch conventions:
    - int or 0-rank Tensor: selects a single index, collapses the dimension
    - 1D Tensor: selects specific indices, keeps the dimension
    - slice: selects a range, keeps the dimension
    Args:
        input: A symbolic tensor with st_relative_to and st_tensor_uid attributes.
        slice_tensors: A list of indexing elements, one per dimension of input.
    Returns:
        A new symbolic tensor with symlinks to the selected source files.
        Output ndim = input ndim - (number of int elements) - (number of 0-rank tensors).
    """
    assert len(slice_tensors) == len(input.size()), (
        f"Expected {len(input.size())} slice elements, got {len(slice_tensors)}"
    )
    input_size = input.size()
    per_dim = []
    output_shape = []
    for d, elem in enumerate(slice_tensors):
        indices, collapses = _normalize_slice_element(elem, input_size[d])
        per_dim.append((indices, collapses))
        if not collapses:
            output_shape.append(len(indices))
    if 0 in output_shape:
        ret = make_tensor([], input.st_relative_to, symlink=True)
        reshaped = ret.view(output_shape)
        reshaped.st_relative_to = ret.st_relative_to
        reshaped.st_tensor_uid = ret.st_tensor_uid
        return reshaped
    all_indices = [idx for idx, _ in per_dim]
    selected_paths: List[Path] = []
    for combo in itertools.product(*all_indices):
        selected_paths.append(_get_storage_path(input, list(combo)))
    for p in selected_paths:
        if not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
    if output_shape:
        nested_paths = _build_nested(selected_paths, output_shape)
    else:
        nested_paths = selected_paths[0]
    ret = make_tensor(nested_paths, input.st_relative_to, symlink=True)
    count_collapsed = sum(1 for _, collapses in per_dim if collapses)
    expected_ndim = len(input.size()) - count_collapsed
    assert len(ret.size()) == expected_ndim, (
        f"Expected output ndim {expected_ndim}, got {len(ret.size())}"
    )
    return ret
if __name__ == "__main__":
    import tempfile
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor as mt
    print("Running tests for slice_view...\n")
    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")
    print("Test 1: 1D Tensor indexing")
    with tempfile.TemporaryDirectory() as tmpdir:
        data = ["a", "b", "c", "d"]
        t = mt(data, tmpdir)
        idx = [torch.tensor([0, 2], dtype=torch.long)]
        result = slice_view(t, idx)
        run_test("Shape is [2]", list(result.shape) == [2])
        root = os.path.join(tmpdir, result.st_tensor_uid, "storage")
        with open(os.path.join(root, "0", "data")) as f:
            run_test("Element 0 -> 'a'", f.read() == "a")
        with open(os.path.join(root, "1", "data")) as f:
            run_test("Element 1 -> 'c'", f.read() == "c")
        run_test("Is symlink", os.path.islink(os.path.join(root, "0", "data")))
    print("Test 2: 2D Tensor indexing")
    with tempfile.TemporaryDirectory() as tmpdir:
        data = [["a", "b", "c"], ["d", "e", "f"]]
        t = mt(data, tmpdir)
        idx = [torch.tensor([0, 1], dtype=torch.long),
               torch.tensor([1, 2], dtype=torch.long)]
        result = slice_view(t, idx)
        run_test("Shape is [2, 2]", list(result.shape) == [2, 2])
        root = os.path.join(tmpdir, result.st_tensor_uid, "storage")
        with open(os.path.join(root, "0", "data")) as f:
            run_test("(0,1) -> 'b'", f.read() == "b")
        with open(os.path.join(root, "1", "data")) as f:
            run_test("(0,2) -> 'c'", f.read() == "c")
        with open(os.path.join(root, "2", "data")) as f:
            run_test("(1,1) -> 'e'", f.read() == "e")
        with open(os.path.join(root, "3", "data")) as f:
            run_test("(1,2) -> 'f'", f.read() == "f")
    print("Test 3: int indexing collapses dim")
    with tempfile.TemporaryDirectory() as tmpdir:
        data = [["a", "b", "c"], ["d", "e", "f"]]
        t = mt(data, tmpdir)
        result = slice_view(t, [1, slice(0, 3)])
        run_test("Shape is [3] (dim 0 collapsed)", list(result.shape) == [3])
        root = os.path.join(tmpdir, result.st_tensor_uid, "storage")
        with open(os.path.join(root, "0", "data")) as f:
            run_test("Row 1, col 0 -> 'd'", f.read() == "d")
        with open(os.path.join(root, "2", "data")) as f:
            run_test("Row 1, col 2 -> 'f'", f.read() == "f")
    print("Test 4: slice indexing")
    with tempfile.TemporaryDirectory() as tmpdir:
        data = ["a", "b", "c", "d", "e"]
        t = mt(data, tmpdir)
        result = slice_view(t, [slice(1, 4)])
        run_test("Shape is [3]", list(result.shape) == [3])
        root = os.path.join(tmpdir, result.st_tensor_uid, "storage")
        with open(os.path.join(root, "0", "data")) as f:
            run_test("slice[1] -> 'b'", f.read() == "b")
        with open(os.path.join(root, "2", "data")) as f:
            run_test("slice[3] -> 'd'", f.read() == "d")
    print("Test 5: 0-rank tensor collapses dim")
    with tempfile.TemporaryDirectory() as tmpdir:
        data = [["x", "y"], ["z", "w"]]
        t = mt(data, tmpdir)
        result = slice_view(t, [torch.tensor(0), slice(None)])
        run_test("Shape is [2] (dim 0 collapsed)", list(result.shape) == [2])
        root = os.path.join(tmpdir, result.st_tensor_uid, "storage")
        with open(os.path.join(root, "0", "data")) as f:
            run_test("(0,0) -> 'x'", f.read() == "x")
        with open(os.path.join(root, "1", "data")) as f:
            run_test("(0,1) -> 'y'", f.read() == "y")
    print("Test 6: All dims collapsed -> scalar")
    with tempfile.TemporaryDirectory() as tmpdir:
        data = [["a", "b"], ["c", "d"]]
        t = mt(data, tmpdir)
        result = slice_view(t, [1, 0])
        run_test("Shape is [] (0-dim)", list(result.shape) == [])
        run_test("numel is 1", result.numel() == 1)
        stored = os.path.join(tmpdir, result.st_tensor_uid, "storage", "0", "data")
        run_test("Storage file exists", os.path.isfile(stored))
        run_test("Is symlink", os.path.islink(stored))
        with open(stored) as f:
            run_test("(1,0) -> 'c'", f.read() == "c")
        # Verify symlink target resolves to the original tensor's storage
        real = os.path.realpath(stored)
        run_test("Resolves to input storage", t.st_tensor_uid in real)
    print("Test 7: Dimension mismatch")
    with tempfile.TemporaryDirectory() as tmpdir:
        data = [["a", "b"], ["c", "d"]]
        t = mt(data, tmpdir)
        try:
            slice_view(t, [torch.tensor([0])])
            run_test("Should have raised", False)
        except AssertionError:
            run_test("AssertionError raised", True)
    print("Test 8: Relative symlinks")
    with tempfile.TemporaryDirectory() as tmpdir:
        data = ["hello"]
        t = mt(data, tmpdir)
        result = slice_view(t, [torch.tensor([0])])
        link = os.path.join(tmpdir, result.st_tensor_uid, "storage", "0", "data")
        target = os.readlink(link)
        run_test("Symlink is relative", not os.path.isabs(target))
    print("Test 9: Empty tensor index")
    with tempfile.TemporaryDirectory() as tmpdir:
        data = [["a", "b"], ["c", "d"]]
        t = mt(data, tmpdir)
        result = slice_view(t, [torch.tensor([], dtype=torch.long), slice(None)])
        run_test("Shape is [0, 2]", list(result.shape) == [0, 2], [0, 2], list(result.shape))
        run_test("numel is 0", result.numel() == 0)
    print("Test 10: slice_view(make_none_tensor) + assign_tensor")
    with tempfile.TemporaryDirectory() as tmpdir:
        from experience.symbolic_tensor.tensor_util.make_none_tensor import make_none_tensor
        from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor
        none_t = make_none_tensor([2, 3], tmpdir)
        view = slice_view(none_t, [0, torch.tensor([1, 2])])
        run_test("View shape is [2]", list(view.shape) == [2])
        view_root = os.path.join(tmpdir, view.st_tensor_uid, "storage")
        link0 = os.path.join(view_root, "0", "data")
        run_test("View symlink exists", os.path.islink(link0))
        target_real = os.path.realpath(link0)
        run_test("Target parent dir exists", os.path.isdir(os.path.dirname(target_real)))
        run_test("Target file not touched (dangling)", not os.path.isfile(target_real))
        src = mt(["hello", "world"], tmpdir)
        assign_tensor(view, src)
        orig_root = os.path.join(tmpdir, none_t.st_tensor_uid, "storage")
        with open(os.path.join(orig_root, "1", "data")) as f:
            run_test("Original [0,1] = 'hello'", f.read() == "hello")
        with open(os.path.join(orig_root, "2", "data")) as f:
            run_test("Original [0,2] = 'world'", f.read() == "world")
    print("\nAll tests completed.")