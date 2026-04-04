import os
import itertools
import torch
from typing import List
def _get_raw_storage_path(tensor: torch.Tensor, coordinates: List[int]) -> str:
    """Get storage file path WITHOUT resolving symlinks."""
    stride = tensor.stride()
    flat_index = sum(c * s for c, s in zip(coordinates, stride))
    digits = list(str(flat_index))
    return os.path.join(
        tensor.st_relative_to,
        tensor.st_tensor_uid,
        "storage",
        os.path.join(*digits),
        "data",
    )
def assign_view(lvalue: torch.Tensor, rvalue: torch.Tensor) -> None:
    """Create symlinks from lvalue storage to rvalue storage element-wise.
    Both tensors must have the same shape. For each coordinate, a relative
    symlink is created from lvalue's storage path to rvalue's resolved storage path.
    Rvalue storage files are touched if they don't exist yet.
    Args:
        lvalue: Destination symbolic tensor (will contain symlinks).
        rvalue: Source symbolic tensor (symlink targets).
    """
    assert lvalue.shape == rvalue.shape, (
        f"Shape mismatch: lvalue {list(lvalue.shape)} != rvalue {list(rvalue.shape)}"
    )
    coords_list = list(itertools.product(*[range(s) for s in lvalue.size()]))
    for coords in coords_list:
        coords = list(coords)
        lvalue[tuple(coords)] = rvalue[tuple(coords)]
        lvalue_path = _get_raw_storage_path(lvalue, coords)
        rvalue_path = os.path.realpath(_get_raw_storage_path(rvalue, coords))
        if not os.path.exists(rvalue_path):
            os.makedirs(os.path.dirname(rvalue_path), exist_ok=True)
            open(rvalue_path, "a").close()
        os.makedirs(os.path.dirname(lvalue_path), exist_ok=True)
        if os.path.islink(lvalue_path) or os.path.exists(lvalue_path):
            os.remove(lvalue_path)
        rel_target = os.path.relpath(rvalue_path, os.path.dirname(lvalue_path))
        os.symlink(rel_target, lvalue_path)
if __name__ == "__main__":
    import tempfile
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
    from experience.symbolic_tensor.tensor_util.make_none_tensor import make_none_tensor
    from experience.symbolic_tensor.tensor_util.slice_view import slice_view
    print("Running assign_view tests...\n")
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
    print("Test 1: 1D assign_view")
    with tempfile.TemporaryDirectory() as tmpdir:
        src = make_tensor(["hello", "world"], tmpdir)
        dst = make_tensor(["aaa", "bbb"], tmpdir)
        assign_view(dst, src)
        run_test("dst[0] == 'hello'", read_storage(dst, 0) == "hello")
        run_test("dst[1] == 'world'", read_storage(dst, 1) == "world")
        dst_path = os.path.join(tmpdir, dst.st_tensor_uid, "storage", "0", "data")
        run_test("dst[0] is symlink", os.path.islink(dst_path))
        run_test("symlink is relative", not os.path.isabs(os.readlink(dst_path)))
    print("Test 2: 2D assign_view")
    with tempfile.TemporaryDirectory() as tmpdir:
        src = make_tensor([["a", "b"], ["c", "d"]], tmpdir)
        dst = make_tensor([["x", "y"], ["z", "w"]], tmpdir)
        assign_view(dst, src)
        run_test("dst[0,0]='a'", read_storage(dst, 0) == "a")
        run_test("dst[0,1]='b'", read_storage(dst, 1) == "b")
        run_test("dst[1,0]='c'", read_storage(dst, 2) == "c")
        run_test("dst[1,1]='d'", read_storage(dst, 3) == "d")
    print("Test 3: Shared storage (write-through)")
    with tempfile.TemporaryDirectory() as tmpdir:
        src = make_tensor(["original"], tmpdir)
        dst = make_tensor(["placeholder"], tmpdir)
        assign_view(dst, src)
        run_test("dst reads 'original'", read_storage(dst, 0) == "original")
        src_path = os.path.join(tmpdir, src.st_tensor_uid, "storage", "0", "data")
        with open(src_path, "w") as f:
            f.write("updated")
        run_test("dst reads 'updated'", read_storage(dst, 0) == "updated")
    print("Test 4: Tensor values copied")
    with tempfile.TemporaryDirectory() as tmpdir:
        src = make_tensor(["a", "b"], tmpdir)
        src[0] = 0.5
        src[1] = 0.8
        dst = make_tensor(["x", "y"], tmpdir)
        assign_view(dst, src)
        run_test("dst value[0] ~ 0.5", abs(dst[0].item() - 0.5) < 1e-2)
        run_test("dst value[1] ~ 0.8", abs(dst[1].item() - 0.8) < 1e-2)
    print("Test 5: Shape mismatch")
    with tempfile.TemporaryDirectory() as tmpdir:
        src = make_tensor(["a", "b"], tmpdir)
        dst = make_tensor(["x", "y", "z"], tmpdir)
        try:
            assign_view(dst, src)
            run_test("Assertion raised", False)
        except AssertionError:
            run_test("Assertion raised", True)
    print("Test 6: Rvalue from make_none_tensor (touch)")
    with tempfile.TemporaryDirectory() as tmpdir:
        src = make_none_tensor([2], tmpdir)
        dst = make_tensor(["a", "b"], tmpdir)
        assign_view(dst, src)
        dst_path = os.path.join(tmpdir, dst.st_tensor_uid, "storage", "0", "data")
        run_test("dst[0] is symlink", os.path.islink(dst_path))
        real_target = os.path.realpath(dst_path)
        run_test("target file exists (touched)", os.path.isfile(real_target))
        run_test("target content is empty", read_storage(dst, 0) == "")
    print("Test 7: Overwrite slice_view symlinks")
    with tempfile.TemporaryDirectory() as tmpdir:
        original = make_tensor(["old_a", "old_b", "old_c"], tmpdir)
        view = slice_view(original, [torch.tensor([0, 2])])
        run_test("view[0]='old_a' before", read_storage(view, 0) == "old_a")
        new_src = make_tensor(["new_x", "new_y"], tmpdir)
        assign_view(view, new_src)
        run_test("view[0]='new_x' after", read_storage(view, 0) == "new_x")
        run_test("view[1]='new_y' after", read_storage(view, 1) == "new_y")
        run_test("original[0] still 'old_a'", read_storage(original, 0) == "old_a")
    print("Test 8: Scalar assign_view")
    with tempfile.TemporaryDirectory() as tmpdir:
        src = make_tensor("scalar_src", tmpdir)
        dst = make_tensor("scalar_dst", tmpdir)
        assign_view(dst, src)
        run_test("dst == 'scalar_src'", read_storage(dst, 0) == "scalar_src")
        dst_path = os.path.join(tmpdir, dst.st_tensor_uid, "storage", "0", "data")
        run_test("is symlink", os.path.islink(dst_path))
    print("\nAll tests completed.")