import torch
from experience.symbolic_tensor.function.st_copy import copy_impl
from experience.symbolic_tensor.tensor_util.patch_tensor import patch_tensor
def st_patched(lvalue: torch.Tensor, rvalue: torch.Tensor) -> torch.Tensor:
    """Stateless version of patch_tensor.
    Creates an independent copy of lvalue, applies unified diffs from rvalue
    onto the copy, and returns the copy. The original lvalue is not modified.
    Args:
        lvalue: Source symbolic tensor (not modified).
        rvalue: Patch symbolic tensor containing unified diffs.
    Returns:
        A new symbolic tensor with patches applied.
    """
    assert lvalue.shape == rvalue.shape, (
        f"Shape mismatch: lvalue {list(lvalue.shape)} != rvalue {list(rvalue.shape)}"
    )
    copy = copy_impl(lvalue, lvalue.st_relative_to)
    patch_tensor(copy, rvalue)
    return copy
if __name__ == "__main__":
    import os
    import tempfile
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
    from experience.symbolic_tensor.tensor_util.get_diff_tensor import get_diff_tensor
    print("Running st_patched tests...\n")
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
            tensor.st_relative_to, tensor.st_tensor_uid,
            "storage", os.path.join(*digits), "data",
        )
        path = os.path.realpath(path)
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return f.read()
    print("Test 1: Basic stateless patch")
    with tempfile.TemporaryDirectory() as tmpdir:
        original = make_tensor(["hello world\n", "foo bar\n"], tmpdir)
        modified = make_tensor(["hello universe\n", "foo bar\n"], tmpdir)
        diff = get_diff_tensor(original, modified)
        result = st_patched(original, diff)
        run_test("result[0] patched", read_storage(result, 0).strip() == "hello universe")
        run_test("result[1] unchanged", read_storage(result, 1).strip() == "foo bar")
        run_test("original[0] intact", read_storage(original, 0) == "hello world\n")
        run_test("original[1] intact", read_storage(original, 1) == "foo bar\n")
        run_test("different uid", result.st_tensor_uid != original.st_tensor_uid)
    print("Test 2: Empty diff")
    with tempfile.TemporaryDirectory() as tmpdir:
        a = make_tensor(["same\n"], tmpdir)
        b = make_tensor(["same\n"], tmpdir)
        diff = get_diff_tensor(a, b)
        result = st_patched(a, diff)
        run_test("content preserved", read_storage(result, 0) == "same\n")
        run_test("different uid", result.st_tensor_uid != a.st_tensor_uid)
    print("Test 3: Shape mismatch")
    with tempfile.TemporaryDirectory() as tmpdir:
        a = make_tensor(["x"], tmpdir)
        b = make_tensor(["x", "y"], tmpdir)
        try:
            st_patched(a, b)
            run_test("assertion raised", False)
        except AssertionError:
            run_test("assertion raised", True)
    print("Test 4: 2D tensor")
    with tempfile.TemporaryDirectory() as tmpdir:
        orig = make_tensor([["a\n", "b\n"], ["c\n", "d\n"]], tmpdir)
        mod = make_tensor([["A\n", "b\n"], ["c\n", "D\n"]], tmpdir)
        diff = get_diff_tensor(orig, mod)
        result = st_patched(orig, diff)
        run_test("[0,0] patched to A", read_storage(result, 0).strip() == "A")
        run_test("[0,1] unchanged b", read_storage(result, 1).strip() == "b")
        run_test("[1,0] unchanged c", read_storage(result, 2).strip() == "c")
        run_test("[1,1] patched to D", read_storage(result, 3).strip() == "D")
        run_test("orig[0,0] intact a", read_storage(orig, 0) == "a\n")
        run_test("orig[1,1] intact d", read_storage(orig, 3) == "d\n")
    print("Test 5: Equivalence with patch_tensor")
    with tempfile.TemporaryDirectory() as tmpdir:
        orig = make_tensor(["line1\nline2\n"], tmpdir)
        mod = make_tensor(["line1\nchanged\n"], tmpdir)
        diff = get_diff_tensor(orig, mod)
        result = st_patched(orig, diff)
        manual_copy = copy_impl(orig, tmpdir)
        patch_tensor(manual_copy, diff)
        run_test("same result", read_storage(result, 0) == read_storage(manual_copy, 0))
    print("\nAll tests completed.")