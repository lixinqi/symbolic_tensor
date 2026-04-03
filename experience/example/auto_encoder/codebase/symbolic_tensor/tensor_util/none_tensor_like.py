import uuid
import torch
def none_tensor_like(input: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    """
    Create a zero-filled symbolic tensor matching the shape and metadata of an input tensor.
    Args:
        input: The tensor whose shape and metadata to match.
        dtype: The tensor dtype (default: torch.bfloat16).
    Returns:
        A zero-filled torch.Tensor with st_relative_to and st_tensor_uid attributes,
        matching the input's shape.
    """
    ret = torch.zeros(input.shape, dtype=dtype)
    ret.st_relative_to = input.st_relative_to
    ret.st_tensor_uid = str(uuid.uuid4())
    return ret
if __name__ == "__main__":
    import tempfile
    print("Running tests for none_tensor_like...\n")
    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"✓ {name}")
        else:
            print(f"✗ {name}")
            if expected is not None and actual is not None:
                print(f"  expected: {expected}")
                print(f"  actual:   {actual}")
    with tempfile.TemporaryDirectory() as tmpdir:
        from experience.symbolic_tensor.tensor_util.make_none_tensor import make_none_tensor
        src = make_none_tensor([2, 3], tmpdir)
        t = none_tensor_like(src)
        run_test("Test1: Shape matches", t.shape == src.shape)
        run_test("Test1: dtype bfloat16", t.dtype == torch.bfloat16)
        run_test("Test1: All zeros", torch.all(t == 0).item())
        run_test("Test1: st_relative_to set", t.st_relative_to == src.st_relative_to)
        run_test("Test1: st_tensor_uid is unique", t.st_tensor_uid != src.st_tensor_uid)
        t2 = none_tensor_like(src, dtype=torch.float32)
        run_test("Test2: Custom dtype float32", t2.dtype == torch.float32)
    print("\nAll tests completed.")