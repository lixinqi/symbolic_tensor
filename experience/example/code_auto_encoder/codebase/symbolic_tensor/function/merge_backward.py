import os
import itertools
import shutil
import tempfile
import torch
from typing import Optional, Type
from experience.symbolic_tensor.tensor_util.todo_tensor_like import todo_tensor_like
from experience.symbolic_tensor.tensor_util.st_patched import st_patched
from experience.symbolic_tensor.tensor_util.slice_view import slice_view
from experience.symbolic_tensor.tensor_util.get_diff_tensor import get_diff_tensor
from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.fs_util.text_merger import TextMerger
def _get_storage_path(tensor: torch.Tensor, flat_index: int) -> str:
    """Get storage file path for a flat index."""
    digits = list(str(flat_index))
    return os.path.join(
        tensor.st_relative_to,
        tensor.st_tensor_uid,
        "storage",
        os.path.join(*digits),
        "data",
    )
def _read_storage(tensor: torch.Tensor, flat_index: int) -> Optional[str]:
    """Read text content at a flat index, resolving symlinks."""
    path = os.path.realpath(_get_storage_path(tensor, flat_index))
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return f.read()
def merge_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    output: torch.Tensor,
    axis: int = -1,
    text_merger: Optional[Type] = None,
) -> Optional[torch.Tensor]:
    """Backward pass for merge_forward.
    For each output position, applies grad_output diff to output to get
    better_version_output (via st_patched), unpacks both, asserts same schema,
    then computes per-frame grad_input via st_get_diff and st_assign.
    Dual-channel gradient:
      Numeric: grad_input.data[..., k, ...] = improved_coefficient[k]
      Symbolic: grad_input storage = diff(original_input[k], improved_content[k])
    Args:
        grad_output: Gradient w.r.t. forward output (symbolic tensor with diffs).
        input: Original input saved from forward ctx.
        output: Original output saved from forward ctx (TextMerger.pack format).
        axis: Reduced axis (default -1).
        text_merger: Class with pack/unpack. Defaults to TextMerger.
    Returns:
        grad_input tensor, or None if input doesn't require grad.
    """
    if not input.requires_grad:
        return None
    if text_merger is None:
        text_merger = TextMerger
    ndim = input.dim()
    if axis < 0:
        axis = ndim + axis
    assert 0 <= axis < ndim
    grad_input = todo_tensor_like(input)
    grad_input.data.zero_()
    better_version_output = st_patched(output, grad_output)
    input_shape = list(input.shape)
    output_shape = input_shape[:axis] + input_shape[axis + 1:]
    if output_shape:
        output_ranges = [range(s) for s in output_shape]
    else:
        output_ranges = []
    tmpdir = tempfile.mkdtemp()
    try:
        for out_coords in itertools.product(*output_ranges) if output_ranges else [()]:
            out_coords = list(out_coords)
            if output.dim() > 0:
                out_flat = sum(c * s for c, s in zip(out_coords, output.stride()))
            else:
                out_flat = 0
            if better_version_output.dim() > 0:
                bv_flat = sum(
                    c * s for c, s in zip(out_coords, better_version_output.stride())
                )
            else:
                bv_flat = 0
            output_content = _read_storage(output, out_flat)
            bv_content = _read_storage(better_version_output, bv_flat)
            if output_content is None or bv_content is None:
                continue
            original_frames = text_merger.unpack(output_content)
            improved_frames = text_merger.unpack(bv_content)
            assert len(original_frames) == len(improved_frames), (
                f"Frame count mismatch at output {out_coords}: "
                f"{len(original_frames)} vs {len(improved_frames)}"
            )
            orig_indices = [f[0] for f in original_frames]
            impr_indices = [f[0] for f in improved_frames]
            assert orig_indices == impr_indices, (
                f"Frame index mismatch at output {out_coords}: "
                f"{orig_indices} vs {impr_indices}"
            )
            for frame_index, improved_coefficient, improved_content in improved_frames:
                in_coords = out_coords[:axis] + [frame_index] + out_coords[axis:]
                grad_input.data[tuple(in_coords)] = improved_coefficient
                input_element_view = slice_view(input, in_coords)
                grad_input_element_view = slice_view(grad_input, in_coords)
                improved_element_tensor = make_tensor(improved_content, tmpdir)
                diff = get_diff_tensor(input_element_view, improved_element_tensor)
                assign_tensor(grad_input_element_view, diff)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return grad_input
if __name__ == "__main__":
    from experience.symbolic_tensor.function.merge_forward import merge_forward
    from experience.fs_util.text_merger import kFrameMarker
    print("Running merge_backward tests...\n")
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
    print("Test 1: No grad when not required")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor(["a\n", "b\n"], tmpdir)
        output = merge_forward(inp, axis=0)
        grad_out = make_tensor([""], tmpdir)
        result = merge_backward(grad_out, inp, output, axis=0)
        run_test("Returns None", result is None)
    print("Test 2: 2D reduce axis=-1, content diff")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["alpha\n", "beta\n"], ["gamma\n", "delta\n"]], tmpdir)
        inp.requires_grad_(True)
        output = merge_forward(inp, axis=-1)
        # Read packed content, modify first row: "alpha" -> "ALPHA"
        out_content_0 = read_out(output, 0)
        modified_content_0 = out_content_0.replace("alpha", "ALPHA")
        out_content_1 = read_out(output, 1)
        modified_output = make_tensor([modified_content_0, out_content_1], tmpdir)
        grad_out = get_diff_tensor(output, modified_output)
        grad_input = merge_backward(grad_out, inp, output, axis=-1)
        run_test("grad_input shape [2, 2]", list(grad_input.shape) == [2, 2])
        gi_00 = read_out(grad_input, 0)
        run_test("grad_input[0,0] has diff", gi_00 is not None and "---" in gi_00)
        run_test("grad_input[0,0] has +ALPHA", "+ALPHA" in gi_00 if gi_00 else False)
        run_test("grad_input[0,0] has -alpha", "-alpha" in gi_00 if gi_00 else False)
        gi_01 = read_out(grad_input, 1)
        run_test("grad_input[0,1] empty diff", gi_01 is not None and "---" not in gi_01)
        gi_10 = read_out(grad_input, 2)
        gi_11 = read_out(grad_input, 3)
        run_test("grad_input[1,0] empty diff", gi_10 is not None and "---" not in gi_10)
        run_test("grad_input[1,1] empty diff", gi_11 is not None and "---" not in gi_11)
    print("Test 3: Coefficient propagation")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["x\n", "y\n"]], tmpdir)
        inp.requires_grad_(True)
        output = merge_forward(inp, axis=-1)
        # Modify coefficient in packed text: change "coefficient: 1.0" to "coefficient: 0.5"
        out_content = read_out(output, 0)
        # Replace first occurrence of "coefficient: 1.0" with "coefficient: 0.5"
        modified_content = out_content.replace(
            "coefficient: 1.0", "coefficient: 0.5", 1
        )
        modified_output = make_tensor([modified_content], tmpdir)
        grad_out = get_diff_tensor(output, modified_output)
        grad_input = merge_backward(grad_out, inp, output, axis=-1)
        run_test("grad_input shape [1, 2]", list(grad_input.shape) == [1, 2])
        run_test(
            "grad_input[0,0] coeff = 0.5",
            abs(grad_input.data[0, 0].item() - 0.5) < 1e-5,
            0.5, grad_input.data[0, 0].item(),
        )
        run_test(
            "grad_input[0,1] coeff = 1.0",
            abs(grad_input.data[0, 1].item() - 1.0) < 1e-5,
            1.0, grad_input.data[0, 1].item(),
        )
    print("Test 4: 2D reduce axis=0")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["a\n", "b\n"], ["c\n", "d\n"]], tmpdir)
        inp.requires_grad_(True)
        output = merge_forward(inp, axis=0)
        # Modify column 0: change "a" to "A" (row 0 of column 0)
        out_content_0 = read_out(output, 0)
        modified_content_0 = out_content_0.replace("a\n", "A\n", 1)
        out_content_1 = read_out(output, 1)
        modified_output = make_tensor([modified_content_0, out_content_1], tmpdir)
        grad_out = get_diff_tensor(output, modified_output)
        grad_input = merge_backward(grad_out, inp, output, axis=0)
        run_test("grad_input shape [2, 2]", list(grad_input.shape) == [2, 2])
        gi_00 = read_out(grad_input, 0)
        run_test("grad_input[0,0] has diff", gi_00 is not None and "---" in gi_00)
    print("Test 5: No-change diff")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor(["hello\n", "world\n"], tmpdir)
        inp.requires_grad_(True)
        output = merge_forward(inp, axis=0)
        grad_out = get_diff_tensor(output, output)
        grad_input = merge_backward(grad_out, inp, output, axis=0)
        run_test("grad_input not None", grad_input is not None)
        gi_0 = read_out(grad_input, 0)
        gi_1 = read_out(grad_input, 1)
        run_test("grad_input[0] empty diff", gi_0 is not None and "---" not in gi_0)
        run_test("grad_input[1] empty diff", gi_1 is not None and "---" not in gi_1)
    print("Test 6: Schema assertion")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["a\n", "b\n"]], tmpdir)
        inp.requires_grad_(True)
        output = merge_forward(inp, axis=-1)
        out_content = read_out(output, 0)
        first_marker_end = out_content.find("\n", out_content.find(kFrameMarker))
        second_marker = out_content.find(kFrameMarker, first_marker_end)
        if second_marker > 0:
            mangled = out_content[:second_marker].rstrip()
            mangled_output = make_tensor([mangled], tmpdir)
            grad_out = get_diff_tensor(output, mangled_output)
            try:
                merge_backward(grad_out, inp, output, axis=-1)
                run_test("schema assertion raised", False)
            except AssertionError:
                run_test("schema assertion raised", True)
        else:
            run_test("schema assertion raised (skipped: couldn't mangle)", False)
    print("\nAll tests completed.")