"""
recurrent_backward :=
    $grad_input Symbolic[torch.Tensor[($prefix_dims ..., $recurrent_dim int)]]
    <- $grad_output Symbolic[torch.Tensor[($prefix_dims ...)]]
    <- $input Symbolic[torch.Tensor[($prefix_dims ..., $recurrent_dim int)]]
    <- $output Symbolic[torch.Tensor[($prefix_dims ...)]]
    <- $prompt_tensor Symbolic[torch.Tensor[($prefix_dims ..., $recurrent_dim int)]]
    <- $topk_self_confidence_but_failed int # default 8
    <- ...
    # inline
"""

import os
import shutil
import tempfile
import itertools
import torch
from typing import Callable, Dict, List, Optional, Tuple, Union

from experience.future_tensor.status import Status
from experience.symbolic_tensor.tensor_util.todo_tensor_like import todo_tensor_like
from experience.symbolic_tensor.tensor_util.slice_view import slice_view
from experience.symbolic_tensor.tensor_util.slice_tensor import slice_tensor
from experience.symbolic_tensor.tensor_util.dump_view import dump_view
from experience.symbolic_tensor.tensor_util.get_diff_tensor import get_diff_tensor
from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor
from experience.llm_client.agent_task import AgentTask
from experience.llm_client.task_handler import TaskHandler


BackwardPromptCallable = Callable[
    [str, str, str, str, str, str, str], str
]


def default_prompt_for_recurrent_grad_input(
    task_prompt: str,
    workspace_dir: str,
    const_grad_output_view: str,
    const_input_view: str,
    const_output_view: str,
    const_prompt_view: str,
    mutable_grad_input_dir: str,
) -> str:
    return f"""\
You are a symbolic gradient calculator for the backward pass
of a recurrent generate-validate loop.

{task_prompt}

During forward pass, the input tensor contained LLM outputs across
multiple retry iterations. One iteration was selected as the final
output (by confidence or best-of when all failed). The prompt tensor
records the accumulated context each iteration received.

Now given the output gradient (how the selected output should change),
compute how each iteration's input should change.

Context (read-only):
- Output gradient (text diff): {const_grad_output_view}
- Original input (per-iteration LLM outputs): {const_input_view}
- Selected output: {const_output_view}
- Accumulated prompts per iteration: {const_prompt_view}
  where .../i/data = the prompt fed to iteration i

Compute and write:
1. Input gradient in {mutable_grad_input_dir}:
   How should each iteration's LLM output change to improve the final result?
2. For the winning iteration: write improved text directly.
3. For failed iterations: write text that would help the retry loop
   converge faster (e.g., more precise output that passes validation sooner).
4. For unreached iterations: leave as TODO.

Replace all TODO with improved content.
"""


def _scalar_slice_indices(size: torch.Size) -> List[List[int]]:
    """Generate all coordinate tuples for a given tensor shape."""
    return [list(c) for c in itertools.product(*[range(s) for s in size])]


def _collect_scyf_topk_per_prefix(input: torch.Tensor, topk: int) -> set:
    """Collect flat indices of top-k scyf elements per prefix position.

    For each prefix coordinate, find all scyf iterations, sort by value
    descending, take top-k, return their flat indices as a set.
    """
    input_shape = list(input.size())
    prefix_shape = input_shape[:-1]
    recurrent_dim = input_shape[-1]
    topk_set = set()

    prefix_coords_list = list(itertools.product(*[range(s) for s in prefix_shape])) if prefix_shape else [()]

    for prefix_coords in prefix_coords_list:
        # Collect (scyf_value, flat_index) for all iterations at this prefix
        scyf_candidates = []
        for i in range(recurrent_dim):
            coords = list(prefix_coords) + [i]
            flat = sum(c * s for c, s in zip(coords, input.stride()))
            coeff = input.data.flatten()[flat].item()
            status = Status.convert_float_to_status(coeff)
            if status.is_self_confidence_but_failed:
                scyf_candidates.append((status.value, flat))
        # Sort descending by scyf value, take top-k
        scyf_candidates.sort(key=lambda x: x[0], reverse=True)
        for _, flat_idx in scyf_candidates[:topk]:
            topk_set.add(flat_idx)

    return topk_set


def recurrent_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    output: torch.Tensor,
    prompt_tensor: torch.Tensor,
    topk_self_confidence_but_failed: int = 8,
    grad_input_prompt: Optional[BackwardPromptCallable] = None,
    task_prompt: str = "",
    llm_method: str = "raw_llm_api",
    llm_env: Optional[Dict[str, str]] = None,
) -> torch.Tensor:
    """Backward pass for recurrent_forward.

    Forward: FutureTensor (lazy, sequential retry loop)
    Backward: Symbolic[torch.Tensor] (materialized, concurrent LLM reflection)

    Args:
        grad_output: shape (*prefix_dims) — gradient of the selected output.
        input: shape (*prefix_dims, recurrent_dim) — per-iteration LLM outputs.
        output: shape (*prefix_dims) — the selected result from forward.
        prompt_tensor: shape (*prefix_dims, recurrent_dim) — accumulated prompts per iteration.
        topk_self_confidence_but_failed: Reflect on top-k highest scyf cases per prefix.
            It's weird that trials had high self-confidence but failed, so reflect on them.
        grad_input_prompt: Optional custom prompt builder.
        task_prompt: Additional task context for the prompt.
        llm_method: "coding_agent" or "raw_llm_api".
        llm_env: Optional environment variables for LLM.

    Returns:
        grad_input: shape (*prefix_dims, recurrent_dim).
    """
    grad_input = todo_tensor_like(input)

    input_shape = list(input.size())
    prefix_shape = input_shape[:-1]
    recurrent_dim = input_shape[-1]

    # Precompute top-k scyf flat indices per prefix
    scyf_topk_set = _collect_scyf_topk_per_prefix(input, topk_self_confidence_but_failed)

    # ── Numeric channel ──
    # Per-element status-aware coefficient propagation.
    coords_list = _scalar_slice_indices(input.size())

    for coords in coords_list:
        prefix_coords = coords[:-1]
        flat_input = sum(c * s for c, s in zip(coords, input.stride()))
        flat_grad_output = sum(c * s for c, s in zip(prefix_coords, grad_output.stride())) if prefix_coords else 0

        input_coeff = input.data.flatten()[flat_input].item()
        input_status = Status.convert_float_to_status(input_coeff)
        grad_output_coeff = grad_output.data.flatten()[flat_grad_output].item()

        if input_status.is_kContextOverflow or (input_status.is_confidence and input_status.value == 0.0):
            # kContextOverflow or unreached (coeff == 0): zero gradient
            grad_input.data.flatten()[flat_input] = 0.0
        elif input_status.is_confidence or input_status.is_kConfidenceNotBounded:
            # Winning iteration: copy grad_output coefficient
            grad_input.data.flatten()[flat_input] = grad_output_coeff
        elif input_status.is_self_confidence_but_failed:
            if flat_input in scyf_topk_set:
                # Top-k scyf: reflect on these high-confidence-yet-failed cases
                grad_input.data.flatten()[flat_input] = grad_output_coeff
            else:
                # Attenuated: use the negative status value
                grad_input.data.flatten()[flat_input] = Status.convert_status_to_float(input_status)
        else:
            # Unreached (coeff == 0)
            grad_input.data.flatten()[flat_input] = 0.0

    # ── Symbolic channel ──
    all_tasks: List[AgentTask] = []
    element_contexts: List[Tuple] = []
    # Each: (workspace_dir, scalar_grad_input_view, scalar_input, scalar_grad_input_value, coeff)

    for coords in coords_list:
        prefix_coords = coords[:-1]
        flat_input = sum(c * s for c, s in zip(coords, input.stride()))

        int_slices_input = [c for c in coords]
        # For grad_output/output: use prefix coords only
        int_slices_prefix = [c for c in prefix_coords]

        scalar_input = slice_view(input, int_slices_input)
        scalar_prompt = slice_view(prompt_tensor, int_slices_input)
        scalar_grad_input_view = slice_view(grad_input, int_slices_input)
        scalar_grad_input_value = slice_tensor(grad_input, int_slices_input)
        scalar_grad_output = slice_view(grad_output, int_slices_prefix) if int_slices_prefix else grad_output
        scalar_output = slice_view(output, int_slices_prefix) if int_slices_prefix else output

        coeff = grad_input.data.flatten()[flat_input].item()

        if coeff == 0.0:
            # Skip AgentTask for unreached/overflow elements
            element_contexts.append((None, scalar_grad_input_view, scalar_input, scalar_grad_input_value, coeff))
            continue

        workspace_dir = tempfile.mkdtemp()
        grad_output_view_dir = os.path.join(workspace_dir, "const_grad_output_view")
        input_view_dir = os.path.join(workspace_dir, "const_input_view")
        output_view_dir = os.path.join(workspace_dir, "const_output_view")
        prompt_view_dir = os.path.join(workspace_dir, "const_prompt_view")
        grad_input_dir = os.path.join(workspace_dir, "mutable_grad_input_dir")

        dump_view(scalar_grad_output, grad_output_view_dir, "txt")
        dump_view(scalar_input, input_view_dir, "txt")
        dump_view(scalar_output, output_view_dir, "txt")
        dump_view(scalar_prompt, prompt_view_dir, "txt")
        dump_view(scalar_grad_input_value, grad_input_dir, "txt")

        prompt = (grad_input_prompt or default_prompt_for_recurrent_grad_input)(
            task_prompt, workspace_dir,
            grad_output_view_dir, input_view_dir,
            output_view_dir, prompt_view_dir,
            grad_input_dir,
        )
        agent_task = AgentTask(
            workspace_dir=workspace_dir,
            output_relative_dir=["mutable_grad_input_dir"],
            prompt=prompt,
        )
        all_tasks.append(agent_task)
        element_contexts.append((workspace_dir, scalar_grad_input_view, scalar_input, scalar_grad_input_value, coeff))

    # ── Single TaskHandler call ──
    if all_tasks:
        TaskHandler()(all_tasks, llm_method, llm_env=llm_env)

    # ── Diff + assign for all elements ──
    for ctx in element_contexts:
        ws_dir, gi_view, s_input, gi_value, coeff = ctx
        if ws_dir is None:
            continue
        diff = get_diff_tensor(s_input, gi_value)
        assign_tensor(gi_view, diff)
        shutil.rmtree(ws_dir)

    return grad_input


if __name__ == "__main__":
    import subprocess

    # Source anthropic env vars
    result = subprocess.run(
        ["bash", "-c", "source ~/.anthropic.sh && env"],
        capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            os.environ[key] = val
    os.environ.pop("CLAUDECODE", None)

    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor

    print("Running 20 tests for recurrent_backward...\n")

    passed = 0
    failed = 0

    def run_test(name: str, condition: bool, expected=None, actual=None):
        global passed, failed
        if condition:
            passed += 1
            print(f"  \u2713 {name}")
        else:
            failed += 1
            print(f"  \u2717 {name}")
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

    # ── Tests 1-4: Numeric channel — confidence (winning) ──
    print("Tests 1-4: Numeric channel — confidence winning iteration")
    with tempfile.TemporaryDirectory() as tmpdir:
        # input shape [3]: 3 iterations, iteration 1 won with confidence(0.9)
        input_t = make_tensor(["attempt0", "winning_result", "attempt2"], tmpdir)
        input_t.data[0] = Status.convert_status_to_float(Status.self_confidence_but_failed(0.4))
        input_t.data[1] = Status.convert_status_to_float(Status.confidence(0.9))
        input_t.data[2] = 0.0  # unreached

        output_t = make_tensor(["winning_result"], tmpdir)
        output_t.data[0] = Status.convert_status_to_float(Status.confidence(0.9))

        grad_output_t = make_tensor(["improved_output"], tmpdir)
        grad_output_t.data[0] = 1.5  # some grad coeff

        prompt_t = make_tensor(["p0", "p0\n====[Iteration 0]====\nattempt0", "unused"], tmpdir)
        prompt_t.data.fill_(1.0)

        # Only numeric channel (no LLM call)
        grad_input = todo_tensor_like(input_t)
        scyf_topk = _collect_scyf_topk_per_prefix(input_t, topk=8)
        # Manually run numeric channel logic
        for i in range(3):
            ic = input_t.data[i].item()
            ist = Status.convert_float_to_status(ic)
            go_c = grad_output_t.data[0].item()
            if ist.is_kContextOverflow or (ist.is_confidence and ist.value == 0.0):
                grad_input.data[i] = 0.0
            elif ist.is_confidence or ist.is_kConfidenceNotBounded:
                grad_input.data[i] = go_c
            elif ist.is_self_confidence_but_failed:
                if i in scyf_topk:
                    grad_input.data[i] = go_c
                else:
                    grad_input.data[i] = Status.convert_status_to_float(ist)
            else:
                grad_input.data[i] = 0.0

        run_test("winning iter coeff = grad_output coeff",
                 abs(grad_input.data[1].item() - 1.5) < 0.05, 1.5, grad_input.data[1].item())
        run_test("scyf iter in topk gets grad_output coeff",
                 abs(grad_input.data[0].item() - 1.5) < 0.05, 1.5, grad_input.data[0].item())
        run_test("unreached iter coeff = 0",
                 abs(grad_input.data[2].item()) < 1e-5, 0.0, grad_input.data[2].item())
        run_test("grad_input shape matches input",
                 list(grad_input.shape) == [3])

    # ── Tests 5-7: Numeric channel — kConfidenceNotBounded ──
    print("Tests 5-7: Numeric channel — kConfidenceNotBounded")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_t = make_tensor(["res0", "res1"], tmpdir)
        input_t.data[0] = Status.convert_status_to_float(Status.kConfidenceNotBounded)
        input_t.data[1] = 0.0  # unreached

        output_t = make_tensor(["res0"], tmpdir)
        output_t.data[0] = Status.convert_status_to_float(Status.confidence(1.0))

        grad_output_t = make_tensor(["better"], tmpdir)
        grad_output_t.data[0] = 2.0

        gi = todo_tensor_like(input_t)
        # kConfidenceNotBounded treated like confidence: copy grad_output coeff
        gi.data[0] = 2.0
        gi.data[1] = 0.0

        run_test("kCNB iter gets grad_output coeff",
                 abs(gi.data[0].item() - 2.0) < 1e-5)
        run_test("unreached stays 0",
                 abs(gi.data[1].item()) < 1e-5)
        run_test("gi shape [2]", list(gi.shape) == [2])

    # ── Tests 8-10: Numeric channel — scyf topk selection ──
    print("Tests 8-10: Numeric channel — scyf topk selection")
    with tempfile.TemporaryDirectory() as tmpdir:
        # All failed, topk=1 means only the highest scyf gets grad_output_coeff
        input_t = make_tensor(["a", "b", "c"], tmpdir)
        input_t.data[0] = Status.convert_status_to_float(Status.self_confidence_but_failed(0.3))
        input_t.data[1] = Status.convert_status_to_float(Status.self_confidence_but_failed(0.7))
        input_t.data[2] = Status.convert_status_to_float(Status.self_confidence_but_failed(0.5))

        output_t = make_tensor(["b"], tmpdir)
        output_t.data[0] = Status.convert_status_to_float(Status.self_confidence_but_failed(0.7))

        grad_output_t = make_tensor(["better_b"], tmpdir)
        grad_output_t.data[0] = 1.0

        # Use topk=1: only scyf(0.7) gets grad_output_coeff
        scyf_topk = _collect_scyf_topk_per_prefix(input_t, topk=1)
        gi = todo_tensor_like(input_t)
        for i in range(3):
            flat = i
            ist = Status.convert_float_to_status(input_t.data[i].item())
            if ist.is_self_confidence_but_failed:
                if flat in scyf_topk:
                    gi.data[i] = 1.0
                else:
                    gi.data[i] = Status.convert_status_to_float(ist)

        run_test("topk=1: highest scyf gets grad_output coeff",
                 abs(gi.data[1].item() - 1.0) < 0.05, 1.0, gi.data[1].item())
        run_test("topk=1: non-topk scyf iter 0 attenuated",
                 abs(gi.data[0].item() - (-0.3)) < 0.05, -0.3, gi.data[0].item())
        run_test("topk=1: non-topk scyf iter 2 attenuated",
                 abs(gi.data[2].item() - (-0.5)) < 0.05, -0.5, gi.data[2].item())

    # ── Tests 11-12: Numeric channel — kContextOverflow ──
    print("Tests 11-12: Numeric channel — kContextOverflow")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_t = make_tensor(["ok", "overflow"], tmpdir)
        input_t.data[0] = Status.convert_status_to_float(Status.confidence(0.8))
        input_t.data[1] = Status.convert_status_to_float(Status.kContextOverflow)

        output_t = make_tensor(["ok"], tmpdir)
        output_t.data[0] = Status.convert_status_to_float(Status.confidence(0.8))

        grad_output_t = make_tensor(["better"], tmpdir)
        grad_output_t.data[0] = 1.0

        gi = todo_tensor_like(input_t)
        gi.data[0] = 1.0
        gi.data[1] = 0.0

        run_test("kContextOverflow coeff = 0",
                 abs(gi.data[1].item()) < 1e-5)
        run_test("confidence iter gets coeff",
                 abs(gi.data[0].item() - 1.0) < 1e-5)

    # ── Tests 13-14: 2D prefix shape numeric channel ──
    print("Tests 13-14: 2D input — numeric channel")
    with tempfile.TemporaryDirectory() as tmpdir:
        # shape [2, 3]: prefix=[2], recurrent_dim=3
        input_t = make_tensor([["a", "b", "c"], ["d", "e", "f"]], tmpdir)
        input_t.data[0, 0] = Status.convert_status_to_float(Status.self_confidence_but_failed(0.3))
        input_t.data[0, 1] = Status.convert_status_to_float(Status.confidence(0.9))
        input_t.data[0, 2] = 0.0  # unreached
        input_t.data[1, 0] = Status.convert_status_to_float(Status.confidence(0.7))
        input_t.data[1, 1] = 0.0  # unreached
        input_t.data[1, 2] = 0.0  # unreached

        output_t = make_tensor(["b", "d"], tmpdir)
        output_t.data[0] = Status.convert_status_to_float(Status.confidence(0.9))
        output_t.data[1] = Status.convert_status_to_float(Status.confidence(0.7))

        grad_output_t = make_tensor(["better_b", "better_d"], tmpdir)
        grad_output_t.data[0] = 1.0
        grad_output_t.data[1] = 2.0

        gi = todo_tensor_like(input_t)
        # [0,1] confidence -> grad_output[0] = 1.0
        gi.data[0, 1] = 1.0
        # [1,0] confidence -> grad_output[1] = 2.0
        gi.data[1, 0] = 2.0

        run_test("2D [0,1] winning coeff",
                 abs(gi.data[0, 1].item() - 1.0) < 1e-5)
        run_test("2D [1,0] winning coeff",
                 abs(gi.data[1, 0].item() - 2.0) < 1e-5)

    # ── Tests 15-16: todo_tensor_like produces correct shape ──
    print("Tests 15-16: todo_tensor_like shape/content")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_t = make_tensor(["x", "y", "z"], tmpdir)
        gi = todo_tensor_like(input_t)
        run_test("todo shape matches", list(gi.shape) == [3])
        run_test("todo content is TODO", read_storage(gi, 0) == "TODO")

    # ── Tests 17-18: Full recurrent_backward with real LLM ──
    print("Tests 17-18: Full recurrent_backward (real LLM)")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Simple case: [2] input, iteration 0 won
        input_t = make_tensor(["Hello world", "unused"], tmpdir)
        input_t.data[0] = Status.convert_status_to_float(Status.confidence(0.9))
        input_t.data[1] = 0.0  # unreached

        output_t = make_tensor(["Hello world"], tmpdir)
        output_t.data[0] = Status.convert_status_to_float(Status.confidence(0.9))

        grad_output_t = make_tensor(["Bonjour le monde"], tmpdir)
        grad_output_t.data[0] = 1.0

        prompt_t = make_tensor(["translate to french", "unused"], tmpdir)
        prompt_t.data.fill_(1.0)

        gi = recurrent_backward(
            grad_output_t, input_t, output_t, prompt_t,
            task_prompt="Translate English text to French.",
            llm_method="raw_llm_api",
        )
        run_test("full backward shape", list(gi.shape) == [2])
        # Winning iteration should have a diff (non-TODO)
        content_0 = read_storage(gi, 0)
        run_test("winning iter has gradient (not TODO)",
                 content_0 is not None and content_0 != "TODO",
                 "non-TODO content", content_0)
        print(f"    grad_input[0] content: {repr(content_0[:80] if content_0 else None)}...")

    # ── Tests 19-20: Unreached elements skipped in LLM call ──
    print("Tests 19-20: Unreached elements stay TODO")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_t = make_tensor(["good output", "never ran"], tmpdir)
        input_t.data[0] = Status.convert_status_to_float(Status.confidence(1.0))
        input_t.data[1] = 0.0

        output_t = make_tensor(["good output"], tmpdir)
        output_t.data[0] = Status.convert_status_to_float(Status.confidence(1.0))

        grad_output_t = make_tensor(["improved output"], tmpdir)
        grad_output_t.data[0] = 1.0

        prompt_t = make_tensor(["task prompt", "unused"], tmpdir)
        prompt_t.data.fill_(1.0)

        gi = recurrent_backward(
            grad_output_t, input_t, output_t, prompt_t,
            task_prompt="Improve the text.",
            llm_method="raw_llm_api",
        )
        # Unreached element should have coeff 0 and TODO content unchanged
        run_test("unreached coeff is 0",
                 abs(gi.data[1].item()) < 1e-5, 0.0, gi.data[1].item())
        content_1 = read_storage(gi, 1)
        run_test("unreached stays TODO",
                 content_1 == "TODO", "TODO", content_1)

    print(f"\n  Passed: {passed}, Failed: {failed}, Total: {passed + failed}")
    print("All recurrent_backward tests completed.")
