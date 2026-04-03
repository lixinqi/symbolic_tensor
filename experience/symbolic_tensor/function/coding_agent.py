"""Coding agent: input symbolic tensor → prompt → LLM → output symbolic tensor.

Generated from coding_agent.viba.

Viba DSL specification:
  coding_agent :=
    $output Symbolic[torch.Tensor[($batch_size, )]]
    <- $input Symbolic[torch.Tensor[($batch_size, $num_files)]] | Symbolic[torch.Tensor[($batch_size,)]]
    <- $output_prompt ForwardPromptCallable # default None
    <- $task_prompt str # default ""
    <- $llm_method ("coding_agent" | "raw_llm_api") # default raw_llm_api
    <- $llm_env dict[str, str] # default None
"""

import os
import tempfile
import shutil
import torch
from typing import Callable, Dict, List, Optional

from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.tensor_util.slice_view import slice_view
from experience.symbolic_tensor.tensor_util.slice_tensor import slice_tensor
from experience.symbolic_tensor.tensor_util.dump_view import dump_view
from experience.llm_client.agent_task import AgentTask
from experience.llm_client.task_handler import TaskHandler


def _copy_back_to_storage_view(mutable_dir: str, view_tensor: torch.Tensor) -> None:
    """Copy LLM results from mutable workspace dir back through view tensor's symlinks."""
    flat_index = 0
    digits = list(str(flat_index))
    view_storage_path = os.path.join(
        view_tensor.st_relative_to,
        view_tensor.st_tensor_uid,
        "storage",
        os.path.join(*digits),
        "data",
    )
    real_storage_path = os.path.realpath(view_storage_path)
    mutable_file = os.path.join(mutable_dir, "data.txt")
    if os.path.isfile(mutable_file):
        with open(mutable_file, "r", encoding="utf-8") as f:
            content = f.read()
        with open(real_storage_path, "w", encoding="utf-8") as f:
            f.write(content)


def default_prompt_for_output(
    task_prompt: str,
    workspace_dir: str,
    const_input_view: str,
    mutable_output_dir: str,
) -> str:
    return (
        "You are a coding agent.\n\n"
        f"{task_prompt}\n\n"
        f"Input: \"{const_input_view}\"\n"
        f"Output: \"{mutable_output_dir}\"\n\n"
        f"Replace TODO in \"{mutable_output_dir}\" with result.\n"
    )


def coding_agent(
    input: torch.Tensor,
    output_prompt: Optional[Callable[..., str]] = None,
    task_prompt: str = "",
    llm_method: str = "raw_llm_api",
    llm_env: Optional[Dict[str, str]] = None,
) -> torch.Tensor:
    """Forward pass of coding agent: input → prompt → LLM → output.

    Mini version of st_moe_forward without autograd or experience.

    Args:
        input: A symbolic tensor of shape (batch_size,) or (batch_size, num_files).
        output_prompt: Callable(task_prompt, workspace_dir, const_input_view,
            mutable_output_dir) -> str. None uses default_prompt_for_output.
        task_prompt: High-level task description.
        llm_method: LLM backend ("coding_agent" or "raw_llm_api").
        llm_env: Optional environment variables for LLM.

    Returns:
        output: A symbolic tensor of shape (batch_size,).
    """
    batch_size = input.shape[0]
    input_ndim = input.dim()

    # <- ($output Symbolic[torch.Tensor] <- Import[make_todo_tensor] <- $shape ($batch_size, ))
    # Create output tensor of shape (batch_size,) with "TODO" placeholders
    tmpdir = input.st_relative_to
    output = make_tensor(["TODO"] * batch_size, tmpdir)

    all_tasks: List[AgentTask] = []
    all_copyback_info: List[tuple] = []

    # <- (list[$row_input_view] <- list[$slice_view] <- $input)
    # <- (list[$row_output_view] <- list[$slice_view] <- $output)
    for row_idx in range(batch_size):
        # Slice input: handle both 1D (batch,) and 2D (batch, num_files)
        if input_ndim == 1:
            # 1D input: slice with single index
            row_input_view = slice_view(input, [row_idx])
        else:
            # 2D input: slice row with [row_idx, slice(None)]
            row_input_view = slice_view(input, [row_idx, slice(None)])
        # Slice output row: shape ()
        row_output_view = slice_view(output, [row_idx])
        row_output_value = slice_tensor(output, [row_idx])

        # <- ($workspace_dir str <- TemporaryDirectory)
        workspace_dir = tempfile.mkdtemp()
        input_view_dir = os.path.join(workspace_dir, "const_input_view")
        output_dir = os.path.join(workspace_dir, "mutable_output_dir")

        # <- (void <- $dump_view <- $row_input_view <- f"{workspace_dir}/const_input_view")
        # <- (void <- $dump_view <- $row_output_value <- f"{workspace_dir}/mutable_output_dir")
        dump_view(row_input_view, input_view_dir, "txt")
        dump_view(row_output_value, output_dir, "txt")

        # <- ($prompt str <- ($output_prompt | default_prompt_for_output) ...)
        prompt = (output_prompt or default_prompt_for_output)(
            task_prompt, workspace_dir, input_view_dir, output_dir,
        )

        # <- ($agent_task AgentTask <- $workspace_dir <- "mutable_output_dir" <- $prompt)
        all_tasks.append(AgentTask(
            workspace_dir=workspace_dir,
            output_relative_dir="mutable_output_dir",
            prompt=prompt,
        ))
        all_copyback_info.append((output_dir, row_output_view))

    # <- (void <- Import[task_handler] <- $all_tasks <- $llm_method <- $llm_env)
    TaskHandler()(all_tasks, llm_method, llm_env=llm_env)

    # <- (void <- copy_back_to_storage_view <- f"{workspace_dir}/mutable_output_dir" <- $row_output_view)
    for output_dir, row_output_view in all_copyback_info:
        _copy_back_to_storage_view(output_dir, row_output_view)

    # Cleanup
    for task in all_tasks:
        shutil.rmtree(task.workspace_dir, ignore_errors=True)

    return output


if __name__ == "__main__":
    import subprocess
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor

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

    print("Running coding_agent tests...\n")

    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    def read_output(tensor, flat_index):
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

    # Test 1: 2D input → 1D output (raw_llm_api)
    print("Test 1: 2D input → 1D output (llm_method=raw_llm_api)")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_data = [["Write hello world"], ["Write goodbye world"]]
        input_tensor = make_tensor(input_data, tmpdir)
        print(f"  Input shape: {list(input_tensor.shape)}")

        output = coding_agent(
            input_tensor,
            task_prompt="You are a Python expert. Write concise code.",
            llm_method="raw_llm_api",
        )

        print(f"  Output shape: {list(output.shape)}")
        run_test("Output shape is (2,)", list(output.shape) == [2])
        for i in range(output.numel()):
            content = read_output(output, i)
            run_test(f"Output[{i}] exists", content is not None)
            if content:
                run_test(f"Output[{i}] not TODO", "TODO" not in content)
                print(f"  Output[{i}]: {repr(content[:50])}")

    # Test 2: 2D input → 1D output (coding_agent)
    print("Test 2: 2D input → 1D output (llm_method=coding_agent)")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_data = [["Write hello world"], ["Write goodbye world"]]
        input_tensor = make_tensor(input_data, tmpdir)
        print(f"  Input shape: {list(input_tensor.shape)}")

        output = coding_agent(
            input_tensor,
            task_prompt="You are a Python expert. Write concise code.",
            llm_method="coding_agent",
        )

        print(f"  Output shape: {list(output.shape)}")
        run_test("Output shape is (2,)", list(output.shape) == [2])
        for i in range(output.numel()):
            content = read_output(output, i)
            run_test(f"Output[{i}] exists", content is not None)
            if content:
                run_test(f"Output[{i}] not TODO", "TODO" not in content)
                print(f"  Output[{i}]: {repr(content[:50])}")

    print("\nAll tests completed.")
