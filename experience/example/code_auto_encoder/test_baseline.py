"""Test baseline model for auto-encoder cloze experiment.

Generated from test_baseline.viba.

Viba DSL specification:
  kMaskedHint := "<AUTOENCODER-CLOZE-MASK-PLACEHOLDER>"

  test_baseline :=
    IterationList[$baseline_loss list[float]]
    <- $total_batch_size int # default 16
    <- $num_iterations int # default 1
    <- $llm_method str # default "raw_llm_api"
    <- $workspace_dir (str | None) # None means temp directory
    <- $dataset_dir <- { ./codebase/ }
    <- Import[./parepare_dataset]
    <- IterationList[int] <- range <- $num_iterations
    <- Import[./baseline_agent_model.viba].BaselineAgentModel <- $llm_method
    <- Import[function/get_edit_distance]
"""

import os
import sys
import tempfile
from typing import Optional, List

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experience.symbolic_tensor.function.get_edit_distance_ratio import get_edit_distance_ratio_impl

from experience.example.code_auto_encoder.prepare_dataset import parepare_dataset
from experience.example.code_auto_encoder.baseline_agent_model import BaselineAgentModel


# <- ($dataset_dir <- { ./codebase/ })
CODEBASE_DIR = os.path.join(os.path.dirname(__file__), "codebase")


def _read_storage(tensor, flat_index: int) -> str:
    """Read the content of a symbolic tensor element from disk."""
    digits = list(str(flat_index))
    path = os.path.join(
        tensor.st_relative_to, tensor.st_tensor_uid,
        "storage", os.path.join(*digits), "data",
    )
    with open(path) as f:
        return f.read()


def _run_single_iteration(
    iteration_idx: int,
    total_batch_size: int,
    dataset_dir: str,
    tmpdir: str,
    llm_method: str,
    interactive: bool = False,
    auto_confirm: bool = True,
    tmux_session: str = None,
) -> List[float]:
    """Run a single iteration of the baseline test."""
    print(f"\n{'='*50}")
    print(f"Iteration {iteration_idx + 1}")
    print(f"{'='*50}")

    # <- Import[./parepare_dataset]
    masked_path_tensor, masked_content_tensor, gt_tensor, file_info = parepare_dataset(
        total_batch_size, dataset_dir, tmpdir,
    )
    print(f"Batch={total_batch_size}, files={masked_path_tensor.shape[1]}")
    for i, info in enumerate(file_info):
        gt_preview = _read_storage(gt_tensor, i)[:60].replace("\n", "\\n")
        print(f"  [{i}] {info} -> {gt_preview}...")

    # <- Import[./baseline_agent_model.viba].BaselineAgentModel <- $llm_method
    mode_str = f"interactive={interactive}" if llm_method == "tmux_cc" else ""
    print(f"\nRunning BaselineAgentModel (llm_method={llm_method} {mode_str})...")
    model = BaselineAgentModel(
        llm_method=llm_method,
        interactive=interactive,
        auto_confirm=auto_confirm,
        tmux_session=tmux_session,
    )
    output = model(masked_path_tensor, masked_content_tensor)

    # <- Import[function/get_edit_distance]
    loss = get_edit_distance_ratio_impl(output, gt_tensor)

    # <- { print $baseline_output.st_uid ground_truth_content_tensor.st_uid }
    print(f"\noutput tensor uid: {output.st_tensor_uid}")
    print(f"ground_truth tensor uid: {gt_tensor.st_tensor_uid}")

    print(f"\nResults (edit_distance_ratio, lower=better):")
    baseline_loss = []
    for i in range(total_batch_size):
        actual = _read_storage(output, i)
        gt = _read_storage(gt_tensor, i)
        r = loss[i].item()
        baseline_loss.append(r)
        print(f"  [{i}] {file_info[i]}  loss={r:.4f}")
        print(f"       gt:     {gt[:80].replace(chr(10), '\\n')}")
        print(f"       actual: {actual[:80].replace(chr(10), '\\n')}")

    mean_loss = loss.float().mean().item()
    print(f"\nIteration {iteration_idx + 1} mean loss: {mean_loss:.4f}")
    return baseline_loss


def test_baseline(
    total_batch_size: int = 16,
    num_iterations: int = 1,
    llm_method: str = "raw_llm_api",
    workspace_dir: Optional[str] = None,
    interactive: bool = False,
    auto_confirm: bool = True,
    tmux_session: Optional[str] = None,
) -> List[List[float]]:
    """test_baseline from test_baseline.viba.

    Args:
        total_batch_size: default 16
        num_iterations: default 1
        llm_method: default "raw_llm_api", also supports "coding_agent" and "tmux_cc"
        workspace_dir: None means temp directory
        interactive: If True and llm_method="tmux_cc", run in tmux for visual observation.
            Use `tmux attach -t <session>` to watch ducc in real-time.
        auto_confirm: If True (and interactive), auto-confirm prompts in tmux.
        tmux_session: Custom tmux session name (interactive mode only).

    Returns:
        List of baseline_loss per iteration
    """
    # <- $dataset_dir <- { ./codebase/ }
    dataset_dir = os.path.realpath(CODEBASE_DIR)
    print(f"Dataset: {dataset_dir}")

    # <- $workspace_dir (str | None) # None means temp directory
    if workspace_dir is None:
        tmpdir = tempfile.mkdtemp()
    else:
        tmpdir = workspace_dir
        os.makedirs(tmpdir, exist_ok=True)

    # <- IterationList[int] <- range <- $num_iterations
    all_iteration_losses: List[List[float]] = []
    for iteration_idx in range(num_iterations):
        baseline_loss = _run_single_iteration(
            iteration_idx, total_batch_size, dataset_dir, tmpdir, llm_method,
            interactive=interactive,
            auto_confirm=auto_confirm,
            tmux_session=tmux_session,
        )
        all_iteration_losses.append(baseline_loss)

    # Summary
    if num_iterations > 1:
        print(f"\n{'='*50}")
        print(f"Summary ({num_iterations} iterations):")
        print(f"{'='*50}")
        mean_losses = [sum(l) / len(l) for l in all_iteration_losses]
        print(f"Mean losses per iteration: {['%.4f' % m for m in mean_losses]}")
        overall_mean = sum(mean_losses) / len(mean_losses)
        print(f"Overall mean loss: {overall_mean:.4f}")

    return all_iteration_losses


if __name__ == "__main__":
    import argparse
    import subprocess

    parser = argparse.ArgumentParser(description="Test baseline model for auto-encoder cloze experiment.")
    parser.add_argument(
        "--total-batch-size", type=int, default=16,
        help="Total batch size (default: 16)"
    )
    parser.add_argument(
        "--num-iterations", type=int, default=1,
        help="Number of iterations (default: 1)"
    )
    parser.add_argument(
        "--llm-method", type=str, default="raw_llm_api",
        choices=["raw_llm_api", "coding_agent", "tmux_cc"],
        help="LLM method to use (default: raw_llm_api)"
    )
    parser.add_argument(
        "--workspace-dir", type=str, default=None,
        help="Workspace directory (default: temp directory)"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Run in interactive tmux mode for visual observation (only for llm_method=tmux_cc)"
    )
    parser.add_argument(
        "--no-auto-confirm", action="store_true",
        help="Disable auto-confirm prompts in interactive mode (manual operation required)"
    )
    parser.add_argument(
        "--tmux-session", type=str, default=None,
        help="Custom tmux session name (only for interactive mode)"
    )

    args = parser.parse_args()

    # Setup environment for LLM API
    result = subprocess.run(
        ["bash", "-c", "source ~/.anthropic.sh && env"],
        capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            os.environ[key] = val
    os.environ.pop("CLAUDECODE", None)

    # Run the test
    test_baseline(
        total_batch_size=args.total_batch_size,
        num_iterations=args.num_iterations,
        llm_method=args.llm_method,
        workspace_dir=args.workspace_dir,
        interactive=args.interactive,
        auto_confirm=not args.no_auto_confirm,
        tmux_session=args.tmux_session,
    )
