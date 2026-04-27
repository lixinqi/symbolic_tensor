"""Test harness model for auto-encoder cloze experiment.

CLI runner that evaluates HarnessModel against baseline.
"""

import os
import sys
import tempfile
from typing import List, Optional

import torch

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from experience.symbolic_tensor.function.get_edit_distance_ratio import get_edit_distance_ratio_impl
from experience.example.code_auto_encoder.harness_model.prepare_worktrees import prepare_worktrees
from experience.example.code_auto_encoder.harness_model.harness_model import HarnessModel


def _read_storage(tensor, flat_index: int) -> str:
    digits = list(str(flat_index))
    path = os.path.join(
        tensor.st_relative_to, tensor.st_tensor_uid,
        "storage", os.path.join(*digits), "data",
    )
    with open(path) as f:
        return f.read()


def test_harness(
    total_batch_size: int = 1,
    seed: int = 42,
    llm_method: str = "raw_llm_api",
    max_codegen_steps: int = 4,
    max_tool_call_retries: int = 5,
    topk: int = 2,
    dataset_dir: Optional[str] = None,
) -> List[float]:
    """Run harness model test.

    Returns:
        List of loss values per sample.
    """
    if dataset_dir is None:
        dataset_dir = os.path.join(os.path.dirname(__file__), "..", "codebase")
    dataset_dir = os.path.realpath(dataset_dir)

    tmpdir = tempfile.mkdtemp()
    print(f"Temp dir: {tmpdir}")
    print(f"Dataset: {dataset_dir}")

    # Prepare worktrees
    worktree_tensor, gt_tensor, file_info = prepare_worktrees(
        total_batch_size, dataset_dir, tmpdir, seed=seed,
    )
    print(f"Batch={total_batch_size}, worktrees prepared")
    for i, info in enumerate(file_info):
        gt_preview = _read_storage(gt_tensor, i)[:60].replace("\n", "\\n")
        print(f"  [{i}] {info} -> {gt_preview}...")

    # Run harness model
    print(f"\nRunning HarnessModel (llm_method={llm_method})...")
    model = HarnessModel(
        max_codegen_steps=max_codegen_steps,
        max_tool_call_retries=max_tool_call_retries,
        topk=topk,
        llm_method=llm_method,
    )
    output = model(worktree_tensor)

    # Compute loss
    loss = get_edit_distance_ratio_impl(output, gt_tensor)

    print(f"\noutput tensor uid: {output.st_tensor_uid}")
    print(f"ground_truth tensor uid: {gt_tensor.st_tensor_uid}")

    print(f"\nResults (edit_distance_ratio, lower=better):")
    harness_loss = []
    for i in range(total_batch_size):
        actual = _read_storage(output, i)
        gt = _read_storage(gt_tensor, i)
        r = loss[i].item()
        harness_loss.append(r)
        newline = '\n'
        print(f"  [{i}] {file_info[i]}  loss={r:.4f}")
        print(f"       gt:     {gt[:80].replace(newline, chr(92) + 'n')}")
        print(f"       actual: {actual[:80].replace(newline, chr(92) + 'n')}")

    mean_loss = loss.float().mean().item()
    print(f"\nMean loss: {mean_loss:.4f}")
    return harness_loss


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test harness model for auto-encoder cloze experiment.")
    parser.add_argument("--total-batch-size", type=int, default=1, help="Total batch size (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--llm-method", type=str, default="raw_llm_api", help="LLM method")
    parser.add_argument("--max-codegen-steps", type=int, default=4, help="Max code generation steps")
    parser.add_argument("--max-tool-call-retries", type=int, default=5, help="Max tool call retries")
    parser.add_argument("--topk", type=int, default=2, help="Top-k experience retrieval")

    args = parser.parse_args()

    # Setup environment
    from experience.llm_client.config import setup_env_for_method, get_config_summary
    setup_env_for_method(args.llm_method)
    print(f"[Config] {args.llm_method}: {get_config_summary(args.llm_method)}")

    test_harness(
        total_batch_size=args.total_batch_size,
        seed=args.seed,
        llm_method=args.llm_method,
        max_codegen_steps=args.max_codegen_steps,
        max_tool_call_retries=args.max_tool_call_retries,
        topk=args.topk,
    )
