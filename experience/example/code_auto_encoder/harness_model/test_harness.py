"""Test harness model for auto-encoder cloze experiment.

CLI runner that evaluates HarnessModel against baseline.
"""

import ast as _ast
import os
import sys
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional

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


def _parse_mask_type(file_info_entry: str) -> str:
    """Extract mask_type from file_info entry.

    Handles both formats:
      "path:start-end"            → "unknown"
      "path:start-end:mask_type"  → mask_type
    """
    mt = file_info_entry.rsplit(":", 1)[-1]
    return mt if mt in ("short", "long", "structural") else "unknown"


def _compute_extra_metrics(
    output_tensor,
    gt_tensor,
    file_info: List[str],
    harness_loss: List[float],
    total_batch_size: int,
) -> None:
    """Compute and print AST pass rate, Exact Match, and per-mask-type loss."""
    ast_pass: List[int] = []
    exact_match: List[int] = []
    type_losses: Dict[str, List[float]] = defaultdict(list)

    for i in range(total_batch_size):
        actual = _read_storage(output_tensor, i)
        gt = _read_storage(gt_tensor, i)
        mask_type = _parse_mask_type(file_info[i])

        try:
            _ast.parse(actual)
            ast_pass.append(1)
        except SyntaxError:
            ast_pass.append(0)

        exact_match.append(1 if actual == gt else 0)
        type_losses[mask_type].append(harness_loss[i])

    n = len(ast_pass)
    print(f"\nAST pass rate:   {sum(ast_pass)/n:.2%}  ({sum(ast_pass)}/{n})")
    print(f"Exact match rate: {sum(exact_match)/n:.2%}  ({sum(exact_match)}/{n})")

    if any(k != "unknown" for k in type_losses):
        print(f"\nStratified loss by mask_type:")
        for mt in ("short", "long", "structural", "unknown"):
            losses = type_losses.get(mt)
            if losses:
                print(f"  {mt:12s}: mean={sum(losses)/len(losses):.4f}  (n={len(losses)})")


def test_harness(
    total_batch_size: int = 1,
    seed: int = 42,
    llm_method: str = "raw_llm_api",
    max_codegen_steps: int = 4,
    max_context_collects: int = 5,
    max_tool_call_retries: int = 2,
    topk: int = 2,
    dataset_dir: Optional[str] = None,
    split: Optional[str] = None,
    dataset_index_dir: Optional[str] = None,
) -> List[float]:
    """Run harness model test.

    Args:
        split: "train" or "eval" to load from pre-generated index.
            When None, uses legacy random masking.
        dataset_index_dir: Path to dataset_index/ directory (auto-detected if None).

    Returns:
        List of loss values per sample.
    """
    if dataset_dir is None:
        dataset_dir = os.path.join(os.path.dirname(__file__), "..", "codebase")
    dataset_dir = os.path.realpath(dataset_dir)

    tmpdir = tempfile.mkdtemp()
    print(f"Temp dir: {tmpdir}")
    print(f"Dataset: {dataset_dir}")
    if split:
        print(f"Split: {split}")

    worktree_tensor, gt_tensor, file_info = prepare_worktrees(
        total_batch_size, dataset_dir, tmpdir,
        seed=seed,
        split=split,
        dataset_index_dir=dataset_index_dir,
    )
    print(f"Batch={total_batch_size}, worktrees prepared")
    for i, info in enumerate(file_info):
        gt_preview = _read_storage(gt_tensor, i)[:60].replace("\n", "\\n")
        print(f"  [{i}] {info} -> {gt_preview}...")

    print(f"\nRunning HarnessModel (llm_method={llm_method})...")
    model = HarnessModel(
        max_codegen_steps=max_codegen_steps,
        max_context_collects=max_context_collects,
        max_tool_call_retries=max_tool_call_retries,
        topk=topk,
        llm_method=llm_method,
    )
    output = model(worktree_tensor)

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

    _compute_extra_metrics(output, gt_tensor, file_info, harness_loss, total_batch_size)

    return harness_loss


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test harness model for auto-encoder cloze experiment.")
    parser.add_argument("--total-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llm-method", type=str, default="raw_llm_api")
    parser.add_argument("--max-codegen-steps", type=int, default=4)
    parser.add_argument("--max-context-collects", type=int, default=5)
    parser.add_argument("--max-tool-call-retries", type=int, default=2)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument(
        "--split", type=str, default=None, choices=["train", "eval"],
        help="Load samples from pre-generated index (train or eval split). "
             "Omit to use legacy random masking.",
    )
    parser.add_argument(
        "--dataset-index-dir", type=str, default=None,
        help="Path to dataset_index/ directory. Auto-detected if not specified.",
    )

    args = parser.parse_args()

    from experience.llm_client.config import setup_env_for_method, get_config_summary
    setup_env_for_method(args.llm_method)
    print(f"[Config] {args.llm_method}: {get_config_summary(args.llm_method)}")

    test_harness(
        total_batch_size=args.total_batch_size,
        seed=args.seed,
        llm_method=args.llm_method,
        max_codegen_steps=args.max_codegen_steps,
        max_context_collects=args.max_context_collects,
        max_tool_call_retries=args.max_tool_call_retries,
        topk=args.topk,
        split=args.split,
        dataset_index_dir=args.dataset_index_dir,
    )
