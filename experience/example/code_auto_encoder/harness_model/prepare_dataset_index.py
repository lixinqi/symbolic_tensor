"""Build train/eval dataset index for harness model cloze experiment.

Run once to generate a dataset_index/ directory with pre-computed file-level
train/eval splits and three mask strategies (short, long, structural).

Usage:
    python -m experience.example.code_auto_encoder.harness_model.prepare_dataset_index
    python -m experience.example.code_auto_encoder.harness_model.prepare_dataset_index \
        --codebase-dir .../codebase --output-dir .../dataset_index
"""

import ast
import json
import os
import random
import sys
from typing import List, Optional, Tuple

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from experience.example.code_auto_encoder.baseline.prepare_dataset import (
    _fetch_all_files,
    _find_maskable_range,
    _build_comment_free_line_map,
    _get_random_mask_range,
)

_INDEX_VERSION = 1


def _find_structural_masks(
    content: str,
    min_body_lines: int = 3,
    max_masks: int = 5,
) -> List[Tuple[int, int, str]]:
    """Find function/method body line ranges using AST.

    Returns list of (start, end, ground_truth) where start/end are 0-indexed
    with end exclusive, covering the function body (after any leading docstring).
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    lines = content.splitlines(keepends=True)
    results = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        body = node.body
        if not body:
            continue

        # Skip leading docstring to find real code start
        first_stmt = body[0]
        if (
            isinstance(first_stmt, ast.Expr)
            and isinstance(getattr(first_stmt, "value", None), ast.Constant)
            and isinstance(first_stmt.value.value, str)
        ):
            if len(body) < 2:
                continue
            first_stmt = body[1]

        start = first_stmt.lineno - 1  # 0-indexed inclusive
        end = node.end_lineno          # 0-indexed exclusive (= 1-indexed inclusive)

        if end - start < min_body_lines:
            continue

        start = max(0, start)
        end = min(len(lines), end)

        gt = "".join(lines[start:end])
        if not gt.strip():
            continue

        results.append((start, end, gt))
        if len(results) >= max_masks:
            break

    return results


def _generate_samples_for_file(
    file_path: str,
    content: str,
    short_per_file: int,
    long_per_file: int,
    max_structural_per_file: int,
) -> List[dict]:
    """Generate all mask samples for one file using global random state."""
    samples = []

    # short masks: 1-3 lines
    for _ in range(short_per_file):
        try:
            start, end, gt = _get_random_mask_range(content, min_size=1, max_size=3)
            if gt.strip():
                samples.append({
                    "file_path": file_path,
                    "mask_start": start,
                    "mask_end": end,
                    "ground_truth": gt,
                    "mask_type": "short",
                })
        except Exception:
            pass

    # long masks: 4-8 lines
    for _ in range(long_per_file):
        try:
            start, end, gt = _get_random_mask_range(content, min_size=4, max_size=8)
            if gt.strip():
                samples.append({
                    "file_path": file_path,
                    "mask_start": start,
                    "mask_end": end,
                    "ground_truth": gt,
                    "mask_type": "long",
                })
        except Exception:
            pass

    # structural masks: entire function body
    structural = _find_structural_masks(
        content, min_body_lines=3, max_masks=max_structural_per_file
    )
    for start, end, gt in structural:
        samples.append({
            "file_path": file_path,
            "mask_start": start,
            "mask_end": end,
            "ground_truth": gt,
            "mask_type": "structural",
        })

    return samples


def build_dataset_index(
    codebase_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    short_per_file: int = 2,
    long_per_file: int = 2,
    max_structural_per_file: int = 3,
    force: bool = False,
) -> None:
    """Build and persist train/eval split index.

    Args:
        codebase_dir: Root directory containing .py source files.
        output_dir: Directory to write dataset_index files.
        train_ratio: Fraction of files assigned to train split.
        seed: Random seed for reproducibility.
        short_per_file: Number of short-mask samples per file.
        long_per_file: Number of long-mask samples per file.
        max_structural_per_file: Max structural-mask samples per file.
        force: Overwrite existing index.
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.json")

    if os.path.exists(metadata_path) and not force:
        print(f"Index already exists at {output_dir}. Use --force to regenerate.")
        with open(metadata_path) as f:
            meta = json.load(f)
        print(f"  train samples: {meta['num_train_samples']}, "
              f"eval samples: {meta['num_eval_samples']}")
        return

    # Collect eligible files (non-empty, have maskable lines)
    all_files = _fetch_all_files(codebase_dir)
    eligible = []
    for fp, fc in all_files:
        cs, me = _find_maskable_range(fc)
        if me - cs >= 1:
            eligible.append((fp, fc))

    if not eligible:
        raise ValueError(f"No eligible .py files in {codebase_dir}")

    print(f"Found {len(eligible)} eligible files ({len(all_files)} total)")

    # File-level split — seed global random for full reproducibility
    random.seed(seed)
    indices = list(range(len(eligible)))
    random.shuffle(indices)
    split_point = int(len(indices) * train_ratio)
    train_indices = indices[:split_point]
    eval_indices = indices[split_point:]

    print(f"Train files: {len(train_indices)}, eval files: {len(eval_indices)}")

    # Generate samples (global random state continues from shuffle)
    train_samples: List[dict] = []
    eval_samples: List[dict] = []

    for idx in train_indices:
        fp, fc = eligible[idx]
        train_samples.extend(
            _generate_samples_for_file(fp, fc, short_per_file, long_per_file, max_structural_per_file)
        )

    for idx in eval_indices:
        fp, fc = eligible[idx]
        eval_samples.extend(
            _generate_samples_for_file(fp, fc, short_per_file, long_per_file, max_structural_per_file)
        )

    print(f"Train samples: {len(train_samples)}, eval samples: {len(eval_samples)}")

    # Persist samples
    for split_name, samples in [("train", train_samples), ("eval", eval_samples)]:
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        with open(os.path.join(split_dir, "samples.json"), "w") as f:
            json.dump(samples, f, indent=2)

    # Persist metadata
    metadata = {
        "version": _INDEX_VERSION,
        "codebase_dir": os.path.realpath(codebase_dir),
        "train_ratio": train_ratio,
        "seed": seed,
        "num_train_files": len(train_indices),
        "num_eval_files": len(eval_indices),
        "num_train_samples": len(train_samples),
        "num_eval_samples": len(eval_samples),
        "short_per_file": short_per_file,
        "long_per_file": long_per_file,
        "max_structural_per_file": max_structural_per_file,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Index saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build dataset index for harness model.")
    parser.add_argument(
        "--codebase-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "codebase"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "dataset_index"),
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--short-per-file", type=int, default=2)
    parser.add_argument("--long-per-file", type=int, default=2)
    parser.add_argument("--max-structural-per-file", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    build_dataset_index(
        codebase_dir=os.path.realpath(args.codebase_dir),
        output_dir=os.path.realpath(args.output_dir),
        train_ratio=args.train_ratio,
        seed=args.seed,
        short_per_file=args.short_per_file,
        long_per_file=args.long_per_file,
        max_structural_per_file=args.max_structural_per_file,
        force=args.force,
    )
