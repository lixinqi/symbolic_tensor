"""Prepare worktree directories for harness model.

Each worktree contains the codebase with one file masked,
plus a .cloze_task.json with task metadata.
"""

import json
import os
import random
import shutil
import torch
from typing import List, Optional, Tuple

from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.example.code_auto_encoder.baseline.prepare_dataset import (
    parepare_dataset,
    kMaskedHint,
    _fetch_all_files,
    _apply_mask,
)


def _prepare_worktrees_from_index(
    samples: List[dict],
    dataset_dir: str,
    tmpdir: str,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Build worktrees from pre-generated sample index entries.

    Each sample dict has: file_path, mask_start, mask_end, ground_truth, mask_type.
    Returns (worktree_tensor, gt_tensor, file_info) where file_info entries are
    formatted as "rel_path:start-end:mask_type".
    """
    all_files = _fetch_all_files(dataset_dir)

    worktree_paths = []
    gt_texts = []
    file_info = []

    for b, sample in enumerate(samples):
        target_file = sample["file_path"]
        mask_start = sample["mask_start"]
        mask_end = sample["mask_end"]
        gt = sample["ground_truth"]
        mask_type = sample.get("mask_type", "unknown")

        worktree_dir = os.path.join(tmpdir, f"worktree_{b}")
        os.makedirs(worktree_dir, exist_ok=True)

        mask_line = None
        for rel_path, content in all_files:
            if rel_path == target_file:
                write_content = _apply_mask(content, mask_start, mask_end)
                for i, line in enumerate(write_content.splitlines()):
                    if kMaskedHint in line:
                        mask_line = i
                        break
            else:
                write_content = content

            full_path = os.path.join(worktree_dir, rel_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as fobj:
                fobj.write(write_content)

        task = {
            "target_file": target_file,
            "mask_hint": kMaskedHint,
            "mask_line": mask_line,
            "mask_start_1idx": mask_start + 1,
            "mask_end_1idx": mask_end,
            "dataset_dir": dataset_dir,
            "mask_type": mask_type,
        }
        with open(os.path.join(worktree_dir, ".cloze_task.json"), "w") as f:
            json.dump(task, f, indent=2)

        worktree_paths.append(worktree_dir)
        gt_texts.append(gt)
        file_info.append(f"{target_file}:{mask_start + 1}-{mask_end}:{mask_type}")

    worktree_tensor = make_tensor(worktree_paths, tmpdir)
    gt_tensor = make_tensor(gt_texts, tmpdir)
    return worktree_tensor, gt_tensor, file_info


def prepare_worktrees(
    total_batch_size: int,
    dataset_dir: str,
    tmpdir: str,
    cache_dir: Optional[str] = None,
    seed: Optional[int] = None,
    split: Optional[str] = None,
    dataset_index_dir: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Create worktree dirs with masked codebases.

    Args:
        total_batch_size: Number of samples.
        dataset_dir: Root codebase directory.
        tmpdir: Temp directory for tensor storage.
        cache_dir: Cache dir for legacy random mode.
        seed: Random seed.
        split: "train" or "eval" to load from pre-generated index.
            When None, falls back to legacy random masking.
        dataset_index_dir: Path to dataset_index/ directory.
            Defaults to <parent of dataset_dir>/dataset_index.

    Returns:
        worktree_tensor: SymbolicTensor(batch,) of worktree paths.
        gt_tensor: SymbolicTensor(batch,) of ground truth strings.
        file_info: list of "rel_path:start-end" or "rel_path:start-end:mask_type".
    """
    if split is not None:
        if dataset_index_dir is None:
            dataset_index_dir = os.path.join(
                os.path.dirname(os.path.realpath(dataset_dir)), "dataset_index"
            )

        samples_path = os.path.join(dataset_index_dir, split, "samples.json")
        if not os.path.exists(samples_path):
            raise FileNotFoundError(
                f"Dataset index not found: {samples_path}\n"
                f"Run prepare_dataset_index.py first."
            )

        with open(samples_path) as f:
            all_samples = json.load(f)

        rng = random.Random(seed)
        if total_batch_size <= len(all_samples):
            selected = rng.sample(all_samples, total_batch_size)
        else:
            selected = [rng.choice(all_samples) for _ in range(total_batch_size)]

        return _prepare_worktrees_from_index(selected, dataset_dir, tmpdir)

    # Legacy random mode
    masked_path_tensor, masked_content_tensor, gt_tensor, file_info = parepare_dataset(
        total_batch_size, dataset_dir, tmpdir, cache_dir=cache_dir, seed=seed,
    )

    batch_size = masked_path_tensor.shape[0]
    num_files = masked_path_tensor.shape[1]

    worktree_paths = []

    for b in range(batch_size):
        worktree_dir = os.path.join(tmpdir, f"worktree_{b}")
        os.makedirs(worktree_dir, exist_ok=True)

        target_file_idx = None
        target_file_path = None
        for f in range(num_files):
            path = _read_storage(masked_path_tensor, b * num_files + f)
            content = _read_storage(masked_content_tensor, b * num_files + f)
            full_path = os.path.join(worktree_dir, path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as fobj:
                fobj.write(content)
            if kMaskedHint in content:
                target_file_idx = f
                target_file_path = path
                lines = content.splitlines()
                mask_line = None
                for i, line in enumerate(lines):
                    if kMaskedHint in line:
                        mask_line = i
                        break

        info = file_info[b]
        mask_start_1idx = None
        mask_end_1idx = None
        try:
            range_part = info.rsplit(":", 1)[1]
            start_str, end_str = range_part.split("-")
            mask_start_1idx = int(start_str)
            mask_end_1idx = int(end_str)
        except (ValueError, IndexError):
            pass

        task = {
            "target_file": target_file_path,
            "mask_hint": kMaskedHint,
            "mask_line": mask_line,
            "mask_start_1idx": mask_start_1idx,
            "mask_end_1idx": mask_end_1idx,
            "dataset_dir": dataset_dir,
        }
        with open(os.path.join(worktree_dir, ".cloze_task.json"), "w") as f:
            json.dump(task, f, indent=2)

        worktree_paths.append(worktree_dir)

    worktree_tensor = make_tensor(worktree_paths, tmpdir)
    return worktree_tensor, gt_tensor, file_info


def _read_storage(tensor, flat_index: int) -> str:
    digits = list(str(flat_index))
    path = os.path.join(
        tensor.st_relative_to, tensor.st_tensor_uid,
        "storage", os.path.join(*digits), "data",
    )
    with open(path) as f:
        return f.read()
