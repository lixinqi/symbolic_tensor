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
)


def prepare_worktrees(
    total_batch_size: int,
    dataset_dir: str,
    tmpdir: str,
    cache_dir: Optional[str] = None,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Create worktree dirs with masked codebases.

    Returns:
        worktree_tensor: SymbolicTensor(batch,) containing worktree paths.
        gt_tensor: SymbolicTensor(batch,) containing ground truth.
        file_info: list of "file_path:start-end" for logging.
    """
    # Use baseline dataset prep to get masked data
    masked_path_tensor, masked_content_tensor, gt_tensor, file_info = parepare_dataset(
        total_batch_size, dataset_dir, tmpdir, cache_dir=cache_dir, seed=seed,
    )

    batch_size = masked_path_tensor.shape[0]
    num_files = masked_path_tensor.shape[1]

    worktree_paths = []

    for b in range(batch_size):
        # Create worktree dir
        worktree_dir = os.path.join(tmpdir, f"worktree_{b}")
        os.makedirs(worktree_dir, exist_ok=True)

        # Find which file has the mask and get its path
        target_file_idx = None
        target_file_path = None
        for f in range(num_files):
            path = _read_storage(masked_path_tensor, b * num_files + f)
            content = _read_storage(masked_content_tensor, b * num_files + f)
            # Write file to worktree
            full_path = os.path.join(worktree_dir, path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as fobj:
                fobj.write(content)
            if kMaskedHint in content:
                target_file_idx = f
                target_file_path = path
                # Count mask line
                lines = content.splitlines()
                mask_line = None
                for i, line in enumerate(lines):
                    if kMaskedHint in line:
                        mask_line = i
                        break

        # Parse mask range from file_info (format: "path:start-end")
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

        # Write .cloze_task.json
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
