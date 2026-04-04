"""Prepare dataset for auto-encoder cloze experiment.

Generated from prepare_dataset.viba.

Viba DSL specification:
  parepare_dataset :=
    ($masked_file_path_tensor, $masked_file_content_tensor, $ground_truth_content_tensor)
    <- $total_batch_size int <- $dataset_dir
    <- { the range size should be in range (5, 10) }
"""

import os
import random
import torch
from typing import List, Tuple

from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor


kMaskedHint = "<AUTOENCODER-CLOZE-MASK-PLACEHOLDER>"


def _fetch_all_files(root_dir: str) -> List[Tuple[str, str]]:
    """Fetch all .py files as (relative_path, content)."""
    files = []
    for dirpath, _dirs, filenames in os.walk(root_dir):
        for fname in sorted(filenames):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fname)
            rel = os.path.relpath(fpath, root_dir)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
            except (UnicodeDecodeError, OSError):
                continue
            if content.strip():
                files.append((rel, content))
    return sorted(files)


def _find_maskable_range(content: str) -> Tuple[int, int]:
    """Find the range of lines eligible for masking: skip import lines."""
    lines = content.splitlines(keepends=True)
    main_start = len(lines)
    for i, line in enumerate(lines):
        if line.strip().startswith("if __name__"):
            main_start = i
            break
    code_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("import ") and not stripped.startswith("from "):
            code_start = i
            break
    return code_start, main_start


def _get_random_mask_range(content: str, min_size: int = 5, max_size: int = 10) -> Tuple[int, int, str]:
    """Pick random line range to mask. Range size in (min_size, max_size)."""
    lines = content.splitlines(keepends=True)
    code_start, main_start = _find_maskable_range(content)
    maskable_len = main_start - code_start

    if maskable_len < min_size:
        mask_len = max(1, maskable_len)
        start = code_start
    else:
        upper_bound = min(max_size, maskable_len)
        lower_bound = min(min_size, upper_bound)
        mask_len = random.randint(lower_bound, upper_bound)
        start = random.randint(code_start, main_start - mask_len)

    end = start + mask_len
    return start, end, "".join(lines[start:end])


def _apply_mask(content: str, start: int, end: int) -> str:
    """Replace lines [start, end) with kMaskedHint."""
    lines = content.splitlines(keepends=True)
    return "".join(lines[:start] + [kMaskedHint + "\n"] + lines[end:])


def parepare_dataset(
    total_batch_size: int,
    dataset_dir: str,
    tmpdir: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """PrepareDataSet from prepare_dataset.viba.

    Returns:
        masked_file_path_tensor:    Symbolic[Tensor(batch, num_files)]
        masked_file_content_tensor: Symbolic[Tensor(batch, num_files)]
        ground_truth_tensor:        Symbolic[Tensor(batch,)]
        file_info: list of "file_path:start-end" for logging
    """
    # <- ($files list[($file_path str, $file_content)] <- { fetch all files })
    files = _fetch_all_files(dataset_dir)

    # Filter files with maskable range >= 5 lines
    files = [(fp, fc) for fp, fc in files
             if _find_maskable_range(fc)[1] - _find_maskable_range(fc)[0] >= 5]
    num_files = len(files)
    assert num_files > 0, f"No sufficiently large .py files in {dataset_dir}"

    # <- {num_files = len(files)}
    file_paths = [fp for fp, _ in files]
    file_contents = [fc for _, fc in files]

    all_masked_paths = []
    all_masked_contents = []
    all_ground_truths = []
    file_info = []

    # <- (list[($random_file_index int, $random_range_line_start int, $random_range_line_end int)]
    #       <- { get multiple random ranges, range size in (5, 10) } <- $total_batch_size <- $num_files)
    for _ in range(total_batch_size):
        file_idx = random.randint(0, num_files - 1)
        fp, fc = files[file_idx]
        start, end, ground_truth = _get_random_mask_range(fc, min_size=5, max_size=10)
        masked_content = _apply_mask(fc, start, end)

        batch_paths = list(file_paths)
        batch_contents = []
        for f_i, fc_i in enumerate(file_contents):
            if f_i == file_idx:
                # Masked file: full content with mask
                batch_contents.append(masked_content)
            else:
                # Non-masked: full content
                batch_contents.append(fc_i)

        all_masked_paths.append(batch_paths)
        all_masked_contents.append(batch_contents)
        all_ground_truths.append(ground_truth)
        file_info.append(f"{fp}:{start+1}-{end}")

    # Create symbolic tensors
    masked_path_tensor = make_tensor(all_masked_paths, tmpdir)
    masked_content_tensor = make_tensor(all_masked_contents, tmpdir)
    gt_tensor = make_tensor(all_ground_truths, tmpdir)

    return masked_path_tensor, masked_content_tensor, gt_tensor, file_info


if __name__ == "__main__":
    import tempfile
    tmpdir = tempfile.mkdtemp()
    dataset_dir = os.path.join(os.path.dirname(__file__), "codebase")
    paths, contents, gt, info = parepare_dataset(2, dataset_dir, tmpdir)
    print(f"Prepared {len(info)} samples")
    for i, inf in enumerate(info):
        print(f"  [{i}] {inf}")
