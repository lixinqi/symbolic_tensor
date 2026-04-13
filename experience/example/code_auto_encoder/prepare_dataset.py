"""Prepare dataset for auto-encoder cloze experiment.

Generated from prepare_dataset.viba.

Viba DSL specification:
  parepare_dataset :=
    ($masked_file_path_tensor, $masked_file_content_tensor, $ground_truth_content_tensor)
    <- $total_batch_size int <- $dataset_dir
    <- { the range size should be in range (5, 10) }
"""

import json
import os
import random
import torch
from typing import List, Optional, Tuple

from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor


kMaskedHint = "<AUTOENCODER-CLOZE-MASK-PLACEHOLDER>"


def _get_seed_subdir(cache_dir: str, seed: Optional[int]) -> str:
    """Get the subdirectory for a specific seed."""
    seed_name = f"seed_{seed}" if seed is not None else "seed_none"
    return os.path.join(cache_dir, seed_name)


# =============================================================================
# New cache format (v2): indexed storage
#
# Structure:
#   seed_XX/
#   ├── metadata.json           # {dataset_dir, seed, num_files, cached_samples, version}
#   ├── source_files/
#   │   ├── index.json          # [file_path_0, file_path_1, ...]
#   │   └── contents/
#   │       ├── 0.txt           # content of file 0
#   │       ├── 1.txt
#   │       └── ...
#   └── samples/
#       ├── 0.json              # {file_idx, mask_start, mask_end, ground_truth}
#       ├── 1.json
#       └── ...
# =============================================================================

def _save_source_files(seed_dir: str, file_paths: List[str], file_contents: List[str]) -> None:
    """Save source files to cache (only once per seed)."""
    source_dir = os.path.join(seed_dir, "source_files")
    contents_dir = os.path.join(source_dir, "contents")
    os.makedirs(contents_dir, exist_ok=True)

    # Save file path index
    with open(os.path.join(source_dir, "index.json"), "w") as f:
        json.dump(file_paths, f, indent=2)

    # Save each file's content
    for i, content in enumerate(file_contents):
        with open(os.path.join(contents_dir, f"{i}.txt"), "w") as f:
            f.write(content)


def _load_source_files(seed_dir: str) -> Optional[Tuple[List[str], List[str]]]:
    """Load source files from cache. Returns (file_paths, file_contents) or None."""
    source_dir = os.path.join(seed_dir, "source_files")
    index_path = os.path.join(source_dir, "index.json")

    if not os.path.exists(index_path):
        return None

    try:
        with open(index_path) as f:
            file_paths = json.load(f)

        file_contents = []
        contents_dir = os.path.join(source_dir, "contents")
        for i in range(len(file_paths)):
            with open(os.path.join(contents_dir, f"{i}.txt")) as f:
                file_contents.append(f.read())

        return file_paths, file_contents
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _save_sample_index(samples_dir: str, sample_idx: int, file_idx: int,
                       mask_start: int, mask_end: int, ground_truth: str) -> None:
    """Save a single sample's mask index."""
    sample_data = {
        "file_idx": file_idx,
        "mask_start": mask_start,
        "mask_end": mask_end,
        "ground_truth": ground_truth,
    }
    with open(os.path.join(samples_dir, f"{sample_idx}.json"), "w") as f:
        json.dump(sample_data, f, indent=2)


def _load_sample_index(samples_dir: str, sample_idx: int) -> Optional[dict]:
    """Load a single sample's mask index. Returns None if not found."""
    sample_path = os.path.join(samples_dir, f"{sample_idx}.json")
    if not os.path.exists(sample_path):
        return None
    try:
        with open(sample_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _rebuild_masked_contents(
    file_paths: List[str],
    file_contents: List[str],
    sample: dict,
) -> Tuple[List[str], List[str], str, str]:
    """Rebuild full masked_paths, masked_contents from sample index.

    Returns: (masked_paths, masked_contents, ground_truth, file_info)
    """
    file_idx = sample["file_idx"]
    mask_start = sample["mask_start"]
    mask_end = sample["mask_end"]
    ground_truth = sample["ground_truth"]

    # Build masked_contents: copy all, then apply mask to target file
    masked_contents = list(file_contents)
    masked_contents[file_idx] = _apply_mask(file_contents[file_idx], mask_start, mask_end)

    # file_info format: "path:start_line-end_line" (1-indexed for display)
    file_info = f"{file_paths[file_idx]}:{mask_start + 1}-{mask_end}"

    return list(file_paths), masked_contents, ground_truth, file_info


def _save_dataset_cache(
    cache_dir: str,
    total_batch_size: int,
    dataset_dir: str,
    seed: Optional[int],
    file_paths: List[str],
    file_contents: List[str],
    sample_indices: List[dict],
    start_index: int = 0,
) -> None:
    """Save generated dataset to cache directory (v2 indexed format).

    Args:
        file_paths: List of all source file paths.
        file_contents: List of all source file contents (original, not masked).
        sample_indices: List of {file_idx, mask_start, mask_end, ground_truth} for each sample.
        start_index: Only save samples starting from this index (for incremental saves).
    """
    seed_dir = _get_seed_subdir(cache_dir, seed)
    os.makedirs(seed_dir, exist_ok=True)

    # Save/update metadata
    metadata = {
        "version": 2,
        "dataset_dir": os.path.realpath(dataset_dir),
        "seed": seed,
        "num_files": len(file_paths),
        "cached_samples": total_batch_size,
    }
    with open(os.path.join(seed_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Save source files (only if not already saved)
    source_index_path = os.path.join(seed_dir, "source_files", "index.json")
    if not os.path.exists(source_index_path):
        _save_source_files(seed_dir, file_paths, file_contents)

    # Save sample indices (only new ones if start_index > 0)
    samples_dir = os.path.join(seed_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    for i in range(start_index, total_batch_size):
        sample = sample_indices[i]
        _save_sample_index(samples_dir, i, sample["file_idx"],
                          sample["mask_start"], sample["mask_end"], sample["ground_truth"])


def _load_dataset_cache(
    cache_dir: str,
    total_batch_size: int,
    dataset_dir: str,
    seed: Optional[int],
    tmpdir: str,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]]:
    """Load dataset from cache if enough samples exist.

    Returns:
        - Full result if cache has >= total_batch_size samples
        - None otherwise (caller should generate missing samples)
    """
    seed_dir = _get_seed_subdir(cache_dir, seed)
    metadata_path = os.path.join(seed_dir, "metadata.json")

    if not os.path.exists(metadata_path):
        return None

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Check dataset_dir matches
    if metadata["dataset_dir"] != os.path.realpath(dataset_dir):
        print(f"Cache dataset_dir mismatch, will regenerate")
        return None

    cached_samples = metadata.get("cached_samples", 0)

    # Check if we have enough samples
    if cached_samples < total_batch_size:
        print(f"Cache has {cached_samples} samples, need {total_batch_size}, will generate more")
        return None

    # Load source files
    source_data = _load_source_files(seed_dir)
    if source_data is None:
        print("Cache source files not found or corrupted")
        return None
    file_paths, file_contents = source_data

    # Load and rebuild samples
    samples_dir = os.path.join(seed_dir, "samples")
    all_masked_paths = []
    all_masked_contents = []
    all_ground_truths = []
    file_info = []

    for i in range(total_batch_size):
        sample = _load_sample_index(samples_dir, i)
        if sample is None:
            print(f"Cache sample {i} not found or corrupted")
            return None

        paths, contents, gt, info = _rebuild_masked_contents(file_paths, file_contents, sample)
        all_masked_paths.append(paths)
        all_masked_contents.append(contents)
        all_ground_truths.append(gt)
        file_info.append(info)

    # Create symbolic tensors
    masked_path_tensor = make_tensor(all_masked_paths, tmpdir)
    masked_content_tensor = make_tensor(all_masked_contents, tmpdir)
    gt_tensor = make_tensor(all_ground_truths, tmpdir)

    return masked_path_tensor, masked_content_tensor, gt_tensor, file_info


def _load_partial_cache(
    cache_dir: str,
    dataset_dir: str,
    seed: Optional[int],
) -> Tuple[List[dict], List[str], List[str], int]:
    """Load partial cache for incremental generation.

    Returns:
        (sample_indices, file_paths, file_contents, cached_count)
        Returns empty values if no valid cache exists.
    """
    seed_dir = _get_seed_subdir(cache_dir, seed)
    metadata_path = os.path.join(seed_dir, "metadata.json")

    if not os.path.exists(metadata_path):
        return [], [], [], 0

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Check dataset_dir matches
    if metadata["dataset_dir"] != os.path.realpath(dataset_dir):
        return [], [], [], 0

    # Load source files
    source_data = _load_source_files(seed_dir)
    if source_data is None:
        return [], [], [], 0
    file_paths, file_contents = source_data

    # Load existing sample indices
    cached_samples = metadata.get("cached_samples", 0)
    samples_dir = os.path.join(seed_dir, "samples")
    sample_indices = []

    for i in range(cached_samples):
        sample = _load_sample_index(samples_dir, i)
        if sample is None:
            break  # Stop at first missing/corrupted sample
        sample_indices.append(sample)

    return sample_indices, file_paths, file_contents, len(sample_indices)


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


def _contains_comment(code_snippet: str) -> bool:
    """Check if code snippet contains Python comments.

    Detects:
    - Single line comments starting with #
    - Docstrings (triple quotes)
    """
    lines = code_snippet.splitlines()
    in_docstring = False
    docstring_char = None

    for line in lines:
        stripped = line.strip()

        # Check for docstring boundaries
        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                docstring_char = stripped[:3]
                # Check if docstring ends on same line
                if stripped.count(docstring_char) >= 2:
                    return True  # Single line docstring
                in_docstring = True
                return True
        else:
            if docstring_char in stripped:
                in_docstring = False
                return True

        # Check for single line comments (# not in string)
        # Simple heuristic: if # appears and is not inside a string
        if '#' in line:
            # Check if # is inside a string by counting quotes before it
            hash_idx = line.find('#')
            before_hash = line[:hash_idx]
            # Count unescaped quotes
            single_quotes = before_hash.count("'") - before_hash.count("\\'")
            double_quotes = before_hash.count('"') - before_hash.count('\\"')
            # If both quote counts are even, # is not inside a string
            if single_quotes % 2 == 0 and double_quotes % 2 == 0:
                return True

    return in_docstring  # Still in unclosed docstring


def _get_random_mask_range(content: str, min_size: int = 5, max_size: int = 10, max_retries: int = 100) -> Tuple[int, int, str]:
    """Pick random line range to mask. Range size in (min_size, max_size).

    Args:
        content: The file content.
        min_size: Minimum number of lines to mask.
        max_size: Maximum number of lines to mask.
        max_retries: Maximum attempts to find a comment-free range.

    Returns:
        Tuple of (start_line, end_line, masked_content).
        The masked content will not contain Python comments.
    """
    lines = content.splitlines(keepends=True)
    code_start, main_start = _find_maskable_range(content)
    maskable_len = main_start - code_start

    if maskable_len < min_size:
        mask_len = max(1, maskable_len)
        start = code_start
        end = start + mask_len
        return start, end, "".join(lines[start:end])

    upper_bound = min(max_size, maskable_len)
    lower_bound = min(min_size, upper_bound)

    for _ in range(max_retries):
        mask_len = random.randint(lower_bound, upper_bound)
        start = random.randint(code_start, main_start - mask_len)
        end = start + mask_len
        snippet = "".join(lines[start:end])

        if not _contains_comment(snippet):
            return start, end, snippet

    # Fallback: return last attempt even if it contains comments
    # This should rarely happen in practice
    return start, end, snippet


def _apply_mask(content: str, start: int, end: int) -> str:
    """Replace lines [start, end) with kMaskedHint."""
    lines = content.splitlines(keepends=True)
    return "".join(lines[:start] + [kMaskedHint + "\n"] + lines[end:])


def parepare_dataset(
    total_batch_size: int,
    dataset_dir: str,
    tmpdir: str,
    cache_dir: Optional[str] = None,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """PrepareDataSet from prepare_dataset.viba.

    Args:
        total_batch_size: Number of samples to generate.
        dataset_dir: Directory containing .py source files.
        tmpdir: Temp directory for symbolic tensor storage.
        cache_dir: If provided, cache dataset to this directory.
            - Cache is organized by seed: {cache_dir}/seed_{N}/
            - If cache has enough samples, load from cache.
            - If cache has partial samples, load existing + generate more.
            - Otherwise, generate all and save to cache.
        seed: Random seed for reproducibility. If None, uses random seed.

    Returns:
        masked_file_path_tensor:    Symbolic[Tensor(batch, num_files)]
        masked_file_content_tensor: Symbolic[Tensor(batch, num_files)]
        ground_truth_tensor:        Symbolic[Tensor(batch,)]
        file_info: list of "file_path:start-end" for logging
    """
    # Try to load full dataset from cache
    if cache_dir is not None:
        cached = _load_dataset_cache(
            cache_dir, total_batch_size, dataset_dir, seed, tmpdir
        )
        if cached is not None:
            seed_dir = _get_seed_subdir(cache_dir, seed)
            print(f"Loaded {total_batch_size} samples from cache: {seed_dir}")
            return cached

    # Try to load partial cache for incremental generation
    sample_indices = []
    file_paths = []
    file_contents = []
    start_index = 0

    if cache_dir is not None:
        sample_indices, file_paths, file_contents, start_index = \
            _load_partial_cache(cache_dir, dataset_dir, seed)
        if start_index > 0:
            seed_dir = _get_seed_subdir(cache_dir, seed)
            print(f"Loaded {start_index} cached samples from: {seed_dir}")

    # Set random seed if provided (advance it to correct position)
    if seed is not None:
        random.seed(seed)
        # Skip random calls for already-cached samples to maintain consistency
        for _ in range(start_index):
            # Each sample uses: 1 randint for file_idx + 2 randints in _get_random_mask_range
            random.randint(0, 1000)  # file_idx
            random.randint(0, 1000)  # mask_len
            random.randint(0, 1000)  # start

    # <- ($files list[($file_path str, $file_content)] <- { fetch all files })
    # Only fetch if not loaded from cache
    if not file_paths:
        files = _fetch_all_files(dataset_dir)
        # Filter files with maskable range >= 5 lines
        files = [(fp, fc) for fp, fc in files
                 if _find_maskable_range(fc)[1] - _find_maskable_range(fc)[0] >= 5]
        assert len(files) > 0, f"No sufficiently large .py files in {dataset_dir}"
        file_paths = [fp for fp, _ in files]
        file_contents = [fc for _, fc in files]

    num_files = len(file_paths)

    # Generate remaining samples
    for _ in range(start_index, total_batch_size):
        file_idx = random.randint(0, num_files - 1)
        fc = file_contents[file_idx]
        start, end, ground_truth = _get_random_mask_range(fc, min_size=5, max_size=10)

        sample_indices.append({
            "file_idx": file_idx,
            "mask_start": start,
            "mask_end": end,
            "ground_truth": ground_truth,
        })

    # Rebuild full masked data from indices
    all_masked_paths = []
    all_masked_contents = []
    all_ground_truths = []
    file_info = []

    for sample in sample_indices:
        paths, contents, gt, info = _rebuild_masked_contents(file_paths, file_contents, sample)
        all_masked_paths.append(paths)
        all_masked_contents.append(contents)
        all_ground_truths.append(gt)
        file_info.append(info)

    # Create symbolic tensors
    masked_path_tensor = make_tensor(all_masked_paths, tmpdir)
    masked_content_tensor = make_tensor(all_masked_contents, tmpdir)
    gt_tensor = make_tensor(all_ground_truths, tmpdir)

    # Save to cache if requested
    if cache_dir is not None:
        _save_dataset_cache(
            cache_dir,
            total_batch_size,
            dataset_dir,
            seed,
            file_paths,
            file_contents,
            sample_indices,
            start_index=start_index,
        )
        seed_dir = _get_seed_subdir(cache_dir, seed)
        if start_index > 0:
            print(f"Saved {total_batch_size - start_index} new samples to cache: {seed_dir}")
        else:
            print(f"Saved {total_batch_size} samples to cache: {seed_dir}")

    return masked_path_tensor, masked_content_tensor, gt_tensor, file_info


if __name__ == "__main__":
    import tempfile
    tmpdir = tempfile.mkdtemp()
    dataset_dir = os.path.join(os.path.dirname(__file__), "codebase")
    paths, contents, gt, info = parepare_dataset(2, dataset_dir, tmpdir)
    print(f"Prepared {len(info)} samples")
    for i, inf in enumerate(info):
        print(f"  [{i}] {inf}")
