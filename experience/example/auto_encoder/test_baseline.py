"""Test baseline model for auto-encoder cloze experiment.

Generated from test_baseline.viba.

Viba DSL specification:
  kMaskedHint := "<AUTOENCODER-CLOZE-MASK-PLACEHOLDER>"

  test_baseline :=
    $baseline_loss list[float]
    <- $total_batch_size int # default 16
    <- $llm_method str # default "raw_llm_api"
    <- $workspace_dir (str | None) # None means temp directory
    <- $dataset_dir <- { ./codebase/ }
    <- Import[./parepare_dataset]
    <- Import[./baseline_coding_agent_model]
    <- Import[function/get_edit_distance]
"""

import os
import sys
import tempfile
from typing import Optional, List

# Add parent path for imports when running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experience.symbolic_tensor.function.get_edit_distance_ratio import get_edit_distance_ratio_impl

from experience.example.auto_encoder.prepare_dataset import parepare_dataset
from experience.example.auto_encoder.baseline_coding_agent_model import BaselineCodingAgentModel


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


def test_baseline(
    total_batch_size: int = 16,
    llm_method: str = "raw_llm_api",
    workspace_dir: Optional[str] = None,
) -> List[float]:
    """test_baseline from test_baseline.viba.

    Args:
        total_batch_size: default 16
        llm_method: default "raw_llm_api"
        workspace_dir: None means temp directory

    Returns:
        List of edit distance ratios per sample
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

    # <- Import[./parepare_dataset]
    masked_path_tensor, masked_content_tensor, gt_tensor, file_info = parepare_dataset(
        total_batch_size, dataset_dir, tmpdir,
    )
    print(f"Batch={total_batch_size}, files={masked_path_tensor.shape[1]}")
    for i, info in enumerate(file_info):
        gt_preview = _read_storage(gt_tensor, i)[:60].replace("\n", "\\n")
        print(f"  [{i}] {info} -> {gt_preview}...")

    # <- Import[./baseline_coding_agent_model]
    print(f"\nRunning BaselineCodingAgentModel (llm_method={llm_method})...")
    model = BaselineCodingAgentModel(llm_method=llm_method)
    output = model(masked_path_tensor, masked_content_tensor)

    # <- Import[function/get_edit_distance]
    loss = get_edit_distance_ratio_impl(output, gt_tensor)

    # <- { print $baseline_output.st_uid ground_truth_content_tensor.st_uid }
    print(f"\noutput tensor uid: {output.st_tensor_uid}")
    print(f"ground_truth tensor uid: {gt_tensor.st_tensor_uid}")

    print(f"\n{'='*50}")
    print(f"Results (edit_distance_ratio, lower=better):")
    print(f"{'='*50}")
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
    print(f"\nMean loss: {mean_loss:.4f}")
    return baseline_loss


if __name__ == "__main__":
    import subprocess

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
    test_baseline(total_batch_size=16, llm_method="raw_llm_api")
