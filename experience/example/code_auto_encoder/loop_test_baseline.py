"""Loop test baseline for auto-encoder experiment.

Generated from loop_test_baseline.viba.

Viba DSL specification:
  loop_test_baseline :=
    void
    <- $num_iterations ParsedCmdArg[int] # default 1
    <- $llm_method ParsedCmdOption[str] # default "raw_llm_api"
    <- Import[./test_baseline]
"""

import os
import sys
import subprocess
import argparse

# Add parent path for imports when running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experience.example.auto_encoder.test_baseline import test_baseline


def loop_test_baseline(
    num_iterations: int = 1,
    llm_method: str = "raw_llm_api",
    workspace_dir: str = "/workspace/code-auto-encoder/",
) -> None:
    """loop_test_baseline from loop_test_baseline.viba.

    Args:
        num_iterations: number of test iterations (default 1)
        llm_method: LLM backend to use (default "raw_llm_api")
        workspace_dir: workspace directory for tensors
    """
    losses = []
    for i in range(num_iterations):
        print(f"\n{'='*50}")
        print(f"Run {i+1}/{num_iterations}")
        print(f"{'='*50}")
        baseline_loss = test_baseline(
            total_batch_size=16,
            llm_method=llm_method,
            workspace_dir=workspace_dir,
        )
        mean_loss = sum(baseline_loss) / len(baseline_loss)
        losses.append(mean_loss)
        print(f">>> Run {i+1} loss: {mean_loss:.4f}")

    print(f"\n{'='*50}")
    print(f"Summary ({num_iterations} runs):")
    print(f"{'='*50}")
    print(f"Losses: {['%.4f' % l for l in losses]}")
    mean = sum(losses) / len(losses)
    std = (sum((l - mean)**2 for l in losses) / len(losses))**0.5
    print(f"Mean: {mean:.4f}")
    print(f"Min:  {min(losses):.4f}")
    print(f"Max:  {max(losses):.4f}")
    print(f"Std:  {std:.4f}")


if __name__ == "__main__":
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

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Loop test baseline model")
    parser.add_argument("--num-iterations", type=int, default=1, help="Number of iterations")
    parser.add_argument("--llm-method", type=str, default="raw_llm_api", help="LLM method")
    args = parser.parse_args()

    loop_test_baseline(num_iterations=args.num_iterations, llm_method=args.llm_method)
