"""Organize experiment results into a structured directory with results.json index.

This module provides utilities to organize experiment results. Supports incremental
updates - results are saved after each task completes.

Directory structure:
    my_experiment/
    ├── dataset/                      # Input data cache (from prepare_dataset)
    │   ├── seed_42/                  # With seed
    │   │   ├── metadata.json
    │   │   ├── source_files/
    │   │   └── samples/
    │   └── no_seed_20260410_143000/  # Without seed (timestamp-based)
    │       └── ...
    ├── runs/
    │   ├── raw_llm_api/              # Grouped by llm_method
    │   │   └── run_20260410_113000/
    │   ├── coding_agent/
    │   │   └── run_20260410_120000/
    │   └── tmux_cc/
    │       └── run_20260410_130000/
    │           ├── config.json       # Run parameters
    │           ├── results.json      # Core index file (input points to dataset/)
    │           ├── output/           # Model outputs and ground truth
    │           │   ├── 0/
    │           │   │   ├── prediction.txt    # Model prediction
    │           │   │   ├── ground_truth.txt  # Expected output for comparison
    │           │   │   └── masked_input.txt  # (optional) The masked input file
    │           │   └── ...
    │           └── logs/
    │               └── run.log       # Run log
    └── latest -> runs/tmux_cc/run_...  # Symlink to latest run
"""

import json
import logging
import os
import shutil
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch


def _setup_file_logger(log_path: str) -> logging.Logger:
    """Setup a file logger that writes to the specified path."""
    logger = logging.getLogger(f"experiment_{os.path.basename(log_path)}")
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers = []

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def _read_tensor_element(tensor: torch.Tensor, index: int) -> str:
    """Read the content of a tensor element from storage."""
    digits = list(str(index))
    path = os.path.join(
        tensor.st_relative_to,
        tensor.st_tensor_uid,
        "storage",
        os.path.join(*digits),
        "data",
    )
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except (FileNotFoundError, UnicodeDecodeError):
        return ""


class ExperimentTracker:
    """Track experiment progress with incremental updates.

    Usage:
        tracker = ExperimentTracker(experiment_dir, config, batch_size, seed, file_info)
        tracker.set_tensors(masked_path, masked_content, gt, output)

        for i, task in enumerate(tasks):
            result = run_task(task)
            tracker.update_task(i, loss, output_content)

        tracker.finalize()
    """

    def __init__(
        self,
        experiment_dir: str,
        config: Dict[str, Any],
        batch_size: int,
        seed: Optional[int] = None,
        file_info: Optional[List[str]] = None,
        dataset_cache_dir: Optional[str] = None,
        llm_method: Optional[str] = None,
    ):
        """Initialize tracker and create run directory.

        Args:
            experiment_dir: Root experiment directory
            config: Run configuration dict
            batch_size: Total number of tasks
            seed: Random seed used (can be None for no-seed experiments)
            file_info: List of file info strings (e.g., "file.py:10-15")
            dataset_cache_dir: Path to dataset cache directory
            llm_method: LLM method used (raw_llm_api, coding_agent, tmux_cc)
        """
        self.experiment_dir = experiment_dir
        self.config = config
        self.batch_size = batch_size
        self.seed = seed
        self.file_info = file_info or []
        self.dataset_cache_dir = dataset_cache_dir
        self.llm_method = llm_method or config.get("llm_method", "unknown")

        # Create run directory grouped by llm_method
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"run_{timestamp}"
        runs_dir = os.path.join(experiment_dir, "runs", self.llm_method)
        self.run_dir = os.path.join(runs_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        # Create subdirectories
        self.output_dir = os.path.join(self.run_dir, "output")
        self.logs_dir = os.path.join(self.run_dir, "logs")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Setup logger
        self.log_path = os.path.join(self.logs_dir, "run.log")
        self.logger = _setup_file_logger(self.log_path)
        self.logger.info(f"Run started: {self.run_id}")
        self.logger.info(f"Config: {json.dumps(config)}")

        # Compute relative path to dataset
        self.dataset_rel_path = None
        if dataset_cache_dir:
            self.dataset_rel_path = os.path.relpath(dataset_cache_dir, self.run_dir)

        # Initialize results structure
        self.results = {
            "run_id": self.run_id,
            "timestamp": timestamp,
            "llm_method": self.llm_method,
            "config": config,
            "status": "running",
            "dataset": {
                "cache_dir": self.dataset_rel_path,
                "seed": seed,  # Can be None
            },
            "summary": {
                "batch_size": batch_size,
                "completed": 0,
                "failed": 0,
                "mean_loss": 0.0,
            },
            "tasks": [None] * batch_size,  # Placeholder for each task
        }

        # Save initial config
        self._save_config()
        self._save_results()
        self._update_latest_symlink()

        self.logger.info(f"Created run directory: {self.run_dir}")
        print(f"[Tracker] Created run directory: {self.run_dir}")

    def set_tensors(
        self,
        masked_path_tensor: torch.Tensor,
        masked_content_tensor: torch.Tensor,
        gt_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
    ) -> None:
        """Register tensor metadata (for reference only, not stored in results)."""
        self.masked_path_tensor = masked_path_tensor
        self.masked_content_tensor = masked_content_tensor
        self.gt_tensor = gt_tensor
        self.output_tensor = output_tensor
        self.logger.info(f"Tensors registered: output_uid={output_tensor.st_tensor_uid}")

    def update_task(
        self,
        task_index: int,
        file_info: str,
        loss: float,
        gt_preview: str = "",
        output_content: str = "",
        masked_input: str = "",
    ) -> None:
        """Update a single task's result (called after each task completes).

        Args:
            task_index: Index of the task (0-based)
            file_info: File path and line info (e.g., "file.py:10-15")
            loss: Loss value for this task
            gt_preview: Full ground truth content (saved to file for comparison)
            output_content: Full model output content
            masked_input: Optional masked input content (the file with mask placeholder)
        """
        # Determine status
        is_todo = output_content.strip() == "TODO"
        status = "failed" if is_todo else "completed"

        # Save output and ground truth to files
        task_output_dir = os.path.join(self.output_dir, str(task_index))
        os.makedirs(task_output_dir, exist_ok=True)

        # Save prediction
        prediction_path = os.path.join(task_output_dir, "prediction.txt")
        with open(prediction_path, "w", encoding="utf-8") as f:
            f.write(output_content)

        # Save ground truth for easy comparison
        gt_path = os.path.join(task_output_dir, "ground_truth.txt")
        with open(gt_path, "w", encoding="utf-8") as f:
            f.write(gt_preview)

        # Save masked input if provided (for debugging)
        if masked_input:
            masked_input_path = os.path.join(task_output_dir, "masked_input.txt")
            with open(masked_input_path, "w", encoding="utf-8") as f:
                f.write(masked_input)

        # Parse file_info to get input reference
        # file_info format: "path/to/file.py:start-end"
        input_ref = self._build_input_reference(file_info)

        # Build task entry
        task_entry = {
            "index": task_index,
            "file_info": file_info,
            "input": input_ref,
            "ground_truth": {
                "path": f"output/{task_index}/ground_truth.txt",
                "preview": gt_preview[:200] if gt_preview else "",
            },
            "output": {
                "path": f"output/{task_index}/prediction.txt",
                "preview": output_content[:200] if output_content else "",
                "loss": loss,
            },
            "status": status,
            "completed_at": datetime.now().isoformat(),
        }

        # Update results
        self.results["tasks"][task_index] = task_entry

        # Update summary
        completed_tasks = [t for t in self.results["tasks"] if t is not None]
        completed_count = sum(1 for t in completed_tasks if t["status"] == "completed")
        failed_count = sum(1 for t in completed_tasks if t["status"] == "failed")
        total_loss = sum(t["output"]["loss"] for t in completed_tasks if t["output"]["loss"] is not None)
        mean_loss = total_loss / len(completed_tasks) if completed_tasks else 0.0

        self.results["summary"]["completed"] = completed_count
        self.results["summary"]["failed"] = failed_count
        self.results["summary"]["mean_loss"] = round(mean_loss, 4)

        # Save immediately
        self._save_results()

        # Log
        status_icon = "✓" if status == "completed" else "✗"
        log_msg = f"Task {task_index + 1}/{self.batch_size} {status_icon} loss={loss:.4f} - {file_info}"
        self.logger.info(log_msg)
        print(f"[Tracker] {log_msg}")

    def _build_input_reference(self, file_info: str) -> Dict[str, Any]:
        """Build input reference pointing to dataset cache."""
        # Parse file_info: "path/to/file.py:start-end"
        if ":" in file_info:
            file_path, line_range = file_info.rsplit(":", 1)
        else:
            file_path = file_info
            line_range = ""

        ref = {
            "file": file_path,
            "lines": line_range,
        }

        # Add dataset reference if available
        if self.dataset_rel_path:
            if self.seed is not None:
                ref["dataset_ref"] = f"{self.dataset_rel_path}/seed_{self.seed}"
            else:
                # For no-seed experiments, reference the dataset dir directly
                ref["dataset_ref"] = self.dataset_rel_path

        return ref

    def log(self, message: str, level: str = "info") -> None:
        """Write a log message."""
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        print(f"[Log] {message}")

    def finalize(self) -> str:
        """Finalize the run.

        Returns:
            Path to the run directory
        """
        # Update status
        self.results["status"] = "completed"
        self.results["completed_at"] = datetime.now().isoformat()
        self._save_results()

        self.logger.info(f"Run completed: mean_loss={self.results['summary']['mean_loss']:.4f}")
        self.logger.info(f"Results saved to: {self.run_dir}")
        print(f"[Tracker] Run finalized: {self.run_dir}")
        return self.run_dir

    def _save_config(self) -> None:
        """Save config.json."""
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def _save_results(self) -> None:
        """Save results.json."""
        results_path = os.path.join(self.run_dir, "results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

    def _update_latest_symlink(self) -> None:
        """Update the 'latest' symlink to point to this run."""
        latest_link = os.path.join(self.experiment_dir, "latest")
        if os.path.islink(latest_link):
            os.unlink(latest_link)
        elif os.path.exists(latest_link):
            shutil.rmtree(latest_link)
        os.symlink(os.path.relpath(self.run_dir, self.experiment_dir), latest_link)


def print_results_summary(results_path: str) -> None:
    """Print a human-readable summary of results.json."""
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    llm_method = results.get('llm_method', results.get('config', {}).get('llm_method', 'unknown'))
    print(f"\n{'='*60}")
    print(f"Run: {results['run_id']} ({results.get('status', 'unknown')})")
    print(f"Method: {llm_method}")
    print(f"{'='*60}")
    print(f"Config: {results['config']}")
    if results.get('dataset'):
        print(f"Dataset: {results['dataset']}")
    print(f"\nSummary:")
    print(f"  Batch size: {results['summary']['batch_size']}")
    print(f"  Completed: {results['summary']['completed']}")
    print(f"  Failed: {results['summary']['failed']}")
    print(f"  Mean loss: {results['summary']['mean_loss']:.4f}")

    print(f"\nTasks:")
    for task in results["tasks"]:
        if task is None:
            continue
        status_icon = "✓" if task["status"] == "completed" else "✗"
        loss_str = f"{task['output']['loss']:.4f}" if task['output']['loss'] is not None else "N/A"
        print(f"  [{task['index']:2d}] {status_icon} {task['file_info']:<50} loss={loss_str}")

    print(f"\nResults saved to: {os.path.dirname(results_path)}")

    # Print log path
    logs_dir = os.path.join(os.path.dirname(results_path), "logs")
    if os.path.exists(logs_dir):
        print(f"Logs: {os.path.join(logs_dir, 'run.log')}")


if __name__ == "__main__":
    print("organize_results.py - Utility module for organizing experiment results")
    print("\nUsage:")
    print("  tracker = ExperimentTracker(experiment_dir, config, batch_size)")
    print("  tracker.set_tensors(...)")
    print("  for i, task in enumerate(tasks):")
    print("      result = run_task(task)")
    print("      tracker.update_task(i, file_info, loss, output_content)")
    print("  tracker.finalize()")
