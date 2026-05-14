"""Multi-epoch training script for TmuxHarnessModel.

Usage:
  python -m experience.example.code_auto_encoder.tmux_harness_model.train_tmux_harness \\
      --batch-size 1 --n-epochs 1 --split train  # smoke test

  python -m experience.example.code_auto_encoder.tmux_harness_model.train_tmux_harness \\
      --batch-size 4 --n-epochs 4 --split both   # convergence test

Acceptance criteria:
  - Train loss decreases across epochs
  - Eval loss tracks train loss (no overfitting)
"""

import argparse
import json
import os
import sys
import tempfile
import time
from typing import List, Optional, Tuple

import torch

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.function.get_edit_distance_ratio import (
    get_edit_distance_ratio,
    get_edit_distance_ratio_impl,
)
from experience.symbolic_tensor.optimizer.st_sgd import StSGD
from experience.symbolic_tensor.function import symbolic_grad_registry
from experience.example.code_auto_encoder.tmux_harness_model.tmux_harness_model import TmuxHarnessModel
from experience.example.code_auto_encoder.harness_model.prepare_worktrees import prepare_worktrees

_DATASET_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "..", "codebase",
)
_DATASET_INDEX_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "..", "dataset_index",
)


def _read_storage(tensor, flat_index: int) -> str:
    digits = list(str(flat_index))
    path = os.path.join(
        tensor.st_relative_to, tensor.st_tensor_uid,
        "storage", os.path.join(*digits), "data",
    )
    with open(path) as f:
        return f.read()


def _prepare_batch(tmpdir: str, split: str, batch_size: int, seed: int):
    """Prepare one batch (worktree_tensor, gt_tensor) for a given split."""
    wt, gt, info = prepare_worktrees(
        total_batch_size=batch_size,
        dataset_dir=_DATASET_DIR,
        tmpdir=tmpdir,
        seed=seed,
        split=split,
        dataset_index_dir=_DATASET_INDEX_DIR,
    )
    return wt, gt, info


def _eval_epoch(model: TmuxHarnessModel, eval_tmpdir: str, batch_size: int, seed: int) -> float:
    """Run eval pass (no backward) and return mean edit-distance loss."""
    wt, gt, _ = _prepare_batch(eval_tmpdir, "eval", batch_size, seed)
    with torch.no_grad():
        output = model(wt)
    loss = get_edit_distance_ratio_impl(output, gt)
    return loss.float().mean().item()


_ROW_LOSSES_FILE = "row_losses.json"


def _load_row_losses(experience_dir: str) -> dict:
    path = os.path.join(experience_dir, _ROW_LOSSES_FILE)
    if os.path.exists(path):
        with open(path) as f:
            return {int(k): v for k, v in json.load(f).items()}
    return {}


def _save_row_losses(experience_dir: str, losses: dict) -> None:
    path = os.path.join(experience_dir, _ROW_LOSSES_FILE)
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in losses.items()}, f)


def _apply_quality_guard(model: TmuxHarnessModel, current_step_loss: float, experience_dir: str) -> None:
    """Zero grad for rows whose stored best loss is already lower than current_step_loss."""
    exp = model.experience
    if exp is None:
        return
    symbolic_grad = symbolic_grad_registry.peek(exp.st_tensor_uid)
    if symbolic_grad is None:
        return

    nz_rows = set(
        torch.nonzero(symbolic_grad.data, as_tuple=True)[0].unique().tolist()
    )

    row_losses = _load_row_losses(experience_dir)
    n_rows = exp.shape[0]
    updated = False
    for row_idx in range(n_rows):
        key_flat = row_idx * 3 + 1
        key_text = _read_storage(exp, key_flat)
        if not key_text.strip():
            continue  # empty row: allow write freely
        best_loss = row_losses.get(row_idx, float("inf"))
        if current_step_loss >= best_loss:
            symbolic_grad.data[row_idx] = 0.0  # protect: existing entry is better or equal
        elif row_idx in nz_rows:
            row_losses[row_idx] = current_step_loss
            updated = True
    if updated:
        _save_row_losses(experience_dir, row_losses)


def _record_newly_written_rows(model: TmuxHarnessModel, step_loss: float, experience_dir: str) -> None:
    """After optimizer.step(), register initial loss for rows that were just written."""
    exp = model.experience
    if exp is None:
        return
    row_losses = _load_row_losses(experience_dir)
    updated = False
    for row_idx in range(exp.shape[0]):
        if row_idx in row_losses:
            continue
        key_flat = row_idx * 3 + 1
        key_text = _read_storage(exp, key_flat)
        if key_text.strip():
            row_losses[row_idx] = step_loss
            updated = True
    if updated:
        _save_row_losses(experience_dir, row_losses)


def _save_experience_snapshot(model: TmuxHarnessModel, experience_dir: str, epoch: int) -> None:
    """Save a snapshot of experience content to experience_dir/snapshots/epoch_{epoch}.json."""
    exp = model.experience
    if exp is None:
        return
    snapshots_dir = os.path.join(experience_dir, "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)
    n_rows = exp.shape[0]
    snapshot = []
    for row_idx in range(n_rows):
        try:
            q = _read_storage(exp, row_idx * 3 + 0)
            k = _read_storage(exp, row_idx * 3 + 1)
            v = _read_storage(exp, row_idx * 3 + 2)
            snapshot.append({"query": q[:100], "key": k[:100], "value": v[:200]})
        except Exception:
            snapshot.append({"query": "", "key": "", "value": ""})
    snapshot_path = os.path.join(snapshots_dir, f"epoch_{epoch}.json")
    with open(snapshot_path, "w") as f:
        json.dump(snapshot, f, indent=2)
    print(f"  [snapshot saved: {snapshot_path}]")


def train(
    n_experience: int = 64,
    n_epochs: int = 4,
    lr: float = 1.0,
    batch_size: int = 4,
    steps_per_epoch: int = 1,
    split: str = "both",
    seed: int = 42,
    experience_dir: Optional[str] = None,
    max_codegen_steps: int = 4,
    topk: int = 2,
    llm_method: str = "raw_llm_api",
    max_concurrent: int = 0,
) -> dict:
    """Run multi-epoch training and return results dict.

    Returns:
        {
          "train_losses": [float, ...],
          "eval_losses": [float, ...],
          "patch_stats": [...],
        }
    """
    if experience_dir is None:
        experience_dir = tempfile.mkdtemp()
    os.makedirs(experience_dir, exist_ok=True)

    if max_concurrent > 0:
        from experience.llm_client.raw_llm_query import set_max_concurrent
        set_max_concurrent(max_concurrent)
        print(f"[train] max_concurrent={max_concurrent}")

    print(f"\n{'='*60}")
    print(f"TmuxHarnessModel Training")
    print(f"  n_experience={n_experience}, n_epochs={n_epochs}, lr={lr}")
    print(f"  batch_size={batch_size}, steps_per_epoch={steps_per_epoch}")
    print(f"  split={split}, seed={seed}")
    print(f"  max_codegen_steps={max_codegen_steps}, topk={topk}")
    print(f"  experience_dir={experience_dir}")
    print(f"{'='*60}\n")

    model = TmuxHarnessModel(
        n_experience=n_experience,
        experience_dir=experience_dir,
        max_codegen_steps=max_codegen_steps,
        topk=topk,
        llm_method=llm_method,
    )
    optimizer = StSGD([model.experience], lr=lr)

    train_losses = []
    eval_losses = []
    all_patch_stats = []

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        print(f"\n{'─'*60}")
        print(f"Epoch {epoch}/{n_epochs}")
        print(f"{'─'*60}")

        if split in ("train", "both"):
            step_losses = []
            epoch_patch_stats = {"applied": 0, "rejected": 0, "fuzzed": 0, "skipped": 0}

            for step in range(steps_per_epoch):
                with tempfile.TemporaryDirectory() as train_tmpdir:
                    optimizer.zero_grad()
                    step_seed = seed + epoch * 1000 + step
                    wt, gt, info = _prepare_batch(train_tmpdir, "train", batch_size, step_seed)

                    print(f"  [Forward] epoch={epoch} step={step+1}/{steps_per_epoch} batch_size={wt.shape[0]}")
                    output = model(wt)

                    output.requires_grad_(True)
                    loss = get_edit_distance_ratio(output, gt)
                    step_loss = loss.float().mean().item()
                    step_losses.append(step_loss)
                    print(f"  [Loss] train={step_loss:.4f}")

                    print("  [Backward]")
                    loss_scalar = loss.mean() * model.last_output_ft
                    loss_scalar.backward()

                    # Quality guard
                    _apply_quality_guard(model, step_loss, experience_dir)

                    print("  [Step]")
                    optimizer.step()
                    _record_newly_written_rows(model, step_loss, experience_dir)
                    stats = optimizer.get_last_step_stats()
                    for k in epoch_patch_stats:
                        epoch_patch_stats[k] += stats.get(k, 0)
                    print(f"  [Patches] applied={stats['applied']} rejected={stats['rejected']} "
                          f"fuzzed={stats['fuzzed']} skipped={stats['skipped']}")

            mean_train_loss = sum(step_losses) / len(step_losses)
            train_losses.append(mean_train_loss)
            all_patch_stats.append(epoch_patch_stats)
            print(f"  [Epoch train loss] mean={mean_train_loss:.4f} over {steps_per_epoch} steps")
        else:
            train_losses.append(float("nan"))

        if split in ("eval", "both"):
            with tempfile.TemporaryDirectory() as eval_tmpdir:
                eval_loss = _eval_epoch(model, eval_tmpdir, batch_size, seed)
                eval_losses.append(eval_loss)
                print(f"  [Loss] eval={eval_loss:.4f}")
        else:
            eval_losses.append(float("nan"))

        _save_experience_snapshot(model, experience_dir, epoch)

        elapsed = time.time() - t0
        train_str = f"{train_losses[-1]:.4f}" if not (train_losses[-1] != train_losses[-1]) else "N/A"
        eval_str = f"{eval_losses[-1]:.4f}" if not (eval_losses[-1] != eval_losses[-1]) else "N/A"
        print(f"\n  Epoch {epoch} | train={train_str} eval={eval_str} | elapsed={elapsed:.1f}s")

    # Summary
    print(f"\n{'='*60}")
    print("Training Complete")
    print(f"{'='*60}")
    print(f"Train losses: {[f'{v:.4f}' for v in train_losses]}")
    if any(not (v != v) for v in eval_losses):
        print(f"Eval  losses: {[f'{v:.4f}' for v in eval_losses]}")

    valid_train = [v for v in train_losses if not (v != v)]
    if len(valid_train) >= 2:
        if valid_train[-1] < valid_train[0]:
            print(f"\nTrain loss CONVERGED: {valid_train[0]:.4f} -> {valid_train[-1]:.4f}")
        else:
            print(f"\nTrain loss DID NOT converge: {valid_train[0]:.4f} -> {valid_train[-1]:.4f}")

    return {
        "train_losses": train_losses,
        "eval_losses": eval_losses,
        "patch_stats": all_patch_stats,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TmuxHarnessModel for code auto-encoder.")
    parser.add_argument("--n-experience", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps-per-epoch", type=int, default=1)
    parser.add_argument("--split", choices=["train", "eval", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experience-dir", type=str, default=None)
    parser.add_argument("--max-codegen-steps", type=int, default=4)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--llm-method", type=str, default="raw_llm_api")
    parser.add_argument("--max-concurrent", type=int, default=0,
                        help="Max concurrent LLM API requests (0=unlimited)")
    args = parser.parse_args()

    results = train(
        n_experience=args.n_experience,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        split=args.split,
        seed=args.seed,
        experience_dir=args.experience_dir,
        max_codegen_steps=args.max_codegen_steps,
        topk=args.topk,
        llm_method=args.llm_method,
        max_concurrent=args.max_concurrent,
    )

    # Save results to reports/
    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    results_path = os.path.join(reports_dir, "train_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
