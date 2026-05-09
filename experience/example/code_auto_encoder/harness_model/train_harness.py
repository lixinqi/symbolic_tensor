"""Multi-epoch training script for HarnessModel.

Usage:
  python -m experience.example.code_auto_encoder.harness_model.train_harness \\
      --n-epochs 5 --n-experience 64 --lr 1.0 \\
      --accumulate-mode weighted --split both --seed 42 \\
      --experience-dir /tmp/harness_experience

Acceptance criteria:
  - Epoch 5 train mean loss < 0.15
  - Eval loss decreases alongside train loss (no overfitting)
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
from experience.example.code_auto_encoder.harness_model.harness_model import HarnessModel
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


def _iter_batches(tmpdir: str, split: str, batch_size: int, seed: int):
    """Yield (worktree_tensor, gt_tensor, file_info) for each mini-batch in the split."""
    samples_path = os.path.join(_DATASET_INDEX_DIR, split, "samples.json")
    with open(samples_path) as f:
        all_samples = json.load(f)

    import random
    rng = random.Random(seed)
    rng.shuffle(all_samples)

    for start in range(0, len(all_samples), batch_size):
        batch = all_samples[start:start + batch_size]
        if not batch:
            break
        wt, gt, info = prepare_worktrees(
            total_batch_size=len(batch),
            dataset_dir=_DATASET_DIR,
            tmpdir=tmpdir,
            seed=seed,
            split=None,  # we supply our own selected samples below
            dataset_index_dir=_DATASET_INDEX_DIR,
        )
        yield wt, gt, info
        break  # one mini-batch per call; caller controls looping


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


def _eval_epoch(model: HarnessModel, eval_tmpdir: str, batch_size: int, seed: int) -> float:
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


def _apply_quality_guard(model: HarnessModel, current_step_loss: float, experience_dir: str) -> None:
    """Zero grad for rows whose stored best loss is already lower than current_step_loss.

    Only updates row_losses for rows that were actually retrieved (selected) this step.
    Non-selected rows have zero gradient already; updating their stored loss based on an
    unrelated step's loss would incorrectly lock them out of future overwrites.
    """
    exp = model.experience
    if exp is None:
        return
    symbolic_grad = symbolic_grad_registry.peek(exp.st_tensor_uid)
    if symbolic_grad is None:
        return

    # Rows actually retrieved this step have non-zero gradient coefficients.
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
            # Selected row with better loss: allow overwrite and record new best
            row_losses[row_idx] = current_step_loss
            updated = True
        # Non-selected rows: gradient is already 0, don't update their stored loss
    if updated:
        _save_row_losses(experience_dir, row_losses)


def _record_newly_written_rows(model: HarnessModel, step_loss: float, experience_dir: str) -> None:
    """After optimizer.step(), register initial loss for rows that were just written (key was empty)."""
    exp = model.experience
    if exp is None:
        return
    row_losses = _load_row_losses(experience_dir)
    updated = False
    for row_idx in range(exp.shape[0]):
        if row_idx in row_losses:
            continue  # already tracked
        key_flat = row_idx * 3 + 1
        key_text = _read_storage(exp, key_flat)
        if key_text.strip():
            row_losses[row_idx] = step_loss
            updated = True
    if updated:
        _save_row_losses(experience_dir, row_losses)


def _save_experience_snapshot(model: HarnessModel, experience_dir: str, epoch: int) -> None:
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
    n_experience: int = 256,
    n_epochs: int = 5,
    lr: float = 1.0,
    batch_size: int = 1,
    steps_per_epoch: int = 1,
    accumulate_mode: str = "weighted",
    split: str = "both",
    seed: int = 42,
    experience_dir: Optional[str] = None,
    max_codegen_steps: int = 4,
    max_context_collects: int = 5,
    max_tool_call_retries: int = 2,
    topk: int = 2,
    llm_method: str = "raw_llm_api",
    max_concurrent: int = 0,
) -> dict:
    """Run multi-epoch training and return results dict.

    Returns:
        {
          "train_losses": [float, ...],   # per-epoch mean train loss
          "eval_losses": [float, ...],    # per-epoch mean eval loss (if split in {"eval","both"})
          "patch_stats": [...],           # per-epoch StSGD stats
        }
    """
    if experience_dir is None:
        experience_dir = tempfile.mkdtemp()
    os.makedirs(experience_dir, exist_ok=True)

    if max_concurrent > 0:
        from experience.llm_client.raw_llm_query import set_max_concurrent
        set_max_concurrent(max_concurrent)
        print(f"[train] max_concurrent={max_concurrent} (LLM API calls serialized)")

    print(f"\n{'='*60}")
    print(f"HarnessModel Training")
    print(f"  n_experience={n_experience}, n_epochs={n_epochs}, lr={lr}")
    print(f"  batch_size={batch_size}, steps_per_epoch={steps_per_epoch}, accumulate_mode={accumulate_mode}")
    print(f"  split={split}, seed={seed}")
    print(f"  experience_dir={experience_dir}")
    print(f"{'='*60}\n")

    # Experience persists across epochs (experience replay)
    model = HarnessModel(
        n_experience=n_experience,
        experience_dir=experience_dir,
        max_codegen_steps=max_codegen_steps,
        max_context_collects=max_context_collects,
        max_tool_call_retries=max_tool_call_retries,
        topk=topk,
        llm_method=llm_method,
        accumulate_mode=accumulate_mode,
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

                    # Quality guard: allow overwriting rows only if current loss is better
                    _apply_quality_guard(model, step_loss, experience_dir)

                    print("  [Step]")
                    optimizer.step()
                    # Record initial loss for rows just written by this step
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

        # Save experience snapshot after each epoch
        _save_experience_snapshot(model, experience_dir, epoch)

        elapsed = time.time() - t0
        train_str = f"{train_losses[-1]:.4f}" if not torch.isnan(torch.tensor(train_losses[-1])) else "N/A"
        eval_str = f"{eval_losses[-1]:.4f}" if not torch.isnan(torch.tensor(eval_losses[-1])) else "N/A"
        print(f"\n  Epoch {epoch} | train={train_str} eval={eval_str} | elapsed={elapsed:.1f}s")

    # Summary
    print(f"\n{'='*60}")
    print("Training Complete")
    print(f"{'='*60}")
    print(f"Train losses: {[f'{v:.4f}' for v in train_losses]}")
    if any(not torch.isnan(torch.tensor(v)) for v in eval_losses):
        print(f"Eval  losses: {[f'{v:.4f}' for v in eval_losses]}")

    valid_train = [v for v in train_losses if not (v != v)]  # filter NaN
    if len(valid_train) >= 2:
        if valid_train[-1] < valid_train[0]:
            print(f"\nTrain loss CONVERGED: {valid_train[0]:.4f} -> {valid_train[-1]:.4f}")
        else:
            print(f"\nTrain loss DID NOT converge: {valid_train[0]:.4f} -> {valid_train[-1]:.4f}")

        if valid_train[-1] < 0.15:
            print(f"ACCEPTANCE CRITERIA MET: final train loss {valid_train[-1]:.4f} < 0.15")
        else:
            print(f"acceptance criteria NOT met: final train loss {valid_train[-1]:.4f} >= 0.15")

    return {
        "train_losses": train_losses,
        "eval_losses": eval_losses,
        "patch_stats": all_patch_stats,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HarnessModel for code auto-encoder.")
    parser.add_argument("--n-experience", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=1)
    parser.add_argument("--accumulate-mode", choices=["naive", "weighted"], default="weighted")
    parser.add_argument("--split", choices=["train", "eval", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experience-dir", type=str, default=None)
    parser.add_argument("--max-codegen-steps", type=int, default=4)
    parser.add_argument("--max-context-collects", type=int, default=5)
    parser.add_argument("--max-tool-call-retries", type=int, default=2)
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
        accumulate_mode=args.accumulate_mode,
        split=args.split,
        seed=args.seed,
        experience_dir=args.experience_dir,
        max_codegen_steps=args.max_codegen_steps,
        max_context_collects=args.max_context_collects,
        max_tool_call_retries=args.max_tool_call_retries,
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
