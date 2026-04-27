"""HmGlob — find files matching a glob pattern in the worktree."""

import glob as _glob
import os

from experience.example.code_auto_encoder.harness_model.function.harness_op import HarnessOp


class HmGlob(HarnessOp):
    name = "glob"
    description = "Find files matching a glob pattern. Returns relative paths."

    def forward(self, worktree_path: str, pattern: str = "**/*.py") -> str:
        """Newline-separated relative paths. Capped at 100 results."""
        matches = _glob.glob(
            os.path.join(worktree_path, pattern), recursive=True,
        )
        rel_paths = sorted(
            os.path.relpath(m, worktree_path)
            for m in matches
            if os.path.isfile(m)
        )
        if len(rel_paths) > 100:
            rel_paths = rel_paths[:100]
            rel_paths.append(f"... ({len(matches) - 100} more)")
        return "\n".join(rel_paths) if rel_paths else "(no matches)"

    def schema(self) -> str:
        return (
            "glob(pattern: str)\n"
            "  Find files matching a glob pattern. Returns relative paths.\n"
            "  Example: glob(pattern=\"**/*.py\")"
        )
