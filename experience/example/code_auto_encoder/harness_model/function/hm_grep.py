"""HmGrep — search file contents by regex in the worktree."""

import os
import re

from experience.example.code_auto_encoder.harness_model.function.harness_op import HarnessOp


class HmGrep(HarnessOp):
    name = "grep"
    description = "Search file contents by regex. Returns file:line:content."

    def forward(
        self,
        worktree_path: str,
        pattern: str = "",
        glob: str = None,
        max_results: int = 50,
    ) -> str:
        """Newline-separated file:line:content matches."""
        import fnmatch

        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"(regex error: {e})"

        results = []
        for dirpath, _dirs, filenames in os.walk(worktree_path):
            for fname in sorted(filenames):
                if glob and not fnmatch.fnmatch(fname, glob):
                    # Also check relative path
                    rel = os.path.relpath(
                        os.path.join(dirpath, fname), worktree_path,
                    )
                    if not fnmatch.fnmatch(rel, glob):
                        continue
                fpath = os.path.join(dirpath, fname)
                rel = os.path.relpath(fpath, worktree_path)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        for lineno, line in enumerate(f, 1):
                            if regex.search(line):
                                results.append(
                                    f"{rel}:{lineno}:{line.rstrip()}"
                                )
                                if len(results) >= max_results:
                                    results.append(f"... (capped at {max_results})")
                                    return "\n".join(results)
                except (UnicodeDecodeError, OSError):
                    continue
        return "\n".join(results) if results else "(no matches)"

    def schema(self) -> str:
        return (
            "grep(pattern: str, glob: str | None = None, max_results: int = 50)\n"
            "  Search file contents by regex. Returns file:line:content.\n"
            "  Example: grep(pattern=\"def forward\", glob=\"*.py\")"
        )
