"""HmRead — read a file with line numbers from the worktree."""

import os

from experience.example.code_auto_encoder.harness_model.function.harness_op import HarnessOp


class HmRead(HarnessOp):
    name = "read"
    description = "Read a file with line numbers. Supports offset/limit."

    def forward(
        self,
        worktree_path: str,
        file_path: str = "",
        offset: int = 0,
        limit: int = 200,
    ) -> str:
        """Numbered lines (cat -n style). file_path relative to worktree."""
        full_path = os.path.join(worktree_path, file_path)
        if not os.path.isfile(full_path):
            return f"(file not found: {file_path})"
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except (UnicodeDecodeError, OSError) as e:
            return f"(read error: {e})"

        selected = lines[offset:offset + limit]
        numbered = []
        for i, line in enumerate(selected, start=offset + 1):
            numbered.append(f"{i:>6}\t{line.rstrip()}")
        return "\n".join(numbered) if numbered else "(empty file)"

    def schema(self) -> str:
        return (
            "read(file_path: str, offset: int = 0, limit: int = 200)\n"
            "  Read a file with line numbers. file_path relative to worktree.\n"
            "  Example: read(file_path=\"src/main.py\", offset=10, limit=30)"
        )
