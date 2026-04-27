"""HmWrite — write content to a file in the worktree."""

import os

from experience.example.code_auto_encoder.harness_model.function.harness_op import HarnessOp


class HmWrite(HarnessOp):
    name = "write"
    description = "Write content to a file in the worktree."

    def forward(self, worktree_path: str, file_path: str = "", content: str = "") -> str:
        """Writes file, returns confirmation."""
        full_path = os.path.join(worktree_path, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"(wrote {len(content)} chars to {file_path})"

    def schema(self) -> str:
        return (
            "write(file_path: str, content: str)\n"
            "  Write content to a file in the worktree.\n"
            "  Example: write(file_path=\"output.py\", content=\"print('hello')\")"
        )
