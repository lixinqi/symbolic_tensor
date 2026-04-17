"""Baseline tmux_cc model for auto-encoder experiment.

Uses the tmux_cc CLI tool (similar to Claude Code) to process tasks,
allowing visual observation of the agent's behavior in the terminal.

Pipeline:
  1. Stack (path, content) -> (batch, num_files, 2)
  2. merge(axis=-1) -> (batch, num_files) -- path+content merged per file
  3. coding_agent(llm_method="tmux_cc") -> (batch,)

Modes:
  - interactive=False (default): Run tmux_cc via subprocess, non-interactive
  - interactive=True: Run tmux_cc in tmux session for visual observation
"""

import os
import sys
import torch
import torch.nn as nn

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experience.symbolic_tensor.function.st_stack import st_stack_forward
from experience.symbolic_tensor.function.merge_forward import merge_forward
from experience.symbolic_tensor.function.coding_agent import coding_agent

from experience.example.code_auto_encoder.prepare_dataset import kMaskedHint


class BaselineTmuxCcModel(nn.Module):
    """BaselineTmuxCcModel -- uses tmux_cc CLI for visual agent execution.

    Pipeline:
      1. Stack (path, content) -> (batch, num_files, 2)
      2. merge(axis=-1) -> (batch, num_files) -- path+content merged per file
      3. coding_agent(llm_method="tmux_cc") -> (batch,)

    Args:
        interactive: If True, run tmux_cc in tmux for visual observation.
        auto_confirm: If True (and interactive), auto-confirm prompts in tmux.
        tmux_session: Custom tmux session name (interactive mode only).
    """

    def __init__(
        self,
        interactive: bool = False,
        auto_confirm: bool = True,
        tmux_session: str = None,
    ):
        super().__init__()
        self.interactive = interactive
        self.auto_confirm = auto_confirm
        self.tmux_session = tmux_session

    def forward(
        self,
        masked_path_tensor: torch.Tensor,
        masked_content_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass implementing the pipeline via tmux_cc."""
        batch_size, num_files = masked_path_tensor.shape

        # Step 1: st_stack(path, content, dim=-1) -> (batch, num_files, 2)
        stacked_tensor = st_stack_forward(
            [masked_path_tensor, masked_content_tensor], dim=-1,
        )

        # Step 2: merge(axis=-1) -> (batch, num_files)
        path_and_contents = merge_forward(stacked_tensor, axis=-1)

        # Step 3: coding_agent via tmux_cc
        task_prompt = (
            f"Get back the original content that was masked in {kMaskedHint}\n"
            "Output ONLY the missing source code lines that were masked. No explanations.\n"
            "Do NOT output function signatures, decorators, or docstrings.\n"
            "Do NOT add any comments that were not in the original code.\n"
            "Only output the exact code that was in the masked region - nothing before or after it."
        )

        output = coding_agent(
            path_and_contents,
            task_prompt=task_prompt,
            llm_method="tmux_cc",
            interactive=self.interactive,
            auto_confirm=self.auto_confirm,
            tmux_session=self.tmux_session,
        )
        return output


if __name__ == "__main__":
    print("BaselineTmuxCcModel module loaded successfully.")
