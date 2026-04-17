"""Baseline coding agent model for auto-encoder experiment.

Generated from baseline_coding_agent_model.viba.

Viba DSL specification:
  BaselineCodingAgentModel[torch.nn.Module] :=
    $baseline_output SymbolicTensor[($total_batch_size,)]
    <- $masked_file_path_tensor Symbolic[Tensor[($total_batch_size, $num_files)]]
    <- $masked_file_content_tensor Symbolic[Tensor[($total_batch_size, $num_files)]]
    <- $llm_method str # default "coding_agent"
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


class BaselineCodingAgentModel(nn.Module):
    """BaselineCodingAgentModel from baseline_coding_agent_model.viba.

    Pipeline:
      1. Stack (path, content) → (batch, num_files, 2)
      2. merge(axis=-1) → (batch, num_files)
      3. coding_agent → (batch,)
    """

    def __init__(self, llm_method: str = "coding_agent"):
        super().__init__()
        self.llm_method = llm_method

    def forward(
        self,
        masked_path_tensor: torch.Tensor,
        masked_content_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass implementing the viba pipeline."""
        batch_size, num_files = masked_path_tensor.shape

        # Step 1: st_stack(path, content, dim=-1) → (batch, num_files, 2)
        stacked_tensor = st_stack_forward(
            [masked_path_tensor, masked_content_tensor], dim=-1,
        )

        # Step 2: merge(axis=-1) → (batch, num_files)
        path_and_contents = merge_forward(stacked_tensor, axis=-1)

        # Step 3: coding_agent
        # $prompt str <- { Get back the original content that was masked in f"{kMaskedHint}" }
        task_prompt = (
            f"Get back the original content that was masked in {kMaskedHint}\n"
            "Output ONLY the missing source code lines that were masked. No explanations.\n"
            "Do NOT output function signatures, decorators, or docstrings.\n"
            "Do NOT add any comments that were not in the original code.\n"
            "Only output the exact code that was in the masked region - nothing before or after it."
        )

        # $baseline_output <- Import[function/coding_agent.viba]
        #   <- $path_and_contents <- $prompt <- $llm_method "coding_agent"
        output = coding_agent(
            path_and_contents,
            task_prompt=task_prompt,
            llm_method="coding_agent",
        )
        return output


if __name__ == "__main__":
    print("BaselineCodingAgentModel module loaded successfully.")
