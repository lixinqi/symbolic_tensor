"""Baseline raw LLM API model for auto-encoder experiment.

Generated from baseline_raw_llm_api_model.viba.

Viba DSL specification:
  BaselineRawLlmApiModel[torch.nn.Module] :=
    $baseline_output SymbolicTensor[($total_batch_size,)]
    <- $masked_file_path_tensor Symbolic[Tensor[($total_batch_size, $num_files)]]
    <- $masked_file_content_tensor Symbolic[Tensor[($total_batch_size, $num_files)]]
    <- ()
    # Pipeline: stack → merge → merge(axis=-1) → coding_agent
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


class BaselineRawLlmApiModel(nn.Module):
    """BaselineRawLlmApiModel from baseline_raw_llm_api_model.viba.

    Pipeline:
      1. Stack (path, content) → (batch, num_files, 2)
      2. merge(axis=-1) → (batch, num_files) — path+content merged per file
      3. merge(axis=-1) → (batch, 1) — all files merged into one
      4. coding_agent → (batch,)
    """

    def __init__(self):
        super().__init__()

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
        # $path_and_contents <- Import[function/merge] <- { concat on last axis }
        path_and_contents = merge_forward(stacked_tensor, axis=-1)

        # Step 3: merge(axis=-1) → (batch, 1)
        # $merged_path_and_contents <- Import[function/merge] <- $path_and_contents <- $axis -1
        merged_path_and_contents = merge_forward(path_and_contents, axis=-1)

        # Step 4: coding_agent
        # $prompt <- { Get back the original content that was masked in f"{kMaskedHint}" }
        task_prompt = (
            f"Get back the original content that was masked in {kMaskedHint}\n"
            "Output ONLY the missing source code lines that were masked. No explanations.\n"
            "Do NOT output function signatures, decorators, or docstrings.\n"
            "Do NOT add any comments that were not in the original code.\n"
            "Only output the exact code that was in the masked region - nothing before or after it."
        )

        # $baseline_output <- Import[function/coding_agent.viba]
        #   <- $merged_path_and_contents <- $prompt <- $llm_method "raw_llm_api"
        output = coding_agent(
            merged_path_and_contents,
            task_prompt=task_prompt,
            llm_method="raw_llm_api",
        )
        return output


if __name__ == "__main__":
    print("BaselineRawLlmApiModel module loaded successfully.")
