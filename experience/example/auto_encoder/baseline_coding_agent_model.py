"""Baseline coding agent model for auto-encoder experiment.

Generated from baseline_coding_agent_model.viba.

Viba DSL specification:
  BaselineCodingAgentModel[torch.nn.Module] :=
    $baseline_output SymbolicTensor[($total_batch_size,)]
    <- $masked_file_path_tensor Symbolic[Tensor[($total_batch_size, $num_files)]]
    <- $masked_file_content_tensor Symbolic[Tensor[($total_batch_size, $num_files)]]
    <- $llm_method str # default "raw_llm_api"
"""

import os
import sys
import torch
import torch.nn as nn

# Add parent path for imports when running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experience.symbolic_tensor.tensor_util.slice_tensor import slice_tensor
from experience.symbolic_tensor.function.st_stack import st_stack_forward
from experience.symbolic_tensor.function.merge_forward import merge_forward
from experience.symbolic_tensor.function.st_attention import st_attention
from experience.symbolic_tensor.function.coding_agent import coding_agent

from experience.example.auto_encoder.prepare_dataset import kMaskedHint


class BaselineCodingAgentModel(nn.Module):
    """BaselineCodingAgentModel from baseline_coding_agent_model.viba.

    Pipeline:
      1. Stack (path, content) → (batch, num_files, 2)
      2. merge(axis=-1) → (batch, num_files)
      3. st_attention(prefix→last mask) → (batch, num_files)
      4. slice last col → (batch,)
      5. coding_agent → (batch,)
    """

    def __init__(self, llm_method: str = "raw_llm_api"):
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
        # $path_and_contents <- Import[function/merge] <- { concat on last axis }
        stacked_tensor = st_stack_forward(
            [masked_path_tensor, masked_content_tensor], dim=-1,
        )

        # Step 2: merge(axis=-1) → (batch, num_files)
        path_and_contents = merge_forward(stacked_tensor, axis=-1)

        # Step 3: st_attention with prefix→last mask
        # $merged_path_and_contents <- { fetch last column }
        #   <- Import[function/st_attention] <- { prepare special attention mask to concat prefixes to last column }
        attention_mask = torch.eye(num_files, dtype=torch.bool).unsqueeze(0).expand(batch_size, -1, -1).clone()
        attention_mask[:, num_files - 1, :] = True
        merged = st_attention(path_and_contents, attention_mask, return_view=True)

        # Step 4: slice last column → (batch,)
        last_col_idx = [slice(None), num_files - 1]
        merged_context = slice_tensor(merged, last_col_idx)

        # Step 5: coding_agent
        # $prompt <- { autoencoder prompt for baseline model }
        task_prompt = (
            "You are an auto-encoder for source code.\n"
            f"The input contains Python files from a repository packed with frame markers.\n"
            f"One file has a masked region marked by {kMaskedHint}.\n"
            "Your task: reconstruct ONLY the masked lines.\n"
            "Output ONLY the missing source code lines. No explanations."
        )

        # $baseline_output <- Import[function/coding_agent.viba]
        #   <- $merged_path_and_contents <- $prompt <- $llm_method
        output = coding_agent(
            merged_context,
            task_prompt=task_prompt,
            llm_method=self.llm_method,
        )
        return output


if __name__ == "__main__":
    print("BaselineCodingAgentModel module loaded successfully.")
