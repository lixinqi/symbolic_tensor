"""Baseline agent model for auto-encoder experiment.

Generated from baseline_agent_model.viba.

Viba DSL specification:
  BaselineAgentModel[torch.nn.Module] :=
    $baseline_output SymbolicTensor[($total_batch_size,)]
    <- $masked_file_path_tensor Symbolic[Tensor[($total_batch_size, $num_files)]]
    <- $masked_file_content_tensor Symbolic[Tensor[($total_batch_size, $num_files)]]
    <- $llm_method str # default "raw_llm_api"
    # Dispatch based on llm_method:
    #   - raw_llm_api → BaselineRawLlmApiModel
    #   - coding_agent → BaselineCodingAgentModel
"""

import os
import sys
import torch
import torch.nn as nn

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experience.example.auto_encoder.baseline_raw_llm_api_model import BaselineRawLlmApiModel
from experience.example.auto_encoder.baseline_coding_agent_model import BaselineCodingAgentModel


class BaselineAgentModel(nn.Module):
    """BaselineAgentModel from baseline_agent_model.viba.

    Dispatches to the appropriate model based on llm_method:
      - "raw_llm_api" → BaselineRawLlmApiModel
      - "coding_agent" → BaselineCodingAgentModel
    """

    def __init__(self, llm_method: str = "raw_llm_api"):
        super().__init__()
        self.llm_method = llm_method

        # <- ({ match llm_method }
        #     <- { case raw_llm_api } <- Import[./baseline_raw_llm_api_model]
        #     <- { case coding_agent } <- Import[./baseline_coding_agent_model])
        if llm_method == "raw_llm_api":
            self._model = BaselineRawLlmApiModel()
        elif llm_method == "coding_agent":
            self._model = BaselineCodingAgentModel(llm_method="coding_agent")
        else:
            raise ValueError(f"Unknown llm_method: {llm_method}. Expected 'raw_llm_api' or 'coding_agent'.")

    def forward(
        self,
        masked_path_tensor: torch.Tensor,
        masked_content_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass dispatching to the appropriate model."""
        return self._model(masked_path_tensor, masked_content_tensor)


if __name__ == "__main__":
    print("Testing BaselineAgentModel dispatch...")

    # Test both dispatch paths
    model_raw = BaselineAgentModel(llm_method="raw_llm_api")
    model_agent = BaselineAgentModel(llm_method="coding_agent")

    print(f"  raw_llm_api model: {type(model_raw._model).__name__}")
    print(f"  coding_agent model: {type(model_agent._model).__name__}")
    print("BaselineAgentModel module loaded successfully.")
