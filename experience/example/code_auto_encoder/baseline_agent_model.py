"""Baseline agent model for auto-encoder experiment.

Generated from baseline_agent_model.viba.

Viba DSL specification:
  BaselineAgentModel[torch.nn.Module] :=
    $baseline_output SymbolicTensor[($total_batch_size,)]
    <- $masked_file_path_tensor Symbolic[Tensor[($total_batch_size, $num_files)]]
    <- $masked_file_content_tensor Symbolic[Tensor[($total_batch_size, $num_files)]]
    <- $llm_method str # default "raw_llm_api"
    <- $interactive bool # default False (only for tmux_cc)
    # Dispatch based on llm_method:
    #   - raw_llm_api → BaselineRawLlmApiModel
    #   - coding_agent → BaselineCodingAgentModel
    #   - tmux_cc → BaselineDuccModel (supports interactive mode)
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Optional

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from experience.example.code_auto_encoder.baseline_raw_llm_api_model import BaselineRawLlmApiModel
from experience.example.code_auto_encoder.baseline_coding_agent_model import BaselineCodingAgentModel
from experience.example.code_auto_encoder.baseline_ducc_model import BaselineDuccModel


class BaselineAgentModel(nn.Module):
    """BaselineAgentModel from baseline_agent_model.viba.

    Dispatches to the appropriate model based on llm_method:
      - "raw_llm_api" → BaselineRawLlmApiModel
      - "coding_agent" → BaselineCodingAgentModel
      - "tmux_cc" → BaselineDuccModel (supports interactive tmux mode)

    Args:
        llm_method: One of "raw_llm_api", "coding_agent", or "tmux_cc".
        interactive: If True and llm_method="tmux_cc", run in tmux for visual observation.
        auto_confirm: If True (and interactive), auto-confirm prompts in tmux.
        tmux_session: Custom tmux session name (interactive mode only).
    """

    def __init__(
        self,
        llm_method: str = "raw_llm_api",
        interactive: bool = False,
        auto_confirm: bool = True,
        tmux_session: Optional[str] = None,
    ):
        super().__init__()
        self.llm_method = llm_method
        self.interactive = interactive

        # <- ({ match llm_method }
        #     <- { case raw_llm_api } <- Import[./baseline_raw_llm_api_model]
        #     <- { case coding_agent } <- Import[./baseline_coding_agent_model])
        if llm_method == "raw_llm_api":
            self._model = BaselineRawLlmApiModel()
        elif llm_method == "coding_agent":
            self._model = BaselineCodingAgentModel(llm_method="coding_agent")
        elif llm_method == "tmux_cc":
            self._model = BaselineDuccModel(
                interactive=interactive,
                auto_confirm=auto_confirm,
                tmux_session=tmux_session,
            )
        else:
            raise ValueError(f"Unknown llm_method: {llm_method}. Expected 'raw_llm_api', 'coding_agent', or 'tmux_cc'.")

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
