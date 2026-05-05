"""
switch_backward :=
    torch.Tensor
    <- $grad_output torch.Tensor
    <- $selected_index int
    <- $branches list[FutureTensor]
    # inline

Backward pass of switch_forward: route grad_output to selected branch.
"""

from typing import List

import torch

from experience.future_tensor.future_tensor import FutureTensor


def switch_backward(grad_output: torch.Tensor, selected_index: int, branches: List[FutureTensor]) -> torch.Tensor:
    """Backward pass of switch_forward: route grad_output to selected branch.

    In practice this is an identity -- grad_output flows through to the selected
    branch. The non-selected branches receive None grad from FtSwitch.backward.

    Args:
        grad_output: Gradient from upstream.
        selected_index: Index of the branch selected during forward.
        branches: All branch FutureTensors from forward.

    Returns:
        grad_output (unchanged).
    """
    return grad_output
