"""
sequential_backward :=
    torch.Tensor
    <- $grad_output torch.Tensor
    <- $num_inputs int
    # inline

Backward pass of sequential_forward: route grad_output through.
"""

import torch


def sequential_backward(grad_output: torch.Tensor, num_inputs: int) -> torch.Tensor:
    """Backward pass of sequential_forward: identity.

    The actual gradient routing to each input is handled by FtSequential.backward,
    which returns a tuple of grads. This function exists primarily as the
    2nd-derivative dispatch key.

    Args:
        grad_output: Gradient from upstream.
        num_inputs: Number of inputs (for record-keeping only).

    Returns:
        grad_output (unchanged).
    """
    return grad_output
