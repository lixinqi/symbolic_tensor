"""
need_2nd_derivative: enable gradient flow through the input for 2nd-derivative pass.
"""

import torch


def need_2nd_derivative(
    input: torch.Tensor,
    second_derivative_start: torch.nn.Parameter,
) -> torch.Tensor:
    """Return input with requires_grad=True.

    Sets requires_grad=True on input so that the autograd graph connects it to
    second_derivative_start through the placeholder scalars returned by each
    2nd-derivative op. Calling second_derivative_start.grad.backward() will
    then traverse those ops.

    Has no other side effects — does not register anything in thread-local state,
    does not modify the harness-model autograd graph.

    Args:
        input: FutureTensor or SymbolicTensor.
        second_derivative_start: Scalar nn.Parameter acting as the entry point
            for the 2nd-derivative backward pass.

    Returns:
        input with requires_grad=True.
    """
    input.requires_grad_(True)
    return input
