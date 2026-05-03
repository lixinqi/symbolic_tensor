"""
need_2nd_derivative: enable gradient flow through the input for 2nd-derivative pass.
"""

import torch


_FUTURE_TENSOR_ATTRS = [
    "ft_static_tensor",
    "ft_incremental_concated_tensors",
    "ft_shape_schema",
    "ft_capacity_shape",
    "ft_forwarded",
    "ft_async_get",
    "ft_forward",
    "ft_get_materialized_value",
    "ft_reset_materialized_value",
]


class _Need2ndDerivative(torch.autograd.Function):
    """Multiplies input by second_derivative_start to create a computational dependency.

    The forward returns ``input * second_derivative_start`` so that during
    ``loss.backward(create_graph=True)`` the gradient computation for
    ``second_derivative_start`` includes the entire backward graph of the model.
    Calling ``second_derivative_start.grad.backward()`` then naturally traverses
    that graph and triggers the 2nd-derivative GradFn backward methods.
    """

    @staticmethod
    def forward(ctx, input, second_derivative_start):
        ctx.save_for_backward(second_derivative_start)
        # Preserve dtype to avoid unwanted up-casts.
        sds = second_derivative_start.to(input.dtype)
        return input * sds

    @staticmethod
    def backward(ctx, grad_output):
        second_derivative_start, = ctx.saved_tensors
        # grad for input: pass-through scaled by second_derivative_start
        grad_input = grad_output * second_derivative_start.to(grad_output.dtype)
        # grad for second_derivative_start: a scalar so the graph continues.
        # Cast to the parameter's dtype so PyTorch doesn't insert CopyBackwards
        # (which would truncate the grad_fn chain).
        grad_sds = (grad_output.sum() * 0).to(second_derivative_start.dtype)
        return grad_input, grad_sds


def need_2nd_derivative(
    input: torch.Tensor,
    second_derivative_start: torch.Tensor,
) -> torch.Tensor:
    """Return input with a computational dependency on second_derivative_start.

    This sets up the graph so that ``second_derivative_start.grad.backward()``
    traverses the model's backward computations and triggers the 2nd-derivative
    Policy dispatches.

    If ``input`` carries FutureTensor monkey-patched attributes, they are copied
    to the result so downstream ops still see a valid FutureTensor.

    Args:
        input: FutureTensor or SymbolicTensor.  Must be scalar (shape ``()``).
        second_derivative_start: Scalar tensor (typically created with
            ``torch.ones((), dtype=torch.bfloat16, requires_grad=True)``)
            acting as the entry point for the 2nd-derivative backward pass.

    Returns:
        A tensor with the same value and monkey-patched attributes as ``input``,
        but with a ``grad_fn`` that connects it to ``second_derivative_start``.
    """
    assert input.shape == torch.Size([]), (
        f"need_2nd_derivative: input must be scalar, got shape {list(input.shape)}"
    )
    assert second_derivative_start.shape == torch.Size([]), (
        f"need_2nd_derivative: second_derivative_start must be scalar, "
        f"got shape {list(second_derivative_start.shape)}"
    )
    assert second_derivative_start.requires_grad, (
        "need_2nd_derivative: second_derivative_start must have requires_grad=True"
    )

    result = _Need2ndDerivative.apply(input, second_derivative_start)

    # Copy FutureTensor attributes so the result is still a valid FutureTensor.
    for attr in _FUTURE_TENSOR_ATTRS:
        if hasattr(input, attr):
            setattr(result, attr, getattr(input, attr))

    return result
