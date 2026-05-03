"""
UnsqueezeGradFn: autograd.Function wrapping the squeeze-via-slice_forward
used as the backward of unsqueeze.

  forward  = slice_forward(grad_output, squeeze_slices)  (1st derivative: squeeze)
  backward = 2nd-derivative dispatch via the active Policy

FtUnsqueeze.backward() calls UnsqueezeGradFn.apply(...) instead of slice_forward(...)
directly so that second_derivative_start.grad.backward() naturally triggers
UnsqueezeGradFn.backward() (the 2nd-derivative dispatch).
"""

import torch
from typing import List

from experience.future_tensor.second_derivative.dispatcher import get_2nd_dispatcher


class UnsqueezeGradFn(torch.autograd.Function):
    """autograd.Function whose forward IS the squeeze (slice_forward at the
    inserted dim) that is the 1st derivative of unsqueeze.

    autograd.Function attaches UnsqueezeGradFnBackward to whatever tensor
    forward() returns, so we simply return the FutureTensor from slice_forward()
    directly.
    """

    @staticmethod
    def forward(
        ctx,
        grad_output,
        dim: int,
        squeeze_slices: List,
    ):
        from experience.future_tensor.function.slice_forward import slice_forward
        from experience.future_tensor.function.unsqueeze_forward import unsqueeze_forward

        ctx.save_for_backward(grad_output)
        ctx.dim = dim
        ctx.squeeze_slices = squeeze_slices
        ctx._unsqueeze_forward_fn = unsqueeze_forward

        grad_input = slice_forward(grad_output, squeeze_slices)
        ctx._grad_input = grad_input
        return grad_input

    @staticmethod
    def backward(ctx, grad_grad_input):
        """2nd derivative: dispatch to the active Policy."""
        (grad_output,) = ctx.saved_tensors

        dispatch = get_2nd_dispatcher(ctx._unsqueeze_forward_fn)
        dispatch({
            "grad_output":   grad_output,
            "dim":           ctx.dim,
            "squeeze_slices": ctx.squeeze_slices,
            "grad_input":    ctx._grad_input,
        })

        # Gradient for (grad_output, dim, squeeze_slices)
        return grad_grad_input, None, None
