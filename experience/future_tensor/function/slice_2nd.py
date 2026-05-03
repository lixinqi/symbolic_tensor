"""
SliceGradFn: autograd.Function wrapping slice_backward.

  forward  = slice_backward  (1st derivative: scatter grad back to original positions)
  backward = 2nd-derivative dispatch via the active Policy

FtSlice.backward() calls SliceGradFn.apply(...) instead of slice_backward(...)
directly so that second_derivative_start.grad.backward() naturally triggers
SliceGradFn.backward() (the 2nd-derivative dispatch).
"""

import torch
from typing import List, Union

from experience.future_tensor.second_derivative.dispatcher import get_2nd_dispatcher


class SliceGradFn(torch.autograd.Function):
    """autograd.Function whose forward IS slice_backward (the 1st derivative).

    autograd.Function attaches SliceGradFnBackward to whatever tensor forward()
    returns, so we simply return the FutureTensor from slice_backward() directly.
    """

    @staticmethod
    def forward(
        ctx,
        grad_output,
        original_shape: List[int],
        slices: List[Union[int, slice]],
    ):
        from experience.future_tensor.function.slice_backward import slice_backward

        ctx.save_for_backward(grad_output)
        ctx.original_shape = original_shape
        ctx.slices = slices
        ctx._slice_backward_fn = slice_backward

        grad_input = slice_backward(grad_output, original_shape, slices)
        ctx._grad_input = grad_input
        return grad_input

    @staticmethod
    def backward(ctx, grad_grad_input):
        """2nd derivative: dispatch to the active Policy."""
        (grad_output,) = ctx.saved_tensors

        dispatch = get_2nd_dispatcher(ctx._slice_backward_fn)
        dispatch({
            "grad_output":    grad_output,
            "original_shape": ctx.original_shape,
            "slices":         ctx.slices,
            "grad_input":     ctx._grad_input,
        })

        # Gradient for (grad_output, original_shape, slices)
        return grad_grad_input, None, None
