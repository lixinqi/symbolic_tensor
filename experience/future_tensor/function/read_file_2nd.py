"""
ReadFileGradFn: autograd.Function for 2nd-derivative dispatch.

  forward  = pass-through (creates the backward node)
  backward = 2nd-derivative dispatch via the active Policy
"""

import torch

from experience.future_tensor.second_derivative.dispatcher import get_2nd_dispatcher
from experience.future_tensor.second_derivative.first_dispatcher import get_1st_dispatcher


class ReadFileGradFn(torch.autograd.Function):
    """autograd.Function whose forward is a pass-through that creates
    ReadFileGradFnBackward for 2nd-derivative dispatch."""

    @staticmethod
    def forward(ctx, grad_output):
        from experience.future_tensor.function.read_file_backward import (
            read_file_backward,
        )

        ctx.save_for_backward(grad_output)
        ctx._read_file_backward_fn = read_file_backward
        ctx._grad_input = grad_output

        # 1st-derivative dispatch: policy replaces default backward
        dispatch_1st = get_1st_dispatcher(read_file_backward)
        dispatch_1st({})  # pass-through: result is grad_output + 0 either way

        return grad_output + 0

    @staticmethod
    def backward(ctx, grad_grad_input):
        """2nd derivative: dispatch to the active Policy."""
        (grad_output,) = ctx.saved_tensors

        dispatch = get_2nd_dispatcher(ctx._read_file_backward_fn)
        dispatch({
            "grad_output": grad_output,
            "grad_input":  ctx._grad_input,
        })

        return grad_grad_input
