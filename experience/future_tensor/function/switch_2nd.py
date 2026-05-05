"""
SwitchGradFn: autograd.Function wrapping switch_backward.

  forward  = switch_backward  (1st derivative: route grad to selected branch)
  backward = 2nd-derivative dispatch via the active Policy

FtSwitch.backward() calls SwitchGradFn.apply(...) instead of switch_backward(...)
directly so that second_derivative_start.grad.backward() naturally triggers
SwitchGradFn.backward() (the 2nd-derivative dispatch).
"""

import torch
from typing import List

from experience.future_tensor.second_derivative.dispatcher import get_2nd_dispatcher


class SwitchGradFn(torch.autograd.Function):
    """autograd.Function whose forward IS switch_backward (the 1st derivative)."""

    @staticmethod
    def forward(ctx, grad_output, selected_index: int, branches: List):
        from experience.future_tensor.function.switch_backward import switch_backward

        ctx.save_for_backward(grad_output)
        ctx.selected_index = selected_index
        ctx.branches = branches
        ctx._switch_backward_fn = switch_backward
        ctx._grad_input = grad_output

        grad_input = switch_backward(grad_output, selected_index, branches)
        # Force creation of SwitchGradFnBackward by returning a new tensor.
        # Inside autograd.Function forward, ``+ 0`` is not tracked by autograd;
        # the returned tensor simply gets SwitchGradFnBackward as its grad_fn.
        return grad_input + 0

    @staticmethod
    def backward(ctx, grad_grad_input):
        """2nd derivative: dispatch to the active Policy."""
        (grad_output,) = ctx.saved_tensors

        dispatch = get_2nd_dispatcher(ctx._switch_backward_fn)
        dispatch({
            "grad_output":    grad_output,
            "selected_index": ctx.selected_index,
            "branches":       ctx.branches,
            "grad_input":     ctx._grad_input,
        })

        # Gradient for (grad_output, selected_index, branches)
        return grad_grad_input, None, None
