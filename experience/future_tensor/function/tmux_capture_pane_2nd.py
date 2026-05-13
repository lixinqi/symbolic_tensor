"""
TmuxCapturePaneGradFn: autograd.Function for 2nd-derivative dispatch.

  forward  = pass-through (creates the backward node)
  backward = 2nd-derivative dispatch via the active Policy
"""

import torch

from experience.future_tensor.second_derivative.dispatcher import get_2nd_dispatcher


class TmuxCapturePaneGradFn(torch.autograd.Function):
    """autograd.Function whose forward is a pass-through that creates
    TmuxCapturePaneGradFnBackward for 2nd-derivative dispatch."""

    @staticmethod
    def forward(ctx, grad_output):
        from experience.future_tensor.function.tmux_capture_pane_backward import (
            tmux_capture_pane_backward,
        )

        ctx.save_for_backward(grad_output)
        ctx._tmux_capture_pane_backward_fn = tmux_capture_pane_backward
        ctx._grad_input = grad_output
        return grad_output + 0

    @staticmethod
    def backward(ctx, grad_grad_input):
        """2nd derivative: dispatch to the active Policy."""
        (grad_output,) = ctx.saved_tensors

        dispatch = get_2nd_dispatcher(ctx._tmux_capture_pane_backward_fn)
        dispatch({
            "grad_output": grad_output,
            "grad_input":  ctx._grad_input,
        })

        return grad_grad_input
