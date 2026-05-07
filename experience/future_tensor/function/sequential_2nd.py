"""
SequentialGradFn: autograd.Function wrapping sequential_backward.

  forward  = sequential_backward  (1st derivative)
  backward = 2nd-derivative dispatch via the active Policy

FtSequential.backward() calls SequentialGradFn.apply(...) instead of
sequential_backward(...) directly so that
second_derivative_start.grad.backward() naturally triggers
SequentialGradFn.backward() (the 2nd-derivative dispatch).
"""

import torch

from experience.future_tensor.second_derivative.dispatcher import get_2nd_dispatcher


class SequentialGradFn(torch.autograd.Function):
    """autograd.Function whose forward IS sequential_backward (the 1st derivative)."""

    @staticmethod
    def forward(ctx, grad_output, num_inputs: int):
        from experience.future_tensor.function.sequential_backward import sequential_backward

        ctx.save_for_backward(grad_output)
        ctx.num_inputs = num_inputs
        ctx._sequential_backward_fn = sequential_backward
        ctx._grad_input = grad_output

        grad_input = sequential_backward(grad_output, num_inputs)
        # Force creation of SequentialGradFnBackward by returning a new tensor.
        # Inside autograd.Function forward, ``+ 0`` is not tracked by autograd;
        # the returned tensor simply gets SequentialGradFnBackward as its grad_fn.
        return grad_input + 0

    @staticmethod
    def backward(ctx, grad_grad_input):
        """2nd derivative: dispatch to the active Policy."""
        (grad_output,) = ctx.saved_tensors

        dispatch = get_2nd_dispatcher(ctx._sequential_backward_fn)
        dispatch({
            "grad_output": grad_output,
            "num_inputs":  ctx.num_inputs,
            "grad_input":  ctx._grad_input,
        })

        # Gradient for (grad_output, num_inputs)
        return grad_grad_input, None
