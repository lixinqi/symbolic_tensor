"""
RetrieveExperienceGradFn: autograd.Function for 2nd-derivative dispatch.

  forward  = pass-through (retrieval is a selection — grad passes to input)
  backward = 2nd-derivative dispatch via the active Policy
"""

import torch

from experience.future_tensor.backward_dispatch.backward_dispatcher import get_backward_dispatcher


class RetrieveExperienceGradFn(torch.autograd.Function):
    """autograd.Function whose forward is a pass-through that creates
    RetrieveExperienceGradFnBackward for 2nd-derivative dispatch."""

    @staticmethod
    def forward(ctx, grad_output, input_st, output_st, experience, indexes_map):
        from experience.future_tensor.function.retrieve_experience_backward import (
            retrieve_experience_backward,
        )

        ctx.save_for_backward(grad_output)
        ctx.input_st = input_st
        ctx.output_st = output_st
        ctx.experience = experience
        ctx.indexes_map = indexes_map
        ctx._retrieve_experience_backward_fn = retrieve_experience_backward
        ctx._grad_input = grad_output

        # Pass-through: retrieval is selection, gradient flows unchanged
        return grad_output + 0

    @staticmethod
    def backward(ctx, grad_grad_input):
        """2nd derivative: dispatch to the active Policy."""
        (grad_output,) = ctx.saved_tensors

        dispatch = get_backward_dispatcher(ctx._retrieve_experience_backward_fn)
        dispatch({
            "grad_output":  grad_output,
            "input":        ctx.input_st,
            "output":       ctx.output_st,
            "experience":   ctx.experience,
            "indexes_map":  ctx.indexes_map,
            "grad_input":   ctx._grad_input,
        })

        # Gradients for (grad_output, input_st, output_st, experience, indexes_map)
        return grad_grad_input, None, None, None, None
