"""
BuildExpertContextGradFn: autograd.Function for 2nd-derivative dispatch.

  forward  = pass-through (context construction is deterministic concatenation)
  backward = 2nd-derivative dispatch via the active Policy
"""

import torch

from experience.future_tensor.backward_dispatch.backward_dispatcher import get_backward_dispatcher


class BuildExpertContextGradFn(torch.autograd.Function):
    """autograd.Function whose forward is a pass-through that creates
    BuildExpertContextGradFnBackward for 2nd-derivative dispatch."""

    @staticmethod
    def forward(ctx, grad_output, input_st, experience_text_st, output_st, task_prompt_st):
        from experience.future_tensor.function.build_expert_context_backward import (
            build_expert_context_backward,
        )

        ctx.save_for_backward(grad_output)
        ctx.input_st = input_st
        ctx.experience_text_st = experience_text_st
        ctx.output_st = output_st
        ctx.task_prompt_st = task_prompt_st
        ctx._build_expert_context_backward_fn = build_expert_context_backward
        ctx._grad_input = grad_output

        # Pass-through: concatenation is deterministic, gradient flows unchanged
        return grad_output + 0

    @staticmethod
    def backward(ctx, grad_grad_input):
        """2nd derivative: dispatch to the active Policy."""
        (grad_output,) = ctx.saved_tensors

        dispatch = get_backward_dispatcher(ctx._build_expert_context_backward_fn)
        dispatch({
            "grad_output":      grad_output,
            "input":            ctx.input_st,
            "experience_text":  ctx.experience_text_st,
            "output":           ctx.output_st,
            "task_prompt":      ctx.task_prompt_st,
            "grad_input":       ctx._grad_input,
        })

        # Gradients for (grad_output, input_st, experience_text_st, output_st, task_prompt_st)
        return grad_grad_input, None, None, None, None
