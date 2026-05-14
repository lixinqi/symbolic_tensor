"""
LlmCallGradFn: autograd.Function for 2nd-derivative dispatch.

  forward  = llm_call_backward_compute  (1st derivative)
  backward = 2nd-derivative dispatch via the active Policy

FtLlmCall.backward() calls LlmCallGradFn.apply(...) instead of
llm_call_backward_compute(...) directly so that
backward_dispatch_start.grad.backward() naturally triggers
LlmCallGradFn.backward() (the 2nd-derivative dispatch).
"""

import torch

from experience.future_tensor.backward_dispatch.backward_dispatcher import get_backward_dispatcher


class LlmCallGradFn(torch.autograd.Function):
    """autograd.Function whose forward IS llm_call_backward_compute (the 1st derivative).

    By wrapping the 1st backward inside an autograd.Function, PyTorch's
    engine can call *this* class's backward() to compute the 2nd derivative,
    which dispatches to whatever Policy is currently active.
    """

    @staticmethod
    def forward(
        ctx,
        grad_output,
        input_st,
        output_st,
        llm_method,
        llm_env,
    ):
        from experience.future_tensor.function.llm_call_backward import (
            llm_call_backward_compute,
            llm_call_backward,
        )

        ctx.save_for_backward(grad_output, input_st, output_st)
        ctx.llm_method = llm_method
        ctx.llm_env = llm_env
        ctx._llm_call_backward_fn = llm_call_backward

        # Run actual backward
        grad_input = llm_call_backward_compute(
            grad_output, input_st, output_st,
            llm_method, llm_env,
        )
        ctx._grad_input = grad_input

        return grad_input

    @staticmethod
    def backward(ctx, grad_grad_input):
        """2nd derivative: dispatch to the active Policy."""
        grad_output, input_st, output_st = ctx.saved_tensors

        dispatch = get_backward_dispatcher(ctx._llm_call_backward_fn)
        dispatch({
            "grad_output":  grad_output,
            "grad_input":   ctx._grad_input,
            "input":        input_st,
            "output":       output_st,
            "llm_method":   ctx.llm_method,
            "llm_env":      ctx.llm_env,
        })

        # Gradients for (grad_output, input_st, output_st,
        #                llm_method, llm_env)
        return None, None, None, None, None
