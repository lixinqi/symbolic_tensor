"""
LlmQueryGradFn: autograd.Function for 2nd-derivative dispatch.

  forward  = llm_query_backward_compute  (1st derivative)
  backward = 2nd-derivative dispatch via the active Policy

FtLlmQuery.backward() calls LlmQueryGradFn.apply(...) instead of
llm_query_backward_compute(...) directly so that
second_derivative_start.grad.backward() naturally triggers
LlmQueryGradFn.backward() (the 2nd-derivative dispatch).
"""

import torch

from experience.future_tensor.second_derivative.dispatcher import get_2nd_dispatcher


class LlmQueryGradFn(torch.autograd.Function):
    """autograd.Function whose forward IS llm_query_backward_compute (the 1st derivative).

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
        system_prompt,
        task_prompt,
        llm_method,
        llm_env,
    ):
        from experience.future_tensor.function.llm_query_backward import (
            llm_query_backward_compute,
            llm_query_backward,
        )

        ctx.save_for_backward(grad_output, input_st, output_st)
        ctx.system_prompt = system_prompt
        ctx.task_prompt = task_prompt
        ctx.llm_method = llm_method
        ctx.llm_env = llm_env
        ctx._llm_query_backward_fn = llm_query_backward

        grad_input = llm_query_backward_compute(
            grad_output, input_st, output_st,
            system_prompt, task_prompt, llm_method, llm_env,
        )
        ctx._grad_input = grad_input
        return grad_input

    @staticmethod
    def backward(ctx, grad_grad_input):
        """2nd derivative: dispatch to the active Policy."""
        grad_output, input_st, output_st = ctx.saved_tensors

        dispatch = get_2nd_dispatcher(ctx._llm_query_backward_fn)
        dispatch({
            "grad_output":   grad_output,
            "grad_input":    ctx._grad_input,
            "input":         input_st,
            "output":        output_st,
            "system_prompt": ctx.system_prompt,
            "task_prompt":   ctx.task_prompt,
            "llm_method":    ctx.llm_method,
            "llm_env":       ctx.llm_env,
        })

        # Gradients for (grad_output, input_st, output_st,
        #                system_prompt, task_prompt, llm_method, llm_env)
        return None, None, None, None, None, None, None
