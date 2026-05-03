"""
RecurrentGradFn: autograd.Function wrapping recurrent_backward.

  forward  = recurrent_backward  (1st derivative)
  backward = 2nd-derivative dispatch via the active Policy

This is the canonical PyTorch higher-order derivative pattern.
FtRecurrent.backward() should call RecurrentGradFn.apply(...) instead of
recurrent_backward(...) directly so that second_derivative_start.grad.backward()
naturally triggers RecurrentGradFn.backward() (the 2nd-derivative dispatch).
"""

import torch
from typing import Any, Dict

from experience.future_tensor.second_derivative.dispatcher import get_2nd_dispatcher


class RecurrentGradFn(torch.autograd.Function):
    """autograd.Function whose forward IS recurrent_backward (the 1st derivative).

    By wrapping the 1st backward inside an autograd.Function, PyTorch's
    engine can call *this* class's backward() to compute the 2nd derivative,
    which dispatches to whatever Policy is currently active.
    """

    @staticmethod
    def forward(
        ctx,
        grad_output,
        input,
        output,
        prompt_tensor,
        # non-tensor kwargs are stored on ctx — pass as explicit args so
        # autograd sees the signature correctly
        topk_self_confidence_but_failed=8,
        grad_input_prompt=None,
        task_prompt="",
        llm_method="raw_llm_api",
        llm_env=None,
    ):
        from experience.future_tensor.function.ft_recurrent_backward import recurrent_backward

        ctx.save_for_backward(grad_output, input, output, prompt_tensor)
        ctx.task_prompt = task_prompt
        ctx.llm_method = llm_method
        ctx.llm_env = llm_env
        ctx.topk_self_confidence_but_failed = topk_self_confidence_but_failed
        ctx.grad_input_prompt = grad_input_prompt
        ctx._recurrent_backward_fn = recurrent_backward

        result = recurrent_backward(
            grad_output, input, output, prompt_tensor,
            topk_self_confidence_but_failed=topk_self_confidence_but_failed,
            grad_input_prompt=grad_input_prompt,
            task_prompt=task_prompt,
            llm_method=llm_method,
            llm_env=llm_env,
        )
        return result

    @staticmethod
    def backward(ctx, grad_grad_input):
        """2nd derivative: dispatch to the active Policy."""
        grad_output, input, output, prompt_tensor = ctx.saved_tensors
        recurrent_backward_fn = ctx._recurrent_backward_fn

        dispatch = get_2nd_dispatcher(recurrent_backward_fn)
        dispatch({
            "grad_output":   grad_output,
            "input":         input,
            "output":        output,
            "prompt_tensor": prompt_tensor,
            "task_prompt":   ctx.task_prompt,
            "llm_method":    ctx.llm_method,
            "llm_env":       ctx.llm_env,
        })

        # Gradients for (grad_output, input, output, prompt_tensor,
        #                topk_scbf, grad_input_prompt, task_prompt, llm_method, llm_env)
        return None, None, None, None, None, None, None, None, None


def recurrent_2nd_backward(
    grad_output,
    input,
    output,
    prompt_tensor,
    **kwargs,
) -> torch.Tensor:
    """Convenience wrapper: calls RecurrentGradFn.apply(...).

    Kept for test compatibility.  The canonical entry-point for
    FtRecurrent.backward() is RecurrentGradFn.apply() directly.
    """
    return RecurrentGradFn.apply(
        grad_output, input, output, prompt_tensor,
        kwargs.get("topk_self_confidence_but_failed", 8),
        kwargs.get("grad_input_prompt", None),
        kwargs.get("task_prompt", ""),
        kwargs.get("llm_method", "raw_llm_api"),
        kwargs.get("llm_env", None),
    )
