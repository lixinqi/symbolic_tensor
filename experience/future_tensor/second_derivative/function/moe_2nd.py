"""
MoeGradFn: autograd.Function wrapping st_moe_backward.

  forward  = st_moe_backward  (1st derivative)
  backward = 2nd-derivative dispatch via the active Policy

This is the canonical PyTorch higher-order derivative pattern.
FtMoe.backward() should call MoeGradFn.apply(...) instead of
st_moe_backward(...) directly so that second_derivative_start.grad.backward()
naturally triggers MoeGradFn.backward() (the 2nd-derivative dispatch).
"""

import torch

from experience.future_tensor.second_derivative.dispatcher import get_2nd_dispatcher


class MoeGradFn(torch.autograd.Function):
    """autograd.Function whose forward IS st_moe_backward (the 1st derivative).

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
        experience,
        # selected_experience_qkv_indexes_list is not a tensor, stash on ctx
        # non-tensor kwargs stored on ctx as well
        task_prompt="",
        topk=16,
        llm_method="raw_llm_api",
        llm_env=None,
        context=None,
        grad_input_prompt=None,
        grad_exp_key_prompt=None,
        grad_exp_value_prompt=None,
        selected_experience_qkv_indexes_list=None,
    ):
        from experience.symbolic_tensor.function.st_moe_backward import st_moe_backward

        ctx.save_for_backward(grad_output, input, output, experience)
        ctx.selected_experience_qkv_indexes_list = selected_experience_qkv_indexes_list
        ctx.task_prompt = task_prompt
        ctx.topk = topk
        ctx.llm_method = llm_method
        ctx.llm_env = llm_env
        ctx.context = context
        ctx.grad_input_prompt = grad_input_prompt
        ctx.grad_exp_key_prompt = grad_exp_key_prompt
        ctx.grad_exp_value_prompt = grad_exp_value_prompt
        ctx._st_moe_backward_fn = st_moe_backward

        grad_input, grad_experience = st_moe_backward(
            grad_output, input, output, experience,
            selected_experience_qkv_indexes_list,
            grad_input_prompt=grad_input_prompt,
            grad_exp_key_prompt=grad_exp_key_prompt,
            grad_exp_value_prompt=grad_exp_value_prompt,
            task_prompt=task_prompt,
            topk=topk,
            llm_method=llm_method,
            llm_env=llm_env,
            context=context,
        )
        return grad_input, grad_experience

    @staticmethod
    def backward(ctx, grad_grad_input, grad_grad_experience):
        """2nd derivative: dispatch to the active Policy."""
        grad_output, input, output, experience = ctx.saved_tensors
        st_moe_backward_fn = ctx._st_moe_backward_fn

        dispatch = get_2nd_dispatcher(st_moe_backward_fn)
        dispatch({
            "grad_output":   grad_output,
            "input":         input,
            "output":        output,
            "experience":    experience,
            "selected_experience_qkv_indexes_list": ctx.selected_experience_qkv_indexes_list,
            "task_prompt":   ctx.task_prompt,
            "llm_method":    ctx.llm_method,
            "llm_env":       ctx.llm_env,
        })

        # Gradients for (grad_output, input, output, experience,
        #                task_prompt, topk, llm_method, llm_env, context,
        #                grad_input_prompt, grad_exp_key_prompt, grad_exp_value_prompt,
        #                selected_experience_qkv_indexes_list)
        return None, None, None, None, None, None, None, None, None, None, None, None, None


def moe_2nd_backward(
    grad_output,
    input,
    output,
    experience,
    selected_experience_qkv_indexes_list,
    **kwargs,
) -> tuple:
    """Convenience wrapper: calls MoeGradFn.apply(...).

    Kept for test compatibility.  The canonical entry-point for
    FtMoe.backward() is MoeGradFn.apply() directly.
    """
    return MoeGradFn.apply(
        grad_output, input, output, experience,
        kwargs.get("task_prompt", ""),
        kwargs.get("topk", 16),
        kwargs.get("llm_method", "raw_llm_api"),
        kwargs.get("llm_env", None),
        kwargs.get("context", None),
        kwargs.get("grad_input_prompt", None),
        kwargs.get("grad_exp_key_prompt", None),
        kwargs.get("grad_exp_value_prompt", None),
        selected_experience_qkv_indexes_list,
    )
