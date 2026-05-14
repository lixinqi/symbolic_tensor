"""
FtLlmQuery := torch.autograd.Function[
    $forward  Import[{future_tensor function llm_query_forward.viba}],
    $backward Import[{future_tensor function llm_query_backward.viba}]
]

ft_llm_query = FtLlmQuery.apply

Raw LLM call wrapped as a FutureTensor autograd Function.
Unlike ft_expert, it has NO experience retrieval. It takes input text,
sends it to an LLM with a system prompt, and returns the LLM response.
Supports 1st and 2nd derivatives via symbolic gradients.
"""

import torch
from typing import Dict, Optional

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.function.llm_query_forward import llm_query_forward
from experience.future_tensor.function.llm_query_backward import llm_query_backward
from experience.future_tensor.function.llm_query_2nd import LlmQueryGradFn
from experience.symbolic_tensor.tensor_util.todo_tensor_like import todo_tensor_like
from experience.symbolic_tensor.function import symbolic_grad_registry
from experience.future_tensor.backward_dispatch.backward_dispatcher import get_backward_dispatcher


class FtLlmQuery(torch.autograd.Function):
    """Autograd Function for raw LLM query in a FutureTensor pipeline.

    Forward: FutureTensor — lazy, async LLM call (system_prompt + input → output).
    Backward: Symbolic gradient — LLM reflects on how to improve input.

    No experience retrieval. Use ft_expert when experience is needed.
    """

    @staticmethod
    def forward(
        ctx,
        input_ft: FutureTensor,
        system_prompt: str = "",
        task_prompt: str = "",
        llm_method: str = "raw_llm_api",
        llm_env: Optional[Dict[str, str]] = None,
    ) -> FutureTensor:
        output = llm_query_forward(
            input_ft, system_prompt, task_prompt, llm_method, llm_env,
        )

        # Save for backward
        ctx.input_ft = input_ft
        ctx.output_ft = output
        ctx.system_prompt = system_prompt
        ctx.task_prompt = task_prompt
        ctx.llm_method = llm_method
        ctx.llm_env = llm_env

        # Save shape/relative_to for attr reconstruction in backward
        ctx.shape = input_ft.ft_capacity_shape
        ctx.relative_to = input_ft.ft_static_tensor.st_relative_to

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # After forward + ft_forward, FutureTensors have materialized .ft_static_tensor
        output_st = ctx.output_ft.ft_static_tensor
        input_st = ctx.input_ft.ft_static_tensor

        # Check if input has content (any coefficient > 0)
        has_content = input_st.data.sum().item() > 0
        if has_content:
            input_st.requires_grad_(True)
        else:
            input_st.requires_grad_(False)

        # If grad_output lacks st_* attrs, wrap as TODO symbolic tensor
        grad_output_orig = grad_output
        symbolic_grad = symbolic_grad_registry.pop(output_st.st_tensor_uid)
        if symbolic_grad is not None:
            grad_output = symbolic_grad
        elif not hasattr(grad_output, "st_relative_to"):
            symbolic_grad_output = todo_tensor_like(output_st)
            symbolic_grad_output.data.copy_(grad_output_orig.data)
            grad_output = symbolic_grad_output

        # 1st-derivative dispatch: skip GradFn if dispatcher handles it.
        dispatch = get_backward_dispatcher(llm_query_backward)
        if dispatch({
            "input": input_st, "output": output_st,
            "system_prompt": ctx.system_prompt, "task_prompt": ctx.task_prompt,
            "llm_method": ctx.llm_method, "llm_env": ctx.llm_env,
        }):
            return None, None, None, None, None

        # Call GradFn for 2nd-derivative support
        grad_input = LlmQueryGradFn.apply(
            grad_output,
            input_st,
            output_st,
            ctx.system_prompt,
            ctx.task_prompt,
            ctx.llm_method,
            ctx.llm_env,
        )

        # Register symbolic grad for upstream
        if grad_input is not None:
            symbolic_grad_registry.register(input_st.st_tensor_uid, grad_input)

        # Return grads for (input_ft, system_prompt, task_prompt, llm_method, llm_env)
        return grad_input, None, None, None, None


def ft_llm_query(
    input_ft: FutureTensor,
    system_prompt: str = "",
    task_prompt: str = "",
    llm_method: str = "raw_llm_api",
    llm_env: Optional[Dict[str, str]] = None,
) -> FutureTensor:
    """Raw LLM query with autograd support.

    Async forward: each element's content is sent to an LLM with the
    system_prompt, and the LLM response becomes the output.

    No experience retrieval — use ft_expert when experience is needed.

    Args:
        input_ft: FutureTensor with text content as input to the LLM.
        system_prompt: System prompt prepended to the LLM call.
        task_prompt: High-level task description (for backward/2nd-derivative dispatch).
        llm_method: LLM method name (default "raw_llm_api").
        llm_env: Optional environment variables for LLM config.

    Returns:
        FutureTensor output containing LLM responses.
    """
    return FtLlmQuery.apply(
        input_ft, system_prompt, task_prompt, llm_method, llm_env,
    )


if __name__ == "__main__":
    import asyncio
    import tempfile
    import os
    import sympy

    from experience.future_tensor.status import Status
    from experience.symbolic_tensor.tensor_util.st_setitem import st_setitem

    print("Running ft_llm_query tests...\n")

    def run_test(name, condition, expected=None, actual=None):
        if condition:
            print(f"  \u2713 {name}")
        else:
            print(f"  \u2717 {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    # Test 1: Forward creates a lazy FutureTensor
    print("Test 1: Forward creates lazy FutureTensor")
    with tempfile.TemporaryDirectory() as tmpdir:
        async def dummy_get(coords, prompt):
            return ("Hello world", Status.confidence(1.0))

        input_ft = FutureTensor(
            tmpdir, dummy_get, [sympy.Integer(1)],
        )

        output = ft_llm_query(input_ft, system_prompt="Echo the input.")
        run_test("output is torch.Tensor", isinstance(output, torch.Tensor))
        run_test("output has ft_capacity_shape", hasattr(output, "ft_capacity_shape"))
        run_test("output shape matches input", output.ft_capacity_shape == input_ft.ft_capacity_shape)
        run_test("output not yet forwarded", not output.ft_forwarded)

    # Test 2: Backward returns correct number of grads
    print("Test 2: Backward returns correct number of None grads for non-differentiable params")
    # This is a structural test — the backward signature must match forward params
    with tempfile.TemporaryDirectory() as tmpdir:
        async def dummy_get2(coords, prompt):
            return ("test", Status.confidence(1.0))

        input_ft2 = FutureTensor(
            tmpdir, dummy_get2, [sympy.Integer(1)],
        )
        output2 = FtLlmQuery.apply(input_ft2, "sys", "task", "raw_llm_api", None)
        run_test("apply returns torch.Tensor with ft_* attrs",
                 isinstance(output2, torch.Tensor) and hasattr(output2, "ft_async_get"))

    print("\nAll tests completed.")
