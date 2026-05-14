"""
ExpandGradFn: autograd.Function wrapping expand_backward.

  forward  = expand_backward  (1st derivative: reduce along expanded dims)
  backward = 2nd-derivative dispatch via the active Policy

FtExpand.backward() calls ExpandGradFn.apply(...) instead of expand_backward(...)
directly so that second_derivative_start.grad.backward() naturally triggers
ExpandGradFn.backward() (the 2nd-derivative dispatch).
"""

import sympy
import torch
from typing import List

from experience.future_tensor.second_derivative.dispatcher import get_2nd_dispatcher
from experience.future_tensor.second_derivative.first_dispatcher import get_1st_dispatcher


class ExpandGradFn(torch.autograd.Function):
    """autograd.Function whose forward IS expand_backward (the 1st derivative)."""

    @staticmethod
    def forward(ctx, grad_output, input_shape: List[int], expanded_shape: List[int]):
        from experience.future_tensor.function.expand_backward import expand_backward
        from experience.future_tensor.future_tensor import FutureTensor
        from experience.future_tensor.status import Status

        # Reconstruct FutureTensor attributes if stripped by autograd
        if not hasattr(grad_output, "ft_static_tensor"):
            shape = expanded_shape
            # Use a dummy FutureTensor to reconstruct attributes
            async def dummy_get(coords, prompt):
                return ("", Status.confidence(0.0))

            ref_ft = FutureTensor(
                "/tmp", dummy_get, [sympy.Integer(s) for s in shape]
            )
            if grad_output.numel() == 1:
                if shape:
                    ref_ft.ft_static_tensor.data.flatten().fill_(grad_output.item())
                else:
                    ref_ft.ft_static_tensor.data.fill_(grad_output.item())
            else:
                ref_ft.ft_static_tensor.data.copy_(
                    grad_output.data.view(ref_ft.ft_static_tensor.shape)
                )
            ref_ft.ft_forwarded = True

            grad_output.ft_static_tensor = ref_ft.ft_static_tensor
            grad_output.ft_capacity_shape = ref_ft.ft_capacity_shape
            grad_output.ft_async_get = ref_ft.ft_async_get
            grad_output.ft_forwarded = ref_ft.ft_forwarded
            grad_output.ft_shape_schema = ref_ft.ft_shape_schema
            grad_output.ft_incremental_concated_tensors = ref_ft.ft_incremental_concated_tensors

        ctx.save_for_backward(grad_output)
        ctx.input_shape = input_shape
        ctx.expanded_shape = expanded_shape
        ctx._expand_backward_fn = expand_backward

        # 1st-derivative dispatch: policy replaces default backward
        dispatch_1st = get_1st_dispatcher(expand_backward)
        if dispatch_1st({"input_shape": input_shape, "expanded_shape": expanded_shape}):
            # Policy handled it — skip default backward
            ctx._grad_input = grad_output
            return grad_output + 0

        # No policy active — run actual backward (default behavior)
        grad_input = expand_backward(grad_output, input_shape, expanded_shape)
        ctx._grad_input = grad_input

        # Force creation of ExpandGradFnBackward
        return grad_input + 0

    @staticmethod
    def backward(ctx, grad_grad_input):
        """2nd derivative: dispatch to the active Policy."""
        (grad_output,) = ctx.saved_tensors

        dispatch = get_2nd_dispatcher(ctx._expand_backward_fn)
        dispatch({
            "grad_output":    grad_output,
            "input_shape":    ctx.input_shape,
            "expanded_shape": ctx.expanded_shape,
            "grad_input":     ctx._grad_input,
        })

        # Gradient for (grad_output, input_shape, expanded_shape)
        return grad_grad_input, None, None
