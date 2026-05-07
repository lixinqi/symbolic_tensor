"""
tmux_send_text_backward :=
    FutureTensor
    <- $grad_output FutureTensor
    # inline

Backward for ft_tmux_send_text.
Reconstructs FutureTensor attributes on grad_output (stripped by autograd),
enables requires_grad for 2nd-derivative graph recording, and calls
TmuxSendTextGradFn.apply.
"""

from typing import List

import sympy

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status
from experience.future_tensor.function.tmux_send_text_2nd import TmuxSendTextGradFn


def tmux_send_text_backward(ctx, grad_output) -> FutureTensor:
    """Backward for ft_tmux_send_text: reconstruct attrs + GradFn."""
    if not hasattr(grad_output, "ft_static_tensor"):
        shape: List[int] = ctx.shape
        relative_to: str = ctx.relative_to

        async def dummy_get(coords, prompt):
            return ("", Status.confidence(0.0))

        ref_ft = FutureTensor(relative_to, dummy_get, [sympy.Integer(s) for s in shape])
        if grad_output.numel() == 1:
            if shape:
                ref_ft.ft_static_tensor.data.flatten().fill_(grad_output.item())
            else:
                ref_ft.ft_static_tensor.data.fill_(grad_output.item())
        else:
            ref_ft.ft_static_tensor.data.copy_(grad_output.data.view(ref_ft.ft_static_tensor.shape))
        ref_ft.ft_forwarded = True

        grad_output.ft_static_tensor = ref_ft.ft_static_tensor
        grad_output.ft_capacity_shape = ref_ft.ft_capacity_shape
        grad_output.ft_async_get = ref_ft.ft_async_get
        grad_output.ft_forwarded = ref_ft.ft_forwarded
        grad_output.ft_shape_schema = ref_ft.ft_shape_schema
        grad_output.ft_incremental_concated_tensors = ref_ft.ft_incremental_concated_tensors

    if not grad_output.requires_grad:
        grad_output.requires_grad_(True)

    return TmuxSendTextGradFn.apply(grad_output)
