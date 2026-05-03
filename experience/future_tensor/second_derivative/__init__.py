"""
experience.future_tensor.second_derivative
==========================================

Public API:

    need_2nd_derivative(input, second_derivative_start) -> torch.Tensor
    get_2nd_dispatcher(fn) -> Callable
    dispatch_policy(policy) -> context manager
    TracePolicy(collector: list)
    Policy (base class)
    ReflectionRecord (dataclass)
    PolicyConflictError

autograd.Function wrappers (1st-backward → 2nd-derivative dispatch):

    RecurrentGradFn   (wraps recurrent_backward)
    MoeGradFn         (wraps st_moe_backward)
    SliceGradFn       (wraps slice_backward)
    UnsqueezeGradFn   (wraps unsqueeze squeeze-via-slice_forward)
"""

from experience.future_tensor.second_derivative.need_2nd_derivative import need_2nd_derivative
from experience.future_tensor.second_derivative.dispatcher import get_2nd_dispatcher
from experience.future_tensor.second_derivative.context import dispatch_policy
from experience.future_tensor.second_derivative.trace_policy import TracePolicy
from experience.future_tensor.second_derivative.policy import (
    Policy,
    ReflectionRecord,
    PolicyConflictError,
)

__all__ = [
    "need_2nd_derivative",
    "get_2nd_dispatcher",
    "dispatch_policy",
    "TracePolicy",
    "Policy",
    "ReflectionRecord",
    "PolicyConflictError",
]
