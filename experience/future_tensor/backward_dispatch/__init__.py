"""
experience.future_tensor.backward_dispatch
==========================================

Public API:

    need_reflection(input, backward_dispatch_start) -> torch.Tensor
    get_backward_dispatcher(fn) -> Callable
    dispatch_policy(policy) -> context manager
    TracePolicy(collector: list)
    Policy (base class)
    ReflectionRecord (dataclass)
    PolicyConflictError

autograd.Function wrappers (1st-backward -> 2nd-derivative dispatch):

    RecurrentGradFn         (wraps recurrent_backward)
    ExpertGradFn            (wraps st_moe_backward)
    SliceGradFn             (wraps slice_backward)
    UnsqueezeGradFn         (wraps unsqueeze squeeze-via-slice_forward)
    SequentialGradFn        (wraps sequential_backward)
    TmuxCreateSessionGradFn  (wraps tmux_create_session_backward)
    TmuxSendTextGradFn       (wraps tmux_send_text_backward)
    TmuxSendCtrlGradFn       (wraps tmux_send_ctrl_backward)
"""

from experience.future_tensor.backward_dispatch.need_reflection import need_reflection, ft_reflection_starter
from experience.future_tensor.backward_dispatch.backward_dispatcher import get_backward_dispatcher
from experience.future_tensor.backward_dispatch.context import dispatch_policy
from experience.future_tensor.backward_dispatch.trace_policy import TracePolicy
from experience.future_tensor.backward_dispatch.policy import (
    Policy,
    ReflectionRecord,
    PolicyConflictError,
)

__all__ = [
    "need_reflection",
    "ft_reflection_starter",
    "get_backward_dispatcher",
    "dispatch_policy",
    "TracePolicy",
    "Policy",
    "ReflectionRecord",
    "PolicyConflictError",
]
