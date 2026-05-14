"""
get_1st_dispatcher: returns a dispatch callable for 1st-derivative replacement.

The 1st-derivative dispatcher REPLACES the default backward behavior:
  - When no first_derivative_policy is active → returns False (caller runs default backward)
  - When a first_derivative_policy IS active → dispatches to policy and returns True
    (caller skips default backward)

This is a replacement, not an addition. The current backward behavior is the
default policy of get_1st_dispatcher.

Usage inside a GradFn.forward() (the 1st-derivative layer):

    dispatch_1st = get_1st_dispatcher(recurrent_backward)
    if dispatch_1st({"input": input, "output": output, ...}):
        return grad_output + 0   # policy handled it, skip actual backward
    # No policy active — run actual backward
    grad_input = recurrent_backward(...)
"""

from typing import Any, Callable, Dict

from experience.future_tensor.second_derivative.context import get_active_first_policy


def get_1st_dispatcher(fn: Callable) -> Callable[[Dict[str, Any]], bool]:
    """Return a dispatch callable for 1st-derivative replacement.

    When no first_derivative_policy is active, returns False (caller runs default).
    When a policy IS active, dispatches to it and returns True (caller skips default).

    Args:
        fn: The backward function object (e.g. recurrent_backward, st_moe_backward).

    Returns:
        A callable that accepts arg_name2inputs and returns True if the policy
        handled the dispatch (caller should skip default backward), or False
        if no policy is active (caller should run default backward).
    """
    def _dispatch(arg_name2inputs: Dict[str, Any]) -> bool:
        policy = get_active_first_policy()
        if policy is None:
            return False  # no policy → caller runs default backward
        policy.dispatch(fn, arg_name2inputs)
        return True  # policy handled it → caller skips default backward

    return _dispatch
