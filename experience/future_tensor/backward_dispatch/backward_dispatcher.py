"""
get_backward_dispatcher: unified backward dispatch for both 1st and 2nd derivatives.

Returns a dispatch callable that checks whether a policy is active:
  - If dispatch_policy is active -> dispatches there, returns True
  - If not -> returns False (caller runs default backward)

Usage in XXX.backward() (1st derivative):

    dispatch = get_backward_dispatcher(recurrent_backward)
    if dispatch({"input": input, "output": output, ...}):
        return grad_output  # policy handled it, skip GradFn.apply()

Usage in GradFn.backward() (2nd derivative):

    dispatch = get_backward_dispatcher(recurrent_backward)
    dispatch({"grad_output": ..., "input": ..., ...})
"""

from typing import Any, Callable, Dict

from experience.future_tensor.backward_dispatch.context import get_active_policy


def get_backward_dispatcher(fn: Callable) -> Callable[[Dict[str, Any]], bool]:
    """Return a dispatch callable for both 1st and 2nd derivative dispatch.

    The dispatcher checks whether a policy is currently active on the thread:
      - Active policy -> dispatches, returns True
      - No policy -> returns False (caller runs default backward)

    Args:
        fn: The backward function object (e.g. recurrent_backward, st_moe_backward).
            Used as a key by the policy to identify which op is being dispatched.

    Returns:
        A callable that accepts arg_name2inputs and returns True if a policy
        handled the dispatch (caller should skip default), or False if no
        policy is active (caller should run default backward).
    """
    def _dispatch(arg_name2inputs: Dict[str, Any]) -> bool:
        policy = get_active_policy()
        if policy is not None:
            policy.dispatch(fn, arg_name2inputs)
            return True
        return False

    return _dispatch
