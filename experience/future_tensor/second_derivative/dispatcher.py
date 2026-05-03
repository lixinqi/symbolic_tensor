"""
get_2nd_dispatcher: returns a dispatch callable keyed on a 1st-derivative fn object.

Usage inside a 2nd-derivative wrapper:

    from experience.future_tensor.function.ft_recurrent_backward import recurrent_backward
    dispatch = get_2nd_dispatcher(recurrent_backward)
    dispatch({"grad_output": ..., "input": ..., ...})
    return torch.ones(())
"""

from typing import Any, Callable, Dict

import torch

from experience.future_tensor.second_derivative.context import get_active_policy


def get_2nd_dispatcher(fn: Callable) -> Callable[[Dict[str, Any]], torch.Tensor]:
    """Return a dispatch callable for the given 1st-derivative backward function.

    Args:
        fn: The 1st-derivative backward function object (e.g. recurrent_backward).

    Returns:
        A callable that accepts arg_name2inputs and fires the active policy.
    """
    def _dispatch(arg_name2inputs: Dict[str, Any]) -> torch.Tensor:
        policy = get_active_policy()
        return policy.dispatch(fn, arg_name2inputs)

    return _dispatch
