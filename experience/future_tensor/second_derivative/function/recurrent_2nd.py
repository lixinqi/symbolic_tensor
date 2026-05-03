"""
2nd-derivative wrapper for recurrent_backward.
"""

import torch

from experience.future_tensor.second_derivative.dispatcher import get_2nd_dispatcher


def recurrent_2nd_backward(
    grad_output,
    input,
    output,
    prompt_tensor,
    **kwargs,
) -> torch.Tensor:
    """2nd-derivative op for recurrent_backward.

    Dispatches to the active policy with the same named arguments as
    recurrent_backward, then returns a placeholder scalar tensor.

    Args:
        grad_output: The 1st-derivative output (LLM reflection text diff).
        input: The original forward input passed to recurrent_backward.
        output: The original forward output passed to recurrent_backward.
        prompt_tensor: The prompt tensor passed to recurrent_backward.
        **kwargs: Any extra kwargs forwarded from recurrent_backward.

    Returns:
        Placeholder scalar tensor (value=1).
    """
    from experience.future_tensor.function.ft_recurrent_backward import recurrent_backward

    dispatch = get_2nd_dispatcher(recurrent_backward)
    dispatch({
        "grad_output":   grad_output,
        "input":         input,
        "output":        output,
        "prompt_tensor": prompt_tensor,
        **kwargs,
    })
    return torch.ones(())
