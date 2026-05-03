"""
2nd-derivative wrapper for st_moe_backward.
"""

import torch

from experience.future_tensor.second_derivative.dispatcher import get_2nd_dispatcher


def moe_2nd_backward(
    grad_output,
    input,
    output,
    experience,
    selected_experience_qkv_indexes_list,
    **kwargs,
) -> torch.Tensor:
    """2nd-derivative op for st_moe_backward.

    Dispatches to the active policy with the same named arguments as
    st_moe_backward, then returns a placeholder scalar tensor.

    Args:
        grad_output: The 1st-derivative output (LLM reflection text diff).
        input: The original forward input passed to st_moe_backward.
        output: The original forward output passed to st_moe_backward.
        experience: The experience tensor passed to st_moe_backward.
        selected_experience_qkv_indexes_list: The indexes list passed to st_moe_backward.
        **kwargs: Any extra kwargs forwarded from st_moe_backward (e.g. context).

    Returns:
        Placeholder scalar tensor (value=1).
    """
    from experience.symbolic_tensor.function.st_moe_backward import st_moe_backward

    dispatch = get_2nd_dispatcher(st_moe_backward)
    dispatch({
        "grad_output":                         grad_output,
        "input":                               input,
        "output":                              output,
        "experience":                          experience,
        "selected_experience_qkv_indexes_list": selected_experience_qkv_indexes_list,
        **kwargs,
    })
    return torch.ones(())
