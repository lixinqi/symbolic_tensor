"""
sleep_forward :=
    FutureTensor
    <- $input FutureTensor
    <- $seconds float
    # inline

Async sleep — pauses the pipeline for the given duration.
"""

from typing import List

import sympy

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status


def sleep_forward(input_ft: FutureTensor, seconds: float) -> FutureTensor:
    """Forward: create a lazy FutureTensor whose ft_async_get sleeps then passes through."""
    shape = input_ft.ft_capacity_shape
    relative_to = input_ft.ft_static_tensor.st_relative_to

    async def sleep_async_get(coordinates: List[int], prompt: str):
        import asyncio
        await asyncio.sleep(seconds)
        return ("", Status.confidence(1.0))

    result = FutureTensor(
        relative_to,
        sleep_async_get,
        [sympy.Integer(s) for s in shape],
    )
    result.ft_capacity_shape = list(shape)
    return result
