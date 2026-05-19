"""
ft_ps1_shows_up: Check whether a PS1 prompt pattern appears in captured pane text.

Pure runtime op (no autograd). Returns "true" if PS1 pattern is found in the
capture text, "false" otherwise. Used to conditionally skip sleep when the
terminal is already idle.
"""

from typing import List

import sympy

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status


def ft_ps1_shows_up(ps1_pattern_ft: FutureTensor, capture_ft: FutureTensor) -> FutureTensor:
    """Check whether PS1 prompt pattern appears in captured terminal output.

    Args:
        ps1_pattern_ft: FutureTensor whose elements contain PS1 patterns to search for.
                        Broadcastable to capture_ft shape.
        capture_ft: FutureTensor whose elements contain captured pane text.

    Returns:
        A lazy FutureTensor with the same shape as capture_ft. Each element
        is "true" if the PS1 pattern is found in the capture text, "false" otherwise.
    """
    shape = capture_ft.ft_capacity_shape
    relative_to = capture_ft.ft_static_tensor.st_relative_to

    async def ps1_check_get(coordinates: List[int], trajactory: str):
        # Read capture text
        if capture_ft.ft_forwarded:
            _coeff, filepath = capture_ft.ft_get_materialized_value(coordinates)
            with open(filepath, "r", encoding="utf-8") as f:
                capture_text = f.read()
        else:
            capture_text, _status = await capture_ft.ft_async_get(coordinates, trajactory)

        # Read PS1 pattern (broadcast: use only first dims if shape differs)
        ps1_shape = ps1_pattern_ft.ft_capacity_shape
        ps1_coords = coordinates[:len(ps1_shape)]
        # Broadcast scalar or [1] pattern to any coordinate
        if ps1_shape == [1] or ps1_shape == []:
            ps1_coords = [0] if ps1_shape == [1] else []

        if ps1_pattern_ft.ft_forwarded:
            _coeff, filepath = ps1_pattern_ft.ft_get_materialized_value(ps1_coords)
            with open(filepath, "r", encoding="utf-8") as f:
                pattern = f.read().strip()
        else:
            pattern, _status = await ps1_pattern_ft.ft_async_get(ps1_coords, trajactory)
            pattern = pattern.strip()

        # Check if pattern appears in any line of capture
        if pattern and pattern in capture_text:
            return ("true", Status.confidence(1.0))
        return ("false", Status.confidence(1.0))

    result = FutureTensor(
        relative_to,
        ps1_check_get,
        [sympy.Integer(s) for s in shape],
    )
    result.ft_capacity_shape = list(shape)
    return result
