"""
TracePolicy: non-destructive collector (default policy).

Records every dispatch call into a caller-supplied list without invoking any LLM.
"""

from typing import Any, Callable, Dict, List

import torch

from experience.future_tensor.second_derivative.policy import Policy, ReflectionRecord


class TracePolicy(Policy):
    """Records every 2nd-derivative op call into `collector` without running any LLM."""

    def __init__(self, collector: List[ReflectionRecord]):
        self._collector = collector

    def dispatch(self, fn: Callable, arg_name2inputs: Dict[str, Any]) -> torch.Tensor:
        placeholder = torch.ones(())
        record = ReflectionRecord(fn=fn, inputs=arg_name2inputs, output=placeholder)
        self._collector.append(record)
        return placeholder
