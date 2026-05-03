"""
Policy base class, ReflectionRecord, PolicyConflictError.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict

import torch


@dataclass
class ReflectionRecord:
    fn: Callable                      # the 1st-derivative backward function object
    inputs: Dict[str, Any]            # arg_name -> tensor/value passed to dispatch
    output: torch.Tensor              # placeholder scalar (value=1)
    timestamp: float = field(default_factory=time.monotonic)


class PolicyConflictError(RuntimeError):
    pass


class Policy:
    """Base class for 2nd-derivative dispatch policies."""

    def dispatch(self, fn: Callable, arg_name2inputs: Dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError
