"""
dispatch_policy context manager + thread-local active policy state.

A module-level default TracePolicy is used when no dispatch_policy block is active,
so bare second_derivative_start.grad.backward() is always safe.

1st-derivative tracing is opt-in only — no module-level default.
Use first_derivative_policy(TracePolicy(records)) to activate.
"""

import threading
from contextlib import contextmanager
from typing import List, Optional

from experience.future_tensor.second_derivative.policy import Policy, PolicyConflictError
from experience.future_tensor.second_derivative.trace_policy import TracePolicy

# Thread-local storage for the active policy
_local = threading.local()

# Module-level default collector; used when no dispatch_policy block is active
_default_collector: List = []
_default_policy: TracePolicy = TracePolicy(_default_collector)


# ── 2nd-derivative policy (existing) ──

def get_active_policy() -> Policy:
    """Return the currently active policy, falling back to the module default."""
    return getattr(_local, "policy", _default_policy)


@contextmanager
def dispatch_policy(policy: Policy):
    """Set the active 2nd-derivative dispatch policy for the duration of the block.

    Args:
        policy: A Policy instance (e.g. TracePolicy(collector)).

    Raises:
        PolicyConflictError: if a dispatch_policy block is already active on this thread.
    """
    if getattr(_local, "policy", None) is not None:
        raise PolicyConflictError(
            "A dispatch_policy block is already active on this thread. "
            "Nesting is not supported."
        )
    _local.policy = policy
    try:
        yield policy
    finally:
        del _local.policy


# ── 1st-derivative policy (new, opt-in only) ──

def get_active_first_policy() -> Optional[Policy]:
    """Return the currently active 1st-derivative policy, or None if not active."""
    return getattr(_local, "first_policy", None)


@contextmanager
def first_derivative_policy(policy: Policy):
    """Set the active 1st-derivative dispatch policy for the duration of the block.

    1st-derivative tracing is opt-in only — when no first_derivative_policy block
    is active, get_active_first_policy() returns None and dispatch is a no-op
    (zero overhead).

    Args:
        policy: A Policy instance (e.g. TracePolicy(collector)).

    Raises:
        PolicyConflictError: if a first_derivative_policy block is already active.
    """
    if getattr(_local, "first_policy", None) is not None:
        raise PolicyConflictError(
            "A first_derivative_policy block is already active on this thread. "
            "Nesting is not supported."
        )
    _local.first_policy = policy
    try:
        yield policy
    finally:
        del _local.first_policy
