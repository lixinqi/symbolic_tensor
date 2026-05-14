"""
dispatch_policy context manager + thread-local active policy state.

Tracing is opt-in only. Use dispatch_policy(TracePolicy(records)) to
trace either 1st or 2nd derivative backward passes.
"""

import threading
from contextlib import contextmanager
from typing import Optional

from experience.future_tensor.backward_dispatch.policy import Policy, PolicyConflictError

# Thread-local storage for the active policy
_local = threading.local()


def get_active_policy() -> Optional[Policy]:
    """Return the currently active policy, or None if not active."""
    return getattr(_local, "policy", None)


@contextmanager
def dispatch_policy(policy: Policy):
    """Set the active dispatch policy for the duration of the block.

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
