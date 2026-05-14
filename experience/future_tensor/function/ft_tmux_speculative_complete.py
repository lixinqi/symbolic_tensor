"""
ft_tmux_speculative_complete: Speculative multi-action dispatch from a single LLM call.

Expert generates multiple actions (separated by sep) in one LLM request.
This op consumes them one at a time, checking screen_capture to confirm
the previous action was applied before returning the next one.

State (bunch text + position) persists across recurrent iterations,
captured in autograd context for backward.

No autograd — pure runtime filter, transparent to backward.
Like ft_first_line but stateful: returns action[0] on first call,
then action[1] when screen confirms action[0] was typed, etc.
When all actions consumed or screen doesn't match, returns empty
(which forces upstream ft_expert to re-generate).
"""

from typing import Dict, List, Optional, Tuple

import sympy

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status


def _parse_bunch(text: str, sep: str = "\n\n") -> List[str]:
    """Split multi-action text into individual actions."""
    if not text or not text.strip():
        return []
    parts = [p.strip() for p in text.split(sep) if p.strip()]
    return parts


def _screen_confirms_action(captured_text: str, action: str) -> bool:
    """Check if the terminal screen shows that the given action was applied.

    For text actions: the action string should appear on the last prompt line.
    For ctrl actions (like Enter): a new prompt line should have appeared.

    Uses suffix matching: checks if any non-empty line in the captured text
    ends with or contains the action text.
    """
    if not captured_text or not action:
        return False

    action_clean = action.strip()
    lines = [l for l in captured_text.split("\n") if l.strip()]
    if not lines:
        return False

    # For control keys, check if a fresh prompt appeared (new prompt after command)
    ctrl_keys = {"Enter", "Escape", "Tab", "C-c", "C-d"}
    if action_clean in ctrl_keys:
        # After pressing Enter, we expect command output + fresh prompt
        # Just check that there's a prompt-like line
        last = lines[-1]
        if "λ" in last or "$ " in last or last.rstrip().endswith("$"):
            return True
        return False

    # For text actions: check if the action text appears on the last prompt line
    last = lines[-1]
    if action_clean in last:
        return True

    return False


def ft_tmux_speculative_complete(
    bunch_input: FutureTensor,
    screen_capture: FutureTensor,
    sep: str = "\n\n",
) -> FutureTensor:
    """Speculative multi-action dispatch from a single LLM call.

    On each recurrent iteration:
      1. If no cached bunch or bunch exhausted: pull from bunch_input,
         parse into actions, return first action.
      2. If cached bunch exists and screen confirms previous action was
         applied: advance position, return next action.
      3. If screen doesn't confirm: invalidate cache, return empty
         (forces upstream re-generation).

    Args:
        bunch_input: FutureTensor whose elements contain multi-action text
            (actions separated by sep). From ft_expert output.
        screen_capture: FutureTensor whose elements contain current terminal
            screen text. From ft_tmux_capture_pane.
        sep: Separator between actions in bunch_input. Default: "\\n\\n".

    Returns:
        A lazy FutureTensor with the same shape. Each element contains the
        current speculative action, or empty string if cache miss.
    """
    shape = bunch_input.ft_capacity_shape
    relative_to = bunch_input.ft_static_tensor.st_relative_to

    # Persistent state: keyed by prefix coordinates (all dims except last)
    # Each entry: {"bunch": List[str], "position": int, "last_action": str}
    _state: Dict[tuple, dict] = {}

    async def speculative_get(coordinates: List[int], trajactory: str):
        # Use all-but-last coordinate as the prefix key (like ft_recurrent)
        # But if shape has only 2 dims [1, N], prefix is just (0,)
        prefix_key = tuple(coordinates[:-1]) if len(coordinates) > 1 else ()
        iter_idx = coordinates[-1] if len(coordinates) > 0 else 0

        state = _state.get(prefix_key)

        # Read screen capture for confirmation check
        screen_text = ""
        if screen_capture.ft_forwarded:
            _coeff, filepath = screen_capture.ft_get_materialized_value(coordinates)
            with open(filepath, "r", encoding="utf-8") as f:
                screen_text = f.read()
        else:
            screen_text, _status = await screen_capture.ft_async_get(
                coordinates, trajactory
            )

        # Case 1: No cached state or first iteration — pull fresh bunch
        if state is None or iter_idx == 0:
            if bunch_input.ft_forwarded:
                _coeff, filepath = bunch_input.ft_get_materialized_value(coordinates)
                with open(filepath, "r", encoding="utf-8") as f:
                    bunch_text = f.read()
            else:
                bunch_text, _status = await bunch_input.ft_async_get(
                    coordinates, trajactory
                )

            actions = _parse_bunch(bunch_text, sep)
            if not actions:
                return ("", Status.self_confidence_but_failed(0.1))

            # Cache the bunch and return first action
            _state[prefix_key] = {
                "bunch": actions,
                "position": 0,
                "last_action": actions[0],
            }
            return (actions[0], Status.confidence(1.0))

        # Case 2: Have cached bunch — check if screen confirms last action
        bunch = state["bunch"]
        pos = state["position"]
        last_action = state["last_action"]

        if _screen_confirms_action(screen_text, last_action):
            # Previous action confirmed — advance to next
            next_pos = pos + 1
            if next_pos < len(bunch):
                next_action = bunch[next_pos]
                _state[prefix_key] = {
                    "bunch": bunch,
                    "position": next_pos,
                    "last_action": next_action,
                }
                return (next_action, Status.confidence(1.0))
            else:
                # Bunch exhausted — need fresh generation
                # Pull new bunch from upstream
                if bunch_input.ft_forwarded:
                    _coeff, filepath = bunch_input.ft_get_materialized_value(coordinates)
                    with open(filepath, "r", encoding="utf-8") as f:
                        bunch_text = f.read()
                else:
                    bunch_text, _status = await bunch_input.ft_async_get(
                        coordinates, trajactory
                    )

                actions = _parse_bunch(bunch_text, sep)
                if not actions:
                    _state.pop(prefix_key, None)
                    return ("", Status.self_confidence_but_failed(0.1))

                _state[prefix_key] = {
                    "bunch": actions,
                    "position": 0,
                    "last_action": actions[0],
                }
                return (actions[0], Status.confidence(1.0))
        else:
            # Screen doesn't confirm — cache miss, need fresh generation
            if bunch_input.ft_forwarded:
                _coeff, filepath = bunch_input.ft_get_materialized_value(coordinates)
                with open(filepath, "r", encoding="utf-8") as f:
                    bunch_text = f.read()
            else:
                bunch_text, _status = await bunch_input.ft_async_get(
                    coordinates, trajactory
                )

            actions = _parse_bunch(bunch_text, sep)
            if not actions:
                _state.pop(prefix_key, None)
                return ("", Status.self_confidence_but_failed(0.1))

            _state[prefix_key] = {
                "bunch": actions,
                "position": 0,
                "last_action": actions[0],
            }
            return (actions[0], Status.confidence(1.0))

    result = FutureTensor(
        relative_to,
        speculative_get,
        [sympy.Integer(s) for s in shape],
    )
    result.ft_capacity_shape = list(shape)

    # Expose state for autograd context capture
    result._speculative_state = _state

    return result
