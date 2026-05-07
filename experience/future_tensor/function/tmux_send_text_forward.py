"""
tmux_send_text_forward :=
    FutureTensor
    <- $input FutureTensor
    <- $get_text Callable[[list[int]], str]
    # inline

Send pure text to a tmux pane via libtmux with literal=True, enter=False.
"""

from typing import Callable, List

import libtmux
import sympy

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status
from experience.future_tensor.function.tmux_session import tmux_session_prefix


def tmux_send_text_forward(
    input_ft: FutureTensor,
    get_text: Callable[[List[int]], str],
) -> FutureTensor:
    """Forward: create a lazy FutureTensor whose ft_async_get sends text to tmux."""
    shape = input_ft.ft_capacity_shape
    relative_to = input_ft.ft_static_tensor.st_relative_to

    async def send_async_get(coordinates: List[int], prompt: str):
        if input_ft.ft_forwarded:
            _coeff, filepath = input_ft.ft_get_materialized_value(coordinates)
            with open(filepath, "r", encoding="utf-8") as f:
                instance_id = f.read().strip()
        else:
            content, _status = await input_ft.ft_async_get(coordinates, prompt)
            instance_id = content.strip()

        text = get_text(coordinates)
        session_name = f"{tmux_session_prefix}{instance_id}"

        try:
            server = libtmux.Server()
            session = None
            for s in server.sessions:
                if s.session_name == session_name:
                    session = s
                    break
            if session is None:
                return ("", Status.self_confidence_but_failed(0.0))
            pane = session.active_window.active_pane
            pane.send_keys(text, literal=True, enter=False)
            return ("", Status.confidence(1.0))
        except Exception:
            return ("", Status.self_confidence_but_failed(0.0))

    result = FutureTensor(
        relative_to,
        send_async_get,
        [sympy.Integer(s) for s in shape],
    )
    result.ft_capacity_shape = list(shape)
    return result
