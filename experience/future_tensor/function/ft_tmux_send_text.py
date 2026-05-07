"""
FtTmuxSendText := torch.autograd.Function[
    $forward  Import[{future_tensor function tmux_send_text_forward.viba}],
    $backward Import[{future_tensor function tmux_send_text_backward.viba}]
]

ft_tmux_send_text = FtTmuxSendText.apply
"""

from typing import Callable, List

import torch

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.function.tmux_send_text_forward import tmux_send_text_forward
from experience.future_tensor.function.tmux_send_text_backward import tmux_send_text_backward


class FtTmuxSendText(torch.autograd.Function):
    """Autograd Function for sending text to tmux sessions."""

    @staticmethod
    def forward(ctx, input_ft: FutureTensor, get_text: Callable[[List[int]], str]):
        ctx.input_ft = input_ft
        ctx.shape = input_ft.ft_capacity_shape
        ctx.relative_to = input_ft.ft_static_tensor.st_relative_to
        return tmux_send_text_forward(input_ft, get_text)

    @staticmethod
    def backward(ctx, grad_output):
        return tmux_send_text_backward(ctx, grad_output), None


def ft_tmux_send_text(
    input_ft: FutureTensor,
    get_text: Callable[[List[int]], str],
) -> FutureTensor:
    """Send text to tmux sessions determined by ``input_ft`` elements.

    For each element, reads the instance_id from the input, looks up the
    corresponding tmux session, and sends ``get_text(coordinates)`` to its
    active pane with ``literal=True, enter=False``.

    Args:
        input_ft: FutureTensor whose elements contain instance IDs.
        get_text: Callable that receives coordinates and returns the text
            to send for that element.

    Returns:
        A FutureTensor with the same shape as ``input_ft``.
    """
    return FtTmuxSendText.apply(input_ft, get_text)


if __name__ == "__main__":
    import os
    import sys
    import tempfile

    import sympy

    from experience.future_tensor.status import Status
    from experience.future_tensor.function.tmux_session import tmux_session_prefix
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor as st_make_tensor
    from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor

    print("Running tests for ft_tmux_send_text...\n")

    passed = 0
    failed = 0

    def run_test(name: str, condition: bool, expected=None, actual=None):
        global passed, failed
        if condition:
            passed += 1
            print(f"  \u2713 {name}")
        else:
            failed += 1
            print(f"  \u2717 {name}")
            if expected is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    def make_forwarded_ft(shape, data_list, tmpdir):
        async def dummy_get(coords, prompt):
            return ("unused", Status.confidence(1.0))

        ft = FutureTensor(tmpdir, dummy_get, [sympy.Integer(s) for s in shape])
        nested = _unflatten_data(data_list, shape)
        result_tensor = st_make_tensor(nested, tmpdir)
        assign_tensor(ft.ft_static_tensor, result_tensor)
        ft.ft_forwarded = True
        return ft

    def _unflatten_data(flat_list, shape):
        if not shape:
            return flat_list[0] if flat_list else None
        if len(shape) == 1:
            return flat_list
        chunk_size = 1
        for s in shape[1:]:
            chunk_size *= s
        return [
            _unflatten_data(flat_list[i * chunk_size : (i + 1) * chunk_size], shape[1:])
            for i in range(shape[0])
        ]

    import libtmux
    server = libtmux.Server()
    test_sessions = []

    def cleanup():
        for name in test_sessions:
            try:
                server.kill_session(name)
            except Exception:
                pass

    # === Group 1: Forward shape and laziness ===
    print("Group 1: Forward shape and laziness")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([2], ["send0", "send1"], tmpdir)
        output = ft_tmux_send_text(ft, lambda coords: f"hello_{coords[0]}")
        run_test("output shape matches input", output.ft_capacity_shape == [2])
        run_test("output is lazy", output.ft_forwarded is False)

    # === Group 2: Send text to existing sessions ===
    print("\nGroup 2: Send text to existing sessions")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([2], ["txt_a", "txt_b"], tmpdir)

        # Pre-create sessions
        for sid in ["txt_a", "txt_b"]:
            name = f"{tmux_session_prefix}{sid}"
            if not server.has_session(name):
                server.new_session(session_name=name, kill_session=True, attach=False)
            test_sessions.append(name)

        sent_texts = []

        def capture_text(coords):
            text = f"test_text_{coords[0]}"
            sent_texts.append(text)
            return text

        output = ft_tmux_send_text(ft, capture_text)
        prompt_t = st_make_tensor(["p", "p"], tmpdir)
        output.ft_forward(prompt_t)

        run_test("output forwarded", output.ft_forwarded is True)
        run_test("all texts sent", len(sent_texts) == 2)
        run_test("text 0 correct", sent_texts[0] == "test_text_0")
        run_test("text 1 correct", sent_texts[1] == "test_text_1")
        run_test("status is confidence",
                 output.ft_static_tensor.data[0].item() == 1.0)

    cleanup()

    # === Group 3: Error when session does not exist ===
    print("\nGroup 3: Error when session does not exist")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([1], ["noexist"], tmpdir)
        output = ft_tmux_send_text(ft, lambda coords: "x")
        prompt_t = st_make_tensor(["p"], tmpdir)
        output.ft_forward(prompt_t)
        run_test("status is scbf when session missing",
                 output.ft_static_tensor.data[0].item() < 0)

    # === Group 4: Autograd connectivity ===
    print("\nGroup 4: Autograd connectivity")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([2], ["g0", "g1"], tmpdir)
        ft.requires_grad_(True)
        output = ft_tmux_send_text(ft, lambda coords: "x")
        run_test("output has grad_fn", output.grad_fn is not None)
        run_test("output requires_grad", output.requires_grad is True)

    cleanup()

    print(f"\n  Passed: {passed}, Failed: {failed}, Total: {passed + failed}")
    if failed == 0:
        print("All ft_tmux_send_text tests passed.")
    else:
        sys.exit(1)
