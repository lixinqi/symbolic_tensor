"""
FtTmuxSendCtrl := torch.autograd.Function[
    $forward  Import[{future_tensor function tmux_send_ctrl_forward.viba}],
    $backward Import[{future_tensor function tmux_send_ctrl_backward.viba}]
]

ft_tmux_send_ctrl = FtTmuxSendCtrl.apply
"""

from typing import Callable, List

import torch

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.function.tmux_send_ctrl_forward import tmux_send_ctrl_forward
from experience.future_tensor.function.tmux_send_ctrl_backward import tmux_send_ctrl_backward


class FtTmuxSendCtrl(torch.autograd.Function):
    """Autograd Function for sending control characters to tmux sessions."""

    @staticmethod
    def forward(ctx, input_ft: FutureTensor, get_ctrl: Callable[[List[int]], str]):
        ctx.input_ft = input_ft
        ctx.shape = input_ft.ft_capacity_shape
        ctx.relative_to = input_ft.ft_static_tensor.st_relative_to
        return tmux_send_ctrl_forward(input_ft, get_ctrl)

    @staticmethod
    def backward(ctx, grad_output):
        return tmux_send_ctrl_backward(ctx, grad_output), None


def ft_tmux_send_ctrl(
    input_ft: FutureTensor,
    get_ctrl: Callable[[List[int]], str],
) -> FutureTensor:
    """Send control characters to tmux sessions determined by ``input_ft`` elements.

    For each element, reads the instance_id from the input, looks up the
    corresponding tmux session, and sends ``get_ctrl(coordinates)`` to its
    active pane with ``literal=False, enter=False``.

    Args:
        input_ft: FutureTensor whose elements contain instance IDs.
        get_ctrl: Callable that receives coordinates and returns the control
            character sequence to send for that element.

    Returns:
        A FutureTensor with the same shape as ``input_ft``.
    """
    return FtTmuxSendCtrl.apply(input_ft, get_ctrl)


if __name__ == "__main__":
    import os
    import sys
    import tempfile

    import sympy

    from experience.future_tensor.status import Status
    from experience.future_tensor.function.tmux_session import tmux_session_prefix
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor as st_make_tensor
    from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor

    print("Running tests for ft_tmux_send_ctrl...\n")

    passed = 0
    failed = 0

    def run_test(name: str, condition: bool, expected=None, actual=None):
        global passed, failed
        if condition:
            passed += 1
            print(f"  ✓ {name}")
        else:
            failed += 1
            print(f"  ✗ {name}")
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
        ft = make_forwarded_ft([2], ["ctrl0", "ctrl1"], tmpdir)
        output = ft_tmux_send_ctrl(ft, lambda coords: "c")
        run_test("output shape matches input", output.ft_capacity_shape == [2])
        run_test("output is lazy", output.ft_forwarded is False)

    # === Group 2: Send ctrl to existing sessions ===
    print("\nGroup 2: Send ctrl to existing sessions")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([2], ["ca", "cb"], tmpdir)

        for sid in ["ca", "cb"]:
            name = f"{tmux_session_prefix}{sid}"
            if not server.has_session(name):
                server.new_session(session_name=name, kill_session=True, attach=False)
            test_sessions.append(name)

        sent_ctrls = []

        def capture_ctrl(coords):
            ctrl = f"C-{coords[0]}"
            sent_ctrls.append(ctrl)
            return ctrl

        output = ft_tmux_send_ctrl(ft, capture_ctrl)
        prompt_t = st_make_tensor(["p", "p"], tmpdir)
        output.ft_forward(prompt_t)

        run_test("output forwarded", output.ft_forwarded is True)
        run_test("all ctrls sent", len(sent_ctrls) == 2)
        run_test("ctrl 0 correct", sent_ctrls[0] == "C-0")
        run_test("ctrl 1 correct", sent_ctrls[1] == "C-1")
        run_test("status is confidence",
                 output.ft_static_tensor.data[0].item() == 1.0)

    cleanup()

    # === Group 3: Error when session does not exist ===
    print("\nGroup 3: Error when session does not exist")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([1], ["noexist_ctrl"], tmpdir)
        output = ft_tmux_send_ctrl(ft, lambda coords: "c")
        prompt_t = st_make_tensor(["p"], tmpdir)
        output.ft_forward(prompt_t)
        run_test("status is scbf when session missing",
                 output.ft_static_tensor.data[0].item() < 0)

    # === Group 4: Autograd connectivity ===
    print("\nGroup 4: Autograd connectivity")

    with tempfile.TemporaryDirectory() as tmpdir:
        ft = make_forwarded_ft([2], ["g0", "g1"], tmpdir)
        ft.requires_grad_(True)
        output = ft_tmux_send_ctrl(ft, lambda coords: "c")
        run_test("output has grad_fn", output.grad_fn is not None)
        run_test("output requires_grad", output.requires_grad is True)

    cleanup()

    print(f"\n  Passed: {passed}, Failed: {failed}, Total: {passed + failed}")
    if failed == 0:
        print("All ft_tmux_send_ctrl tests passed.")
    else:
        sys.exit(1)
