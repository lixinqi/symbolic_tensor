"""
Status = Import[status].Status

FutureTensor[$shape list[int]] :=
    SymbolicTensor[$shape list[int]]
    * $ft_forwarded bool
    * $ft_forward (void <- $prompt_tensor SymbolicTensor[$shape list[int]])
    * $ft_async_get (Awaitable[(str, $confidence Status)] <- $coordinates list[int] <- $prompt)

FutureTensor.ft_forward :=
    void
    <- $self
    <- $prompt_tensor SymbolicTensor[$shape list[int]]
    # inline
    <- { early return if self.ft_forwarded }
    <- { set self.ft_forwarded at the end of this function }
    <- (AsyncList[$coordinates list[int]] <- $self.shape)
    <- (AsyncList[$prompt str] <- $prompt_tensor)
    <- Async[($sole_elem_output, $confidence Status)<- self.ft_async_get <- $coordinates <- $prompt]
    <- (void
        <- { self.data[coordinates] = Status.convert_status_to_float(confidence) }
        <- $self.st_assign
        <- Import[symbolic_tensor make_tensor]
        <- list[$sole_elem_output])
"""

import asyncio
import itertools
import os
from typing import Callable, Awaitable, List, Tuple

import torch

from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor
from experience.future_tensor.status import Status



def _read_element(tensor: torch.Tensor, flat_index: int) -> str:
    """Read the symbolic content at a flat index."""
    digits = list(str(flat_index))
    path = os.path.join(
        tensor.st_relative_to,
        tensor.st_tensor_uid,
        "storage",
        os.path.join(*digits),
        "data",
    )
    path = os.path.realpath(path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _coords_to_flat(coordinates: List[int], shape: List[int]) -> int:
    """Convert multi-dimensional coordinates to flat index."""
    flat = 0
    stride = 1
    for i in reversed(range(len(shape))):
        flat += coordinates[i] * stride
        stride *= shape[i]
    return flat


class FutureTensor:
    """A lazy-materialization symbolic tensor.

    Wraps a SymbolicTensor (torch.Tensor with st_* attributes) and defers
    actual content computation until ft_forward is called with a prompt_tensor.

    Args:
        shape: The tensor shape.
        relative_to: Storage root directory.
        ft_async_get: Async callable (coordinates, prompt) -> (str, Status)
            that produces (content, status) for the given coordinates and prompt.
    """

    def __init__(
        self,
        shape: List[int],
        relative_to: str,
        ft_async_get: Callable[[List[int], str], Awaitable[Tuple[str, "Status"]]],
    ):
        from experience.symbolic_tensor.tensor_util.make_none_tensor import make_none_tensor

        self._tensor: torch.Tensor = make_none_tensor(shape, relative_to)
        self.ft_forwarded: bool = False
        self.ft_async_get = ft_async_get

    @property
    def shape(self) -> List[int]:
        return list(self._tensor.shape)

    @property
    def st_relative_to(self) -> str:
        return self._tensor.st_relative_to

    @property
    def st_tensor_uid(self) -> str:
        return self._tensor.st_tensor_uid

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    def ft_forward(self, prompt_tensor: torch.Tensor) -> None:
        """Materialize this FutureTensor by calling ft_async_get for each element.

        Args:
            prompt_tensor: A SymbolicTensor of the same shape whose elements
                provide the prompt string for each coordinate.
        """
        if self.ft_forwarded:
            return

        shape = self.shape

        # Generate all coordinates from shape
        all_coordinates: List[List[int]] = [
            list(coords)
            for coords in itertools.product(*[range(s) for s in shape])
        ]

        # Read prompts from prompt_tensor for each coordinate
        all_prompts: List[str] = [
            _read_element(prompt_tensor, _coords_to_flat(coords, shape))
            for coords in all_coordinates
        ]

        # Async call ft_async_get for each (coordinates, prompt) pair
        async def _gather():
            tasks = [
                self.ft_async_get(coords, prompt)
                for coords, prompt in zip(all_coordinates, all_prompts)
            ]
            return await asyncio.gather(*tasks)

        results: List[Tuple[str, Status]] = asyncio.run(_gather())

        # Unpack (content, status) pairs; convert Status to float for tensor storage
        sole_elem_output: List[str] = [content for content, _ in results]
        float_values: List[float] = [Status.convert_status_to_float(s) for _, s in results]

        # Set self.data[coordinates] = Status.convert_status_to_float(confidence)
        for coords, fval in zip(all_coordinates, float_values):
            flat_idx = _coords_to_flat(coords, shape)
            self._tensor.data.flatten()[flat_idx] = fval

        # Reshape flat list into nested structure matching shape
        nested_data = _unflatten(sole_elem_output, shape)

        # Make a new symbolic tensor from the results
        result_tensor = make_tensor(nested_data, self.st_relative_to)

        # Assign result back to self (symbolic channel only)
        _assign_symbolic_only(self._tensor, result_tensor)

        self.ft_forwarded = True


def _unflatten(flat_list: List[str], shape: List[int]):
    """Rebuild nested list structure from flat list given shape."""
    if not shape:
        return flat_list[0] if flat_list else None
    if len(shape) == 1:
        return flat_list
    chunk_size = 1
    for s in shape[1:]:
        chunk_size *= s
    return [
        _unflatten(flat_list[i * chunk_size:(i + 1) * chunk_size], shape[1:])
        for i in range(shape[0])
    ]


def _assign_symbolic_only(lvalue: torch.Tensor, rvalue: torch.Tensor) -> None:
    """Copy symbolic storage from rvalue to lvalue without overwriting data coefficients."""
    import shutil

    assert lvalue.shape == rvalue.shape
    shape = list(lvalue.shape)
    for coords in itertools.product(*[range(s) for s in shape]):
        coords = list(coords)
        lvalue_path = _storage_path_for_tensor(lvalue, coords, shape)
        rvalue_path = _storage_path_for_tensor(rvalue, coords, shape)
        os.makedirs(os.path.dirname(lvalue_path), exist_ok=True)
        if os.path.isfile(rvalue_path):
            shutil.copy2(rvalue_path, lvalue_path)


def _storage_path_for_tensor(
    tensor: torch.Tensor, coordinates: List[int], shape: List[int]
) -> str:
    """Get storage file path for a tensor at given coordinates."""
    flat_index = _coords_to_flat(coordinates, shape)
    digits = list(str(flat_index))
    return os.path.join(
        tensor.st_relative_to, tensor.st_tensor_uid,
        "storage", os.path.join(*digits), "data",
    )


if __name__ == "__main__":
    import tempfile

    print("Running tests for future_tensor...\n")

    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    def read_storage(tensor, flat_index):
        digits = list(str(flat_index))
        path = os.path.join(
            tensor.st_relative_to,
            tensor.st_tensor_uid,
            "storage",
            os.path.join(*digits),
            "data",
        )
        with open(path) as f:
            return f.read()

    # ── Test 1: Basic 1D forward ──────────────────────────────────────

    print("Test 1: Basic 1D forward")
    with tempfile.TemporaryDirectory() as tmpdir:
        async def echo_get(coordinates, prompt):
            return (f"output({coordinates}, {prompt})", Status.confidence(0.9))

        ft = FutureTensor([3], tmpdir, echo_get)
        run_test("Shape is [3]", ft.shape == [3])
        run_test("Not forwarded initially", ft.ft_forwarded is False)

        prompt_t = make_tensor(["prompt_0", "prompt_1", "prompt_2"], tmpdir)
        ft.ft_forward(prompt_t)

        run_test("Forwarded after ft_forward", ft.ft_forwarded is True)
        run_test("Element 0", read_storage(ft.tensor, 0) == "output([0], prompt_0)")
        run_test("Element 1", read_storage(ft.tensor, 1) == "output([1], prompt_1)")
        run_test("Element 2", read_storage(ft.tensor, 2) == "output([2], prompt_2)")
        run_test("Confidence 0", abs(ft.tensor.data.flatten()[0].item() - 0.9) < 0.01)
        run_test("Confidence 1", abs(ft.tensor.data.flatten()[1].item() - 0.9) < 0.01)

    # ── Test 2: Idempotent forward (early return) ─────────────────────

    print("Test 2: Idempotent forward")
    with tempfile.TemporaryDirectory() as tmpdir:
        counter = [0]

        async def counting_get(coordinates, prompt):
            counter[0] += 1
            return ("x", Status.confidence(1.0))

        ft = FutureTensor([2], tmpdir, counting_get)
        prompt_t = make_tensor(["a", "b"], tmpdir)

        ft.ft_forward(prompt_t)
        first_count = counter[0]
        run_test("Called 2 times on first forward", first_count == 2, 2, first_count)

        ft.ft_forward(prompt_t)
        run_test("No additional calls on second forward", counter[0] == first_count)

    # ── Test 3: 2D tensor ─────────────────────────────────────────────

    print("Test 3: 2D tensor [2, 3]")
    with tempfile.TemporaryDirectory() as tmpdir:
        async def coord_get(coordinates, prompt):
            return (f"{coordinates}", Status.confidence(0.75))

        ft = FutureTensor([2, 3], tmpdir, coord_get)
        run_test("Shape is [2, 3]", ft.shape == [2, 3])

        prompts = [["p00", "p01", "p02"], ["p10", "p11", "p12"]]
        prompt_t = make_tensor(prompts, tmpdir)
        ft.ft_forward(prompt_t)

        run_test("Element [0,0]", read_storage(ft.tensor, 0) == "[0, 0]")
        run_test("Element [0,2]", read_storage(ft.tensor, 2) == "[0, 2]")
        run_test("Element [1,0]", read_storage(ft.tensor, 3) == "[1, 0]")
        run_test("Element [1,2]", read_storage(ft.tensor, 5) == "[1, 2]")

    # ── Test 4: Async concurrency ─────────────────────────────────────

    print("Test 4: Async concurrency (all tasks run concurrently)")
    with tempfile.TemporaryDirectory() as tmpdir:
        import time

        async def slow_get(coordinates, prompt):
            await asyncio.sleep(0.05)
            return (prompt.upper(), Status.confidence(1.0))

        ft = FutureTensor([4], tmpdir, slow_get)
        prompt_t = make_tensor(["alpha", "beta", "gamma", "delta"], tmpdir)

        start = time.perf_counter()
        ft.ft_forward(prompt_t)
        elapsed = time.perf_counter() - start

        run_test("Concurrent execution (<0.15s for 4x50ms)",
                 elapsed < 0.15, "<0.15s", f"{elapsed:.3f}s")
        run_test("Element 0 uppercased", read_storage(ft.tensor, 0) == "ALPHA")
        run_test("Element 3 uppercased", read_storage(ft.tensor, 3) == "DELTA")

    # ── Test 5: Scalar tensor ─────────────────────────────────────────

    print("Test 5: Scalar tensor [1]")
    with tempfile.TemporaryDirectory() as tmpdir:
        async def scalar_get(coordinates, prompt):
            return (f"scalar:{prompt}", Status.confidence(0.5))

        ft = FutureTensor([1], tmpdir, scalar_get)
        prompt_t = make_tensor(["hello"], tmpdir)
        ft.ft_forward(prompt_t)

        run_test("Single element", read_storage(ft.tensor, 0) == "scalar:hello")
        run_test("Confidence 0.5", abs(ft.tensor.data.flatten()[0].item() - 0.5) < 0.01)

    # ── Test 6: Tensor coefficients after forward ────────────────────

    print("Test 6: Tensor coefficients after forward")
    with tempfile.TemporaryDirectory() as tmpdir:
        async def fill_get(coordinates, prompt):
            return ("filled", Status.confidence(0.8))

        ft = FutureTensor([3], tmpdir, fill_get)
        run_test("All zeros before forward",
                 torch.all(ft.tensor == 0).item())

        prompt_t = make_tensor(["a", "b", "c"], tmpdir)
        ft.ft_forward(prompt_t)

        run_test("Coefficients are confidence (0.8)",
                 all(abs(ft.tensor.data.flatten()[i].item() - 0.8) < 0.01
                     for i in range(3)))

    # ── Test 7: Prompt content correctly passed ───────────────────────

    print("Test 7: Prompt content fidelity")
    with tempfile.TemporaryDirectory() as tmpdir:
        received_prompts = []

        async def capture_get(coordinates, prompt):
            received_prompts.append((coordinates, prompt))
            return ("ok", Status.confidence(1.0))

        ft = FutureTensor([2], tmpdir, capture_get)
        prompt_t = make_tensor(["multi\nline\nprompt", "second prompt"], tmpdir)
        ft.ft_forward(prompt_t)

        run_test("Received 2 calls", len(received_prompts) == 2)
        run_test("Prompt 0 multiline preserved",
                 received_prompts[0] == ([0], "multi\nline\nprompt"),
                 ([0], "multi\nline\nprompt"), received_prompts[0])
        run_test("Prompt 1 correct",
                 received_prompts[1] == ([1], "second prompt"))

    # ── Test 8: st_assign overwrites storage ──────────────────────────

    print("Test 8: Storage overwritten by ft_forward")
    with tempfile.TemporaryDirectory() as tmpdir:
        async def overwrite_get(coordinates, prompt):
            return ("new_content", Status.confidence(0.95))

        ft = FutureTensor([2], tmpdir, overwrite_get)
        initial = make_tensor(["old_0", "old_1"], tmpdir)
        assign_tensor(ft._tensor, initial)
        run_test("Before: old_0", read_storage(ft.tensor, 0) == "old_0")

        prompt_t = make_tensor(["p0", "p1"], tmpdir)
        ft.ft_forward(prompt_t)

        run_test("After: new_content[0]", read_storage(ft.tensor, 0) == "new_content")
        run_test("After: new_content[1]", read_storage(ft.tensor, 1) == "new_content")

    # ── Test 9: _unflatten helper ─────────────────────────────────────

    print("Test 9: _unflatten utility")
    run_test("1D", _unflatten(["a", "b", "c"], [3]) == ["a", "b", "c"])
    run_test("2D [2,3]",
             _unflatten(["a", "b", "c", "d", "e", "f"], [2, 3])
             == [["a", "b", "c"], ["d", "e", "f"]])
    run_test("3D [2,2,2]",
             _unflatten(list("abcdefgh"), [2, 2, 2])
             == [[["a", "b"], ["c", "d"]], [["e", "f"], ["g", "h"]]])
    run_test("Scalar", _unflatten(["x"], []) == "x")

    # ── Test 10: Exception in ft_async_get propagates ─────────────────

    print("Test 10: Exception propagation")
    with tempfile.TemporaryDirectory() as tmpdir:
        async def failing_get(coordinates, prompt):
            if coordinates == [1]:
                raise ValueError("boom")
            return ("ok", Status.confidence(1.0))

        ft = FutureTensor([3], tmpdir, failing_get)
        prompt_t = make_tensor(["a", "b", "c"], tmpdir)

        try:
            ft.ft_forward(prompt_t)
            run_test("Should have raised", False)
        except ValueError as e:
            run_test("ValueError propagated", "boom" in str(e))
            run_test("ft_forwarded still False on error", ft.ft_forwarded is False)

    # ── Test 11: Status variants in tensor storage ─────────────────────

    print("Test 11: Status variants")
    with tempfile.TemporaryDirectory() as tmpdir:
        async def status_get(coordinates, prompt):
            if coordinates == [0]:
                return ("over", Status.confidence(0.9))
            elif coordinates == [1]:
                return ("failed", Status.self_confidence_yet_failed(0.6))
            else:
                return ("normal", Status.confidence(0.7))

        ft = FutureTensor([3], tmpdir, status_get)
        prompt_t = make_tensor(["a", "b", "c"], tmpdir)
        ft.ft_forward(prompt_t)

        run_test("confidence(0.9) -> 0.9",
                 abs(ft.tensor.data.flatten()[0].item() - 0.9) < 0.01)
        run_test("self_confidence_yet_failed(0.6) -> -0.6",
                 abs(ft.tensor.data.flatten()[1].item() - (-0.6)) < 0.01)
        run_test("confidence(0.7) -> 0.7",
                 abs(ft.tensor.data.flatten()[2].item() - 0.7) < 0.01)
        run_test("Content still correct", read_storage(ft.tensor, 0) == "over")

    # ── Test 12: Status enum variants ────────────────────────────────

    print("Test 12: Status enum variants (kConfidenceNotBounded, kContextOverflow)")
    with tempfile.TemporaryDirectory() as tmpdir:
        async def enum_get(coordinates, prompt):
            if coordinates == [0]:
                return ("not_bounded", Status.kConfidenceNotBounded)
            elif coordinates == [1]:
                return ("overflow", Status.kContextOverflow)
            else:
                return ("ok", Status.confidence(0.5))

        ft = FutureTensor([3], tmpdir, enum_get)
        prompt_t = make_tensor(["a", "b", "c"], tmpdir)
        ft.ft_forward(prompt_t)

        run_test("kConfidenceNotBounded -> -2.0",
                 abs(ft.tensor.data.flatten()[0].item() - (-2.0)) < 0.01)
        run_test("kContextOverflow -> -3.0",
                 abs(ft.tensor.data.flatten()[1].item() - (-3.0)) < 0.01)
        run_test("confidence(0.5) -> 0.5",
                 abs(ft.tensor.data.flatten()[2].item() - 0.5) < 0.01)

    # ── Test 13: Per-element varying confidence ────────────────────────

    print("Test 13: Per-element varying confidence")
    with tempfile.TemporaryDirectory() as tmpdir:
        async def varying_get(coordinates, prompt):
            c = coordinates[0]
            return (f"item_{c}", Status.confidence(c * 0.25))

        ft = FutureTensor([5], tmpdir, varying_get)
        prompt_t = make_tensor(["p"] * 5, tmpdir)
        ft.ft_forward(prompt_t)

        for i in range(5):
            expected_conf = i * 0.25
            actual_conf = ft.tensor.data.flatten()[i].item()
            run_test(f"Confidence [{i}] = {expected_conf}",
                     abs(actual_conf - expected_conf) < 0.01,
                     expected_conf, actual_conf)

    print("\nAll tests completed.")
