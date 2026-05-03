"""
Status = Import[status].Status

FutureTensor :=
    torch.Tensor[(), bfloat16, value=1]
    * $ft_static_tensor SymbolicTensor[...]
    * $ft_incremental_concated_tensors list[($tensor SymbolicTensor[...], $concat_axis int)]
    * $ft_shape_schema list[sympy.Symbol]
    * $ft_capacity_shape list[int]
    * $ft_forwarded bool
    * $ft_forward (void <- $prompt FutureTensor)
    * $ft_async_get (Awaitable[($output str, $status Status)] <- $coordinates list[int] <- $prompt str)
    * $ft_get_materialized_value (($coefficient float, $element_file_path str) <- $coordinates list[int])
    * $ft_reset_materialized_value (void <- $coordinates list[int] <- $coefficient float <- $filepath str <- $symlink bool)
"""

import asyncio
import itertools
import os
import shutil
from typing import Callable, Awaitable, List, Tuple

import sympy
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
    assert lvalue.shape == rvalue.shape
    shape = list(lvalue.shape)
    for coords in itertools.product(*[range(s) for s in shape]):
        coords = list(coords)
        lvalue_path = _storage_path_for_tensor(lvalue, coords, shape)
        rvalue_path = _storage_path_for_tensor(rvalue, coords, shape)
        os.makedirs(os.path.dirname(lvalue_path), exist_ok=True)
        if os.path.isfile(rvalue_path):
            shutil.copy2(rvalue_path, lvalue_path)


def FutureTensor(
    relative_to: str,
    ft_async_get: Callable[[List[int], str], Awaitable[Tuple[str, "Status"]]],
    ft_shape_schema: List["sympy.Symbol"],
) -> torch.Tensor:
    """Create a FutureTensor: a scalar bfloat16 torch.Tensor monkey-patched with ft_* attributes.

    FutureTensor is an interface — a plain torch.Tensor of shape (), dtype bfloat16, value
    always 1. It is monkey-patched with ft_* attributes. Not a subclass of SymbolicTensor.

    Status / coefficient semantics live inside ft_static_tensor and
    ft_incremental_concated_tensors, not on the scalar itself.

    Args:
        relative_to: Storage root directory.
        ft_async_get: Async callable (coordinates, prompt) -> (str, Status).
        ft_shape_schema: Declared logical shape schema (list of sympy.Symbol or sympy.Integer).
            Determines the shape of ft_static_tensor.

    Returns:
        A scalar torch.Tensor (shape=(), dtype=bfloat16, value=1) with ft_* attributes.
    """
    from experience.symbolic_tensor.tensor_util.make_none_tensor import make_none_tensor

    # Resolve concrete shape from schema (sympy.Integer -> int, sympy.Symbol -> error)
    concrete_shape: List[int] = []
    for dim in ft_shape_schema:
        if isinstance(dim, (int, sympy.Integer)):
            concrete_shape.append(int(dim))
        elif isinstance(dim, sympy.Symbol):
            # Symbolic dim: use 0 for now (dynamic case — zero-sized)
            concrete_shape.append(0)
        else:
            concrete_shape.append(int(dim))

    # Create the scalar reference tensor (shape=(), bfloat16, value=1)
    ft = torch.ones((), dtype=torch.bfloat16)

    # Create ft_static_tensor: always a valid SymbolicTensor
    ft_static = make_none_tensor(concrete_shape, relative_to)

    # Monkey-patch ft_* attributes
    ft.ft_static_tensor = ft_static
    ft.ft_incremental_concated_tensors = []  # list[(SymbolicTensor, concat_axis)]
    ft.ft_shape_schema = list(ft_shape_schema)
    ft.ft_capacity_shape = list(concrete_shape)  # must match logical view tensor shape
    ft.ft_forwarded = False
    ft.ft_async_get = ft_async_get

    def ft_forward(prompt_tensor: torch.Tensor) -> None:
        """Materialize this FutureTensor by calling ft_async_get for each element."""
        if ft.ft_forwarded:
            return

        shape = ft.ft_capacity_shape

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
                ft.ft_async_get(coords, prompt)
                for coords, prompt in zip(all_coordinates, all_prompts)
            ]
            return await asyncio.gather(*tasks)

        results: List[Tuple[str, Status]] = asyncio.run(_gather())

        # Unpack (content, status) pairs; Status -> float stored in ft_static_tensor
        sole_elem_output: List[str] = [content for content, _ in results]
        float_values: List[float] = [Status.convert_status_to_float(s) for _, s in results]

        # Write status floats into ft_static_tensor coefficients
        for coords, fval in zip(all_coordinates, float_values):
            flat_idx = _coords_to_flat(coords, shape)
            ft.ft_static_tensor.data.flatten()[flat_idx] = fval

        # Make a new symbolic tensor from the results and assign its storage
        nested_data = _unflatten(sole_elem_output, shape)
        result_tensor = make_tensor(nested_data, relative_to)
        _assign_symbolic_only(ft.ft_static_tensor, result_tensor)

        ft.ft_forwarded = True

    ft.ft_forward = ft_forward

    def ft_get_materialized_value(coordinates: List[int]) -> Tuple[float, str]:
        """Look up element in logical view tensor by coordinates.

        Returns (coefficient, element_file_path) — the bfloat16 numeric value
        and the path to the symbolic content file on disk.
        """
        shape = ft.ft_capacity_shape
        flat_idx = _coords_to_flat(coordinates, shape)
        coefficient = ft.ft_static_tensor.data.flatten()[flat_idx].item()
        element_file_path = _storage_path_for_tensor(ft.ft_static_tensor, coordinates, shape)
        return (coefficient, element_file_path)

    ft.ft_get_materialized_value = ft_get_materialized_value

    def ft_reset_materialized_value(
        coordinates: List[int],
        coefficient: float,
        filepath: str,
        symlink: bool = False,
    ) -> None:
        """Overwrite element at coordinates with given coefficient and filepath.

        If symlink=False, copies the file content.
        If symlink=True, creates a symlink instead.
        """
        shape = ft.ft_capacity_shape
        flat_idx = _coords_to_flat(coordinates, shape)
        # Update coefficient in ft_static_tensor
        ft.ft_static_tensor.data.flatten()[flat_idx] = coefficient
        dst_path = _storage_path_for_tensor(ft.ft_static_tensor, coordinates, shape)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        if symlink:
            if os.path.exists(dst_path) or os.path.islink(dst_path):
                os.remove(dst_path)
            os.symlink(os.path.realpath(filepath), dst_path)
        else:
            shutil.copy2(filepath, dst_path)

    ft.ft_reset_materialized_value = ft_reset_materialized_value

    return ft


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

    # ── Test 1: Basic structure ──────────────────────────────────────────

    print("Test 1: FutureTensor is a scalar bfloat16 torch.Tensor")
    with tempfile.TemporaryDirectory() as tmpdir:
        async def echo_get(coordinates, prompt):
            return (f"output({coordinates}, {prompt})", Status.confidence(0.9))

        ft = FutureTensor(tmpdir, echo_get, [sympy.Integer(3)])
        run_test("FT is torch.Tensor", isinstance(ft, torch.Tensor))
        run_test("FT shape is ()", list(ft.shape) == [])
        run_test("FT dtype is bfloat16", ft.dtype == torch.bfloat16)
        run_test("FT value is 1", ft.item() == 1.0)
        run_test("ft_capacity_shape is [3]", ft.ft_capacity_shape == [3])
        run_test("ft_static_tensor shape is [3]", list(ft.ft_static_tensor.shape) == [3])
        run_test("Not forwarded initially", ft.ft_forwarded is False)

        prompt_t = make_tensor(["prompt_0", "prompt_1", "prompt_2"], tmpdir)
        ft.ft_forward(prompt_t)

        run_test("Forwarded after ft_forward", ft.ft_forwarded is True)
        run_test("Element 0", read_storage(ft.ft_static_tensor, 0) == "output([0], prompt_0)")
        run_test("Element 1", read_storage(ft.ft_static_tensor, 1) == "output([1], prompt_1)")
        run_test("Element 2", read_storage(ft.ft_static_tensor, 2) == "output([2], prompt_2)")
        run_test("FT value still 1 after forward", ft.item() == 1.0)
        run_test("Status in ft_static_tensor coeff[0]",
                 abs(ft.ft_static_tensor.data.flatten()[0].item() - 0.9) < 0.01)

    # ── Test 2: ft_get_materialized_value / ft_reset_materialized_value ──

    print("Test 2: ft_get_materialized_value / ft_reset_materialized_value")
    with tempfile.TemporaryDirectory() as tmpdir:
        async def simple_get(coordinates, prompt):
            return (f"content_{coordinates[0]}", Status.confidence(0.8))

        ft = FutureTensor(tmpdir, simple_get, [sympy.Integer(2)])
        prompt_t = make_tensor(["a", "b"], tmpdir)
        ft.ft_forward(prompt_t)

        coeff, path = ft.ft_get_materialized_value([0])
        run_test("ft_get_materialized_value coeff", abs(coeff - 0.8) < 0.01)
        run_test("ft_get_materialized_value path exists", os.path.isfile(path))

        # Reset element 0
        new_content_path = os.path.join(tmpdir, "new_content.txt")
        with open(new_content_path, "w") as f:
            f.write("new_content")
        ft.ft_reset_materialized_value([0], 0.5, new_content_path, symlink=False)
        coeff2, path2 = ft.ft_get_materialized_value([0])
        run_test("ft_reset_materialized_value coeff", abs(coeff2 - 0.5) < 0.01)
        with open(path2) as f:
            run_test("ft_reset_materialized_value content", f.read() == "new_content")

    # ── Test 3: Idempotent forward ─────────────────────────────────────

    print("Test 3: Idempotent forward")
    with tempfile.TemporaryDirectory() as tmpdir:
        counter = [0]

        async def counting_get(coordinates, prompt):
            counter[0] += 1
            return ("x", Status.confidence(1.0))

        ft = FutureTensor(tmpdir, counting_get, [sympy.Integer(2)])
        prompt_t = make_tensor(["a", "b"], tmpdir)

        ft.ft_forward(prompt_t)
        first_count = counter[0]
        run_test("Called 2 times on first forward", first_count == 2, 2, first_count)

        ft.ft_forward(prompt_t)
        run_test("No additional calls on second forward", counter[0] == first_count)

    # ── Test 4: ft_incremental_concated_tensors ──────────────────────────

    print("Test 4: ft_incremental_concated_tensors empty by default")
    with tempfile.TemporaryDirectory() as tmpdir:
        ft = FutureTensor(tmpdir, None, [sympy.Integer(3)])
        run_test("ft_incremental_concated_tensors is []", ft.ft_incremental_concated_tensors == [])

    print("\nAll tests completed.")
