import os
import itertools
import pathlib
import torch
from typing import Any, List, Union
NestedList = Union[pathlib.Path, List["NestedList"]]
def _get_storage_path(tensor: torch.Tensor, coordinates: List[int]) -> pathlib.Path:
    """Compute the flat index from coordinates and strides, return the storage file path."""
    flat_index = sum(c * s for c, s in zip(coordinates, tensor.stride()))
    digits = list(str(flat_index))
    return pathlib.Path(
        tensor.st_relative_to,
        tensor.st_tensor_uid,
        "storage",
        os.path.join(*digits),
        "data",
    )
def _build_nested(
    tensor: torch.Tensor,
    coordinates_list: List[List[int]],
    shape: List[int],
) -> NestedList:
    """Reshape a flat list of paths into a nested list matching the tensor shape."""
    paths = [_get_storage_path(tensor, coords) for coords in coordinates_list]
    if not shape:
        return paths[0]
    if len(shape) == 1:
        return paths
    chunk_size = 1
    for s in shape[1:]:
        chunk_size *= s
    return [
        _build_nested(
            tensor,
            coordinates_list[i * chunk_size:(i + 1) * chunk_size],
            shape[1:],
        )
        for i in range(shape[0])
    ]
def get_nested_list_file_pathes(tensor: torch.Tensor) -> NestedList:
    """Return a nested list of pathlib.Path matching the tensor's shape.
    Each path points to the storage file for the corresponding tensor element.
    Args:
        tensor: A symbolic tensor with st_relative_to and st_tensor_uid attributes.
    Returns:
        A nested list of pathlib.Path matching tensor.shape.
    """
    shape = list(tensor.size())
    coordinates_list = [list(c) for c in itertools.product(*[range(s) for s in shape])]
    return _build_nested(tensor, coordinates_list, shape)
