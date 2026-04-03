import torch
from typing import Any, List, Tuple
def _is_leaf(obj: Any) -> bool:
    """Check if obj is a leaf (list[torch.Tensor]) rather than a nested list of lists."""
    if not isinstance(obj, list):
        return True
    if len(obj) == 0:
        return True
    return isinstance(obj[0], torch.Tensor)
Pair = Tuple[List[torch.Tensor], Any]
def _collect(
    obj: Any,
    coords: List[int],
    pairs: List[Pair],
) -> None:
    """Recursively collect (coordinate_key, leaf_value) pairs."""
    if _is_leaf(obj):
        key = [torch.tensor(c, dtype=torch.long) for c in coords]
        pairs.append((key, obj))
    else:
        for i, item in enumerate(obj):
            _collect(item, coords + [i], pairs)
def convert_nested_list_coordinates_to_pairs_coordinates(
    nested: Any,
) -> List[Pair]:
    """
    Flatten a nested list into a list of (key, value) pairs.
    Keys are the multi-dimensional coordinates (as scalar index tensors)
    to reach each leaf value in the nested structure.
    Args:
        nested: A nested list where leaves are list[torch.Tensor] (index tensor lists).
    Returns:
        List of (key, value) pairs where:
        - key: list of scalar torch.Tensor coordinates
        - value: the leaf value (list[torch.Tensor])
    """
    pairs: List[Pair] = []
    _collect(nested, [], pairs)
    return pairs
