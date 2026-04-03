import torch
from typing import Any, List, Tuple
from collections import defaultdict
Pair = Tuple[List[torch.Tensor], List[torch.Tensor]]
def transpose_pairs_coordinates(
    pairs: List[Pair],
) -> List[Pair]:
    """
    Transpose coordinate pairs: swap keys and values.
    Each input pair is (key, value) where key is a list of scalar index tensors
    (coordinates) and value is a list of multi-element index tensors. This function
    expands each pair into element-level (source_coord, target_coord) pairs, groups
    by target coordinate, and returns (target_coord, collected_source_indices) pairs.
    Args:
        pairs: List of (key, value) pairs.
            - key: list of scalar torch.Tensor (source coordinates)
            - value: list of torch.Tensor (target index tensors, possibly multi-element)
    Returns:
        List of (output_key, output_value) pairs where:
        - output_key: list of scalar torch.Tensor — target coordinate
        - output_value: list of torch.Tensor — collected source indices
    """
    all_pairs = []
    for key_tensors, value_tensors in pairs:
        key_tuple = tuple(t.item() for t in key_tensors)
        if not value_tensors:
            continue
        int_lists = []
        for t in value_tensors:
            if t.dim() == 0:
                int_lists.append([t.item()])
            else:
                int_lists.append(t.tolist())
        value_coords = list(zip(*int_lists))
        for vc in value_coords:
            all_pairs.append((key_tuple, vc))
    groups = defaultdict(list)
    for key_tuple, value_tuple in all_pairs:
        groups[value_tuple].append(key_tuple)
    result = []
    for target_coord, source_coords in groups.items():
        out_key = [torch.tensor(c, dtype=torch.long) for c in target_coord]
        if source_coords and source_coords[0]:
            n_dims = len(source_coords[0])
            out_value = [
                torch.tensor([sc[d] for sc in source_coords], dtype=torch.long)
                for d in range(n_dims)
            ]
        else:
            out_value = []
        result.append((out_key, out_value))
    return result
