import torch
def group_random_select(group_ids):
    """
    Randomly select one element uniformly from each group, returning original indices.
    group_ids: (N,) integer tensor
    """
    sorted_idx = torch.argsort(group_ids)
    sorted_groups = group_ids[sorted_idx]
    start_mask = torch.cat([torch.tensor([True], device=group_ids.device),
                            sorted_groups[1:] != sorted_groups[:-1]])
    group_starts = torch.nonzero(start_mask, as_tuple=True)[0]
    group_ends = torch.cat([group_starts[1:],
                            torch.tensor([len(sorted_groups)], device=group_ids.device)])
    group_sizes = group_ends - group_starts
    rand_float = torch.rand(len(group_sizes), device=group_ids.device)
    local_offsets = (rand_float * group_sizes).long()
    global_pos_in_sorted = group_starts + local_offsets
    original_indices = sorted_idx[global_pos_in_sorted]
    return original_indices
