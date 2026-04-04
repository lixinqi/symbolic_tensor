"""Test that symbolic st_attention aligns with traditional transformer attention behavior.
Compares the symbolic pipeline (slice_attention + merge) against the behavioral
properties of traditional scaled dot-product attention:
  output[b, i] = Aggregate(input[b, j] for j where mask[b, i, j])
"""
import os
import tempfile
import torch
from experience.symbolic_tensor.function.st_attention import st_attention
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.function.get_causal_attention_mask import (
    get_causal_attention_mask,
)
from experience.fs_util.text_merger import TextMerger
def read_storage(tensor, flat_index):
    digits = list(str(flat_index))
    path = os.path.join(
        tensor.st_relative_to,
        tensor.st_tensor_uid,
        "storage",
        os.path.join(*digits),
        "data",
    )
    path = os.path.realpath(path)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return f.read()
def get_frames(tensor, flat_index):
    """Read and unpack frames at a flat index."""
    content = read_storage(tensor, flat_index)
    if content is None:
        return None
    return TextMerger.unpack(content)
def frame_texts(frames):
    """Extract just the text content from frames."""
    return [f[2] for f in frames]
