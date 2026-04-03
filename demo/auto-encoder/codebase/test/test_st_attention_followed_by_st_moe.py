import os
import subprocess
import tempfile
import torch
from typing import List, Optional
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.function.st_attention import st_attention
from experience.symbolic_tensor.function.st_moe import st_moe
from experience.symbolic_tensor.module.st_moe import StMoeModule
from experience.symbolic_tensor.module.with_dense_view import WithDenseView
from experience.symbolic_tensor.function.st_moe_backward import (
    _read_storage, _detect_input_content_type, _PLAIN, _MERGED,
)
from experience.symbolic_tensor.function.st_copy import copy_impl
from experience.symbolic_tensor.function.get_causal_attention_mask import (
    get_causal_attention_mask,
)
from experience.symbolic_tensor.optimizer.st_sgd import StSGD
from experience.fs_util.text_merger import TextMerger, kFrameMarker
def run_test(name: str, condition: bool, expected=None, actual=None):
    if condition:
        print(f"  \u2713 {name}")
    else:
        print(f"  \u2717 {name}")
        if expected is not None and actual is not None:
            print(f"    expected: {expected}")
            print(f"    actual:   {actual}")
def read_storage(tensor, flat_index):
    digits = list(str(flat_index))
    path = os.path.join(
        tensor.st_relative_to, tensor.st_tensor_uid,
        "storage", os.path.join(*digits), "data",
    )
    path = os.path.realpath(path)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return f.read()
def data_file_sizes(tensor):
    """Return dict {flat_index: file_size} for all data files."""
    storage_dir = os.path.join(tensor.st_relative_to, tensor.st_tensor_uid, "storage")
    result = {}
    for root, dirs, files in os.walk(storage_dir):
        for f in files:
            if f == "data":
                rel = os.path.relpath(root, storage_dir)
                flat_idx = int("".join(rel.split(os.sep)))
                fpath = os.path.realpath(os.path.join(root, f))
                result[flat_idx] = os.path.getsize(fpath) if os.path.isfile(fpath) else 0
    return result
def check_coeff_storage_consistency(tensor, label):
    """Verify coefficient > 0 iff data file exists with non-zero size."""
    sizes = data_file_sizes(tensor)
    flat_data = tensor.data.flatten()
    ok = True
    for i in range(tensor.numel()):
        coeff = flat_data[i].item()
        has_file = i in sizes and sizes[i] > 0
        if coeff > 0 and not has_file:
            print(f"    {label}[{i}]: coeff={coeff} but no data file")
            ok = False
        elif coeff == 0 and has_file:
            print(f"    {label}[{i}]: coeff=0 but data file exists (size={sizes[i]})")
            ok = False
    return ok
def build_model(tmpdir, experience_data, topk=1, task_prompt="Translate English to French"):
    """Build a WithDenseView(StMoeModule) with loaded experience."""
    num_entries = len(experience_data)
    moe = StMoeModule(
        experience_shape=[num_entries, 3],
        topk=topk,
        task_prompt=task_prompt,
    )
    src = make_tensor(experience_data, tmpdir)
    loaded = copy_impl(src, moe._experience_dir)
    moe.experience = loaded
    moe.experience.requires_grad_(True)
    model = WithDenseView(dense_handler=lambda x: moe(x)[0])
    model._moe = moe
    return model
def run_pipeline(model, inp, mask):
    """Run st_attention → WithDenseView(StMoe) full pipeline."""
    attn_out = st_attention(inp, mask)
    output = model(attn_out)
    return attn_out, output
