import os
import subprocess
import tempfile
import torch
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.optimizer.st_sgd import StSGD
from experience.symbolic_tensor.function.st_moe_forward import st_moe_forward
from experience.symbolic_tensor.function.st_moe_backward import st_moe_backward
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
