import os
import time
import subprocess
import tempfile
import torch
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.function.st_moe_forward import st_moe_forward
def run_benchmark(input_data, experience_data, topk, llm_method):
    """Run a single benchmark, return (time_seconds, outputs)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_tensor = make_tensor(input_data, tmpdir)
        experience_tensor = make_tensor(experience_data, tmpdir)
        t0 = time.time()
        output, selected_indexes = st_moe_forward(
            input_tensor, experience_tensor,
            topk=topk,
            llm_method=llm_method,
        )
        t1 = time.time()
        elapsed = t1 - t0
        outputs = []
        root = os.path.join(tmpdir, output.st_tensor_uid, "storage")
        for i in range(output.numel()):
            digits = list(str(i))
            path = os.path.join(root, os.path.join(*digits), "data")
            with open(path) as f:
                content = f.read()
            outputs.append(content)
        return elapsed, outputs
