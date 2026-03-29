import torch
import torch.nn as nn
from typing import Callable, Dict, Optional

from experience.symbolic_tensor.function.slice_attention import slice_attention


class AttentionSlicer(nn.Module):
    """Module wrapper for slice_attention (autograd.Function).

    Stores forward/backward configuration and applies slice_attention
    in forward().

    Args:
        return_view: If True, use symlinks (views) instead of copies in forward.
        grad_input_prompt: Custom prompt callable for backward. None uses default.
        task_prompt: High-level task description.
        llm_method: LLM backend to use.
        llm_env: Environment variable dict for LLM client.
    """

    def __init__(
        self,
        return_view: bool = False,
        grad_input_prompt: Optional[Callable[..., str]] = None,
        task_prompt: str = "",
        llm_method: str = "raw_llm_api",
        llm_env: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.return_view = return_view
        self.grad_input_prompt = grad_input_prompt
        self.task_prompt = task_prompt
        self.llm_method = llm_method
        self.llm_env = llm_env

    def forward(
        self,
        input: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return slice_attention(
            input, attention_mask,
            self.return_view, self.grad_input_prompt,
            self.task_prompt, self.llm_method, self.llm_env,
        )


if __name__ == "__main__":
    import os
    import tempfile
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor

    print("Running AttentionSlicer module tests...\n")

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

    # Test 1: Construction with defaults
    print("Test 1: Construction with defaults")
    module = AttentionSlicer()
    run_test("is nn.Module", isinstance(module, nn.Module))
    run_test("return_view default False", module.return_view is False)
    run_test("grad_input_prompt default None", module.grad_input_prompt is None)
    run_test("task_prompt default ''", module.task_prompt == "")
    run_test("llm_method default 'raw_llm_api'", module.llm_method == "raw_llm_api")
    run_test("llm_env default None", module.llm_env is None)

    # Test 2: Construction with custom params
    print("Test 2: Construction with custom params")
    module = AttentionSlicer(
        return_view=True,
        task_prompt="Test task",
        llm_method="raw_llm_api",
    )
    run_test("return_view True", module.return_view is True)
    run_test("task_prompt stored", module.task_prompt == "Test task")

    # Test 3: Forward pass (copy mode)
    print("Test 3: Forward pass (copy mode)")
    with tempfile.TemporaryDirectory() as tmpdir:
        module = AttentionSlicer()
        inp = make_tensor([["hello", "world"]], tmpdir)
        mask = torch.tensor([[[True, False], [True, True]]])

        output = module(inp, mask)
        run_test("Output shape (1, 2, 2)", list(output.shape) == [1, 2, 2])
        run_test("Output has st_relative_to", hasattr(output, "st_relative_to"))
        run_test("[0,0,0]='hello'", read_storage(output, 0) == "hello")
        run_test("[0,1,0]='hello'", read_storage(output, 2) == "hello")
        run_test("[0,1,1]='world'", read_storage(output, 3) == "world")
        run_test("attended are 1.0", output[mask].eq(1.0).all().item())

    # Test 4: Forward pass (view mode)
    print("Test 4: Forward pass (view mode)")
    with tempfile.TemporaryDirectory() as tmpdir:
        module = AttentionSlicer(return_view=True)
        inp = make_tensor([["alpha", "beta"]], tmpdir)
        mask = torch.tensor([[[True, False], [True, True]]])

        output = module(inp, mask)
        run_test("Output shape (1, 2, 2)", list(output.shape) == [1, 2, 2])
        run_test("[0,0,0]='alpha'", read_storage(output, 0) == "alpha")
        run_test("[0,1,0]='alpha'", read_storage(output, 2) == "alpha")
        run_test("[0,1,1]='beta'", read_storage(output, 3) == "beta")

    # Test 5: Causal mask
    print("Test 5: Causal mask 1x3")
    with tempfile.TemporaryDirectory() as tmpdir:
        module = AttentionSlicer()
        inp = make_tensor([["a", "b", "c"]], tmpdir)
        mask = torch.tril(torch.ones(1, 3, 3, dtype=torch.bool))

        output = module(inp, mask)
        run_test("shape (1, 3, 3)", list(output.shape) == [1, 3, 3])
        run_test("[0,2,0]='a'", read_storage(output, 6) == "a")
        run_test("[0,2,1]='b'", read_storage(output, 7) == "b")
        run_test("[0,2,2]='c'", read_storage(output, 8) == "c")

    # Test 6: Empty mask
    print("Test 6: Empty mask")
    with tempfile.TemporaryDirectory() as tmpdir:
        module = AttentionSlicer()
        inp = make_tensor([["a", "b"]], tmpdir)
        mask = torch.zeros(1, 2, 2, dtype=torch.bool)

        output = module(inp, mask)
        run_test("all zero", (output == 0).all().item())

    # Test 7: Multi-batch
    print("Test 7: Multi-batch 2x2")
    with tempfile.TemporaryDirectory() as tmpdir:
        module = AttentionSlicer()
        inp = make_tensor([["p", "q"], ["r", "s"]], tmpdir)
        mask = torch.zeros(2, 2, 2, dtype=torch.bool)
        mask[0, 1, 0] = True
        mask[0, 1, 1] = True
        mask[1, 0, 0] = True

        output = module(inp, mask)
        run_test("shape (2, 2, 2)", list(output.shape) == [2, 2, 2])
        run_test("b0[1,0]='p'", read_storage(output, 2) == "p")
        run_test("b0[1,1]='q'", read_storage(output, 3) == "q")
        run_test("b1[0,0]='r'", read_storage(output, 4) == "r")

    print("\nAll tests completed.")
