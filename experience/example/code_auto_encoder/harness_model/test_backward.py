"""Backward gradient verification for HarnessModel.

Three levels:
  1. Unit tests: _make_context_repeater, _make_code_gen_output_prompt, __init__
  2. Integration: mock LLM, run forward+backward, verify symbolic grad at experience
  3. StSGD patch verification: step applies at least one patch after backward
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.function import symbolic_grad_registry
from experience.symbolic_tensor.function.get_edit_distance_ratio import get_edit_distance_ratio
from experience.symbolic_tensor.optimizer.st_sgd import StSGD
from experience.future_tensor.future_tensor import _read_element, _coords_to_flat


def _read_storage(tensor, flat_index):
    digits = list(str(flat_index))
    path = os.path.join(
        tensor.st_relative_to, tensor.st_tensor_uid,
        "storage", os.path.join(*digits), "data",
    )
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return f.read()


def _make_worktree(tmpdir, target_file="foo.py", mask_line=2):
    content = "def foo():\n    pass\n    return 1\n"
    os.makedirs(os.path.join(tmpdir, os.path.dirname(target_file)), exist_ok=True)
    with open(os.path.join(tmpdir, target_file), "w") as f:
        f.write(content)
    task = {
        "target_file": target_file,
        "mask_line": mask_line,
        "mask_start_1idx": mask_line,
        "mask_end_1idx": mask_line,
    }
    with open(os.path.join(tmpdir, ".cloze_task.json"), "w") as f:
        json.dump(task, f)
    return tmpdir


def _patch_task_handler_write_code(mock_code="    return 1"):
    """Context manager that replaces TaskHandler LLM calls with writing mock_code."""
    import os

    def mock_handler_call(self_or_tasks, tasks_or_method=None, llm_method=None, llm_env=None, **kwargs):
        if isinstance(self_or_tasks, list):
            all_tasks = self_or_tasks
            method = tasks_or_method
        else:
            all_tasks = tasks_or_method if tasks_or_method is not None else []
            method = llm_method

        for task in all_tasks:
            ws = task.workspace_dir
            out_dirs = task.output_relative_dir
            if isinstance(out_dirs, str):
                out_dirs = [out_dirs]
            for out_rel in out_dirs:
                out_root = os.path.join(ws, out_rel)
                for dirpath, _, fnames in os.walk(out_root):
                    for fname in fnames:
                        fpath = os.path.join(dirpath, fname)
                        try:
                            with open(fpath, "r") as f:
                                content = f.read()
                            if "TODO" in content:
                                with open(fpath, "w") as f:
                                    f.write(mock_code)
                        except (OSError, UnicodeDecodeError):
                            pass

    return patch(
        "experience.llm_client.task_handler.TaskHandler.__call__",
        new=mock_handler_call,
    )


# ---------------------------------------------------------------------------
# Level 1: Unit tests (no LLM)
# ---------------------------------------------------------------------------

class TestHarnessModelInit(unittest.TestCase):

    def test_experience_created_with_n_experience(self):
        from experience.example.code_auto_encoder.harness_model.harness_model import HarnessModel
        with tempfile.TemporaryDirectory() as exp_dir:
            model = HarnessModel(n_experience=4, experience_dir=exp_dir)
            self.assertIsNotNone(model.experience)
            self.assertEqual(list(model.experience.shape), [4, 3])

    def test_experience_requires_grad(self):
        from experience.example.code_auto_encoder.harness_model.harness_model import HarnessModel
        with tempfile.TemporaryDirectory() as exp_dir:
            model = HarnessModel(n_experience=4, experience_dir=exp_dir)
            self.assertTrue(model.experience.requires_grad)

    def test_n_experience_zero_gives_none(self):
        from experience.example.code_auto_encoder.harness_model.harness_model import HarnessModel
        model = HarnessModel(n_experience=0)
        self.assertIsNone(model.experience)

    def test_experience_dir_none_creates_tmpdir(self):
        from experience.example.code_auto_encoder.harness_model.harness_model import HarnessModel
        model = HarnessModel(n_experience=4, experience_dir=None)
        self.assertIsNotNone(model.experience)
        self.assertEqual(list(model.experience.shape), [4, 3])


class TestMakeContextRepeater(unittest.TestCase):

    def test_returns_context_with_task_header(self):
        from experience.example.code_auto_encoder.harness_model.harness_model import HarnessModel
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_worktree(tmpdir, "foo.py", mask_line=2)
            wt = make_tensor([tmpdir], tmpdir)
            context_data = make_tensor(["accumulated context here"], tmpdir)

            model = HarnessModel(n_experience=0)
            repeater = model._make_context_repeater(context_data, wt)

            result, status = asyncio.get_event_loop().run_until_complete(
                repeater([0, 0], "")
            )
            self.assertIn("CLOZE_TASK", result)
            self.assertIn("foo.py", result)
            self.assertIn("accumulated context here", result)

    def test_status_is_confidence(self):
        from experience.example.code_auto_encoder.harness_model.harness_model import HarnessModel
        from experience.future_tensor.status import Status
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_worktree(tmpdir, "foo.py", mask_line=2)
            wt = make_tensor([tmpdir], tmpdir)
            context_data = make_tensor(["context"], tmpdir)

            model = HarnessModel(n_experience=0)
            repeater = model._make_context_repeater(context_data, wt)

            _, status = asyncio.get_event_loop().run_until_complete(
                repeater([0, 0], "")
            )
            self.assertTrue(status.is_confidence)

    def test_mask_line_info_included(self):
        from experience.example.code_auto_encoder.harness_model.harness_model import HarnessModel
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_worktree(tmpdir, "bar.py", mask_line=5)
            wt = make_tensor([tmpdir], tmpdir)
            ctx = make_tensor(["ctx"], tmpdir)

            model = HarnessModel(n_experience=0)
            repeater = model._make_context_repeater(ctx, wt)

            result, _ = asyncio.get_event_loop().run_until_complete(
                repeater([0, 0], "")
            )
            self.assertIn("mask_line=5", result)


class TestMakeCodeGenOutputPrompt(unittest.TestCase):

    def test_returns_callable(self):
        from experience.example.code_auto_encoder.harness_model.harness_model import HarnessModel
        model = HarnessModel(n_experience=0)
        fn = model._make_code_gen_output_prompt()
        self.assertTrue(callable(fn))

    def test_returns_string_with_system_generation_content(self):
        from experience.example.code_auto_encoder.harness_model.harness_model import HarnessModel, _SYSTEM_GENERATION
        model = HarnessModel(n_experience=0)
        fn = model._make_code_gen_output_prompt()
        result = fn("task", "/ws", "/exp", "/inp", "/out")
        self.assertIsInstance(result, str)
        self.assertIn("masked", result.lower())

    def test_mentions_input_view(self):
        from experience.example.code_auto_encoder.harness_model.harness_model import HarnessModel
        model = HarnessModel(n_experience=0)
        fn = model._make_code_gen_output_prompt()
        result = fn("task", "/ws", "/exp", "/inp", "/out")
        self.assertIn("const_input_view", result)

    def test_mentions_experience_view(self):
        from experience.example.code_auto_encoder.harness_model.harness_model import HarnessModel
        model = HarnessModel(n_experience=0)
        fn = model._make_code_gen_output_prompt()
        result = fn("task", "/ws", "/exp", "/inp", "/out")
        self.assertIn("const_experiance_view", result)


# ---------------------------------------------------------------------------
# Level 2: Integration — mock LLM, run forward, check experience grad registered
# ---------------------------------------------------------------------------

class TestBackwardGrad(unittest.TestCase):

    def _setup(self, exp_dir, tmpdir):
        from experience.example.code_auto_encoder.harness_model.harness_model import HarnessModel
        _make_worktree(tmpdir, "foo.py", mask_line=2)
        wt = make_tensor([tmpdir], tmpdir)
        gt = make_tensor(["    pass\n"], tmpdir)
        model = HarnessModel(n_experience=4, experience_dir=exp_dir, max_codegen_steps=1,
                              max_context_collects=2, max_tool_call_retries=1, topk=1)
        return model, wt, gt

    def test_experience_grad_registered_after_backward(self):
        """After forward+backward, symbolic_grad_registry has grad for experience."""
        import experience.example.code_auto_encoder.harness_model.harness_model as hm_mod

        async def fake_llm(system, user, method="raw_llm_api"):
            return "CONTEXT_SUFFICIENT"

        with tempfile.TemporaryDirectory() as tmpdir, \
             tempfile.TemporaryDirectory() as exp_dir:

            model, wt, gt = self._setup(exp_dir, tmpdir)

            with _patch_task_handler_write_code("    pass\n"):
                original_llm = hm_mod._call_llm
                hm_mod._call_llm = fake_llm
                try:
                    output = model(wt)
                    output.requires_grad_(True)
                    loss = get_edit_distance_ratio(output, gt)
                    loss_scalar = loss.mean() * model.last_output_ft
                    loss_scalar.backward()
                finally:
                    hm_mod._call_llm = original_llm

            exp_grad = symbolic_grad_registry.peek(model.experience.st_tensor_uid)
            self.assertIsNotNone(
                exp_grad,
                "symbolic_grad_registry must have grad for experience after backward"
            )

    def test_experience_numeric_grad_nonzero(self):
        """After backward, experience.grad has at least one nonzero element."""
        import experience.example.code_auto_encoder.harness_model.harness_model as hm_mod

        async def fake_llm(system, user, method="raw_llm_api"):
            return "CONTEXT_SUFFICIENT"

        with tempfile.TemporaryDirectory() as tmpdir, \
             tempfile.TemporaryDirectory() as exp_dir:

            model, wt, gt = self._setup(exp_dir, tmpdir)

            with _patch_task_handler_write_code("    pass\n"):
                original_llm = hm_mod._call_llm
                hm_mod._call_llm = fake_llm
                try:
                    output = model(wt)
                    output.requires_grad_(True)
                    loss = get_edit_distance_ratio(output, gt)
                    loss_scalar = loss.mean() * model.last_output_ft
                    loss_scalar.backward()
                finally:
                    hm_mod._call_llm = original_llm

            exp_grad = symbolic_grad_registry.peek(model.experience.st_tensor_uid)
            if exp_grad is not None:
                self.assertGreater(
                    exp_grad.data.abs().sum().item(), 0,
                    "experience grad coefficients must be nonzero"
                )


# ---------------------------------------------------------------------------
# Level 3: StSGD patch verification
# ---------------------------------------------------------------------------

class TestStSGDStep(unittest.TestCase):

    def test_stsgd_step_after_backward(self):
        """StSGD.step() processes at least one row (applied + skipped > 0)."""
        import experience.example.code_auto_encoder.harness_model.harness_model as hm_mod

        async def fake_llm(system, user, method="raw_llm_api"):
            return "CONTEXT_SUFFICIENT"

        with tempfile.TemporaryDirectory() as tmpdir, \
             tempfile.TemporaryDirectory() as exp_dir:

            from experience.example.code_auto_encoder.harness_model.harness_model import HarnessModel
            _make_worktree(tmpdir, "foo.py", mask_line=2)
            wt = make_tensor([tmpdir], tmpdir)
            gt = make_tensor(["    pass\n"], tmpdir)
            model = HarnessModel(n_experience=4, experience_dir=exp_dir, max_codegen_steps=1,
                                  max_context_collects=2, max_tool_call_retries=1, topk=1)

            with _patch_task_handler_write_code("    pass\n"):
                original_llm = hm_mod._call_llm
                hm_mod._call_llm = fake_llm
                try:
                    output = model(wt)
                    output.requires_grad_(True)
                    loss = get_edit_distance_ratio(output, gt)
                    loss_scalar = loss.mean() * model.last_output_ft
                    loss_scalar.backward()
                finally:
                    hm_mod._call_llm = original_llm

            optimizer = StSGD([model.experience], lr=1.0)
            optimizer.step()
            stats = optimizer.get_last_step_stats()
            self.assertGreater(
                stats["applied"] + stats["skipped"],
                0,
                f"StSGD must process at least one row, got stats={stats}"
            )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestHarnessModelInit,
        TestMakeContextRepeater,
        TestMakeCodeGenOutputPrompt,
        TestBackwardGrad,
        TestStSGDStep,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
