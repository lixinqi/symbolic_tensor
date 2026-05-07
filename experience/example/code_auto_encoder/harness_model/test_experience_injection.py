"""Validation tests for experience top-k injection in context gather stage.

Three levels:
  1. Pure unit tests: _build_context_query, _fetch_experience_snippets
  2. Injection logic tests: mock _call_llm, verify user_prompt content
  3. Regression test: experience=None, verify prompts unchanged
"""

import asyncio
import os
import sys
import tempfile
import unittest
from typing import List
from unittest.mock import AsyncMock, patch

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from experience.example.code_auto_encoder.harness_model.harness_model import (
    _build_context_query,
    _fetch_experience_snippets,
)
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_experience(rows: List[dict], tmpdir: str) -> object:
    """Build a [N, 3] experience tensor from a list of {query, key, value} dicts."""
    nested = [
        [row.get("query", ""), row.get("key", ""), row.get("value", "")]
        for row in rows
    ]
    return make_tensor(nested, tmpdir)


def _make_worktree(tmpdir: str, target_file: str, content: str, mask_line: int = 5) -> str:
    """Create a minimal worktree with a target file and .cloze_task.json."""
    import json
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


# ---------------------------------------------------------------------------
# Level 1: Pure function unit tests
# ---------------------------------------------------------------------------

class TestBuildContextQuery(unittest.TestCase):

    def test_filename_keywords_extracted(self):
        # basename of "st_moe.py" → stem "st_moe" → keywords ["st", "moe"]
        task = {"target_file": "symbolic_tensor/st_moe.py"}
        result = _build_context_query(task, [])
        self.assertIn("st", result)
        self.assertIn("moe", result)
        # directory components are NOT included (only basename is used)
        keywords = result.split("\n")
        self.assertNotIn("symbolic", keywords)
        self.assertNotIn("tensor", keywords)

    def test_class_def_names_extracted(self):
        task = {"target_file": "foo/bar.py"}
        lines = [
            "import os\n",
            "class SwitchMoE(nn.Module):\n",
            "    def forward(self, x):\n",
            "        pass\n",
        ]
        result = _build_context_query(task, lines)
        self.assertIn("SwitchMoE", result)
        self.assertIn("forward", result)

    def test_only_first_60_lines_scanned(self):
        task = {"target_file": "x.py"}
        lines = ["    pass\n"] * 60 + ["class LateClass:\n"]
        result = _build_context_query(task, lines)
        self.assertNotIn("LateClass", result)

    def test_empty_target_lines(self):
        task = {"target_file": "harness_model.py"}
        result = _build_context_query(task, [])
        self.assertIn("harness", result)
        self.assertIn("model", result)

    def test_no_crash_empty_task(self):
        result = _build_context_query({}, [])
        self.assertIsInstance(result, str)


class TestFetchExperienceSnippets(unittest.TestCase):

    def test_none_experience_returns_empty(self):
        result = _fetch_experience_snippets(None, "query", topk=2)
        self.assertEqual(result, "")

    def test_all_todo_values_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = _make_experience(
                [{"query": "foo", "key": "foo", "value": "TODO fill in value"}],
                tmpdir,
            )
            result = _fetch_experience_snippets(exp, "foo", topk=1)
            self.assertEqual(result, "")

    def test_valid_value_returned_in_snippet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = _make_experience(
                [{"query": "st moe forward", "key": "st moe", "value": "read(file_path='st_moe.py')"}],
                tmpdir,
            )
            result = _fetch_experience_snippets(exp, "st\nmoe\nforward", topk=1)
            self.assertIn("=== Similar past tool traces ===", result)
            self.assertIn("read(file_path='st_moe.py')", result)

    def test_topk_limits_snippets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = [
                {"query": f"row{i}", "key": f"row{i}", "value": f"trace {i}"}
                for i in range(5)
            ]
            exp = _make_experience(rows, tmpdir)
            # topk=2 should return at most 2 examples
            result = _fetch_experience_snippets(exp, "row0", topk=2)
            count = result.count("[Example")
            self.assertLessEqual(count, 2)

    def test_mixed_todo_and_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = _make_experience(
                [
                    {"query": "st moe", "key": "st", "value": "TODO"},
                    {"query": "st moe forward", "key": "st moe", "value": "grep(pattern='SwitchMoE')"},
                ],
                tmpdir,
            )
            result = _fetch_experience_snippets(exp, "st\nmoe\nforward", topk=2)
            self.assertNotIn("TODO", result)
            self.assertIn("grep(pattern='SwitchMoE')", result)

    def test_exception_in_retrieval_returns_empty(self):
        # Pass a non-tensor object to trigger an exception
        result = _fetch_experience_snippets(object(), "query", topk=1)
        self.assertEqual(result, "")


# ---------------------------------------------------------------------------
# Level 2: Injection logic tests (mock LLM)
# ---------------------------------------------------------------------------

class TestInjectionLogic(unittest.TestCase):

    def _make_model(self, experience, topk=1):
        from experience.example.code_auto_encoder.harness_model.harness_model import HarnessModel
        return HarnessModel(experience=experience, topk=topk)

    def _run_tool_use(self, model, worktree_tensor, coords, prompt):
        """Invoke _make_tool_use's inner async function with given coords."""
        fn = model._make_tool_use(worktree_tensor)
        return asyncio.get_event_loop().run_until_complete(fn(coords, prompt))

    def _run_tool_use_capture_prompt(self, model, worktree_tensor, coords, prompt):
        """Run tool_use and capture the user_prompt sent to _call_llm."""
        captured = {}

        async def fake_llm(system, user, method):
            captured["system"] = system
            captured["user"] = user
            return "CONTEXT_SUFFICIENT"

        fn = model._make_tool_use(worktree_tensor)

        import experience.example.code_auto_encoder.harness_model.harness_model as hm_mod
        original = hm_mod._call_llm
        hm_mod._call_llm = fake_llm
        try:
            result = asyncio.get_event_loop().run_until_complete(fn(coords, prompt))
        finally:
            hm_mod._call_llm = original
        return result, captured

    def test_bootstrap_no_llm_call(self):
        """collect_idx=0, retry_idx=0 → LLM never called regardless of experience."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = "class Foo:\n    def bar(self): pass\n"
            _make_worktree(tmpdir, "foo.py", content, mask_line=2)
            wt = make_tensor([tmpdir], tmpdir)
            exp = _make_experience(
                [{"query": "foo bar", "key": "foo", "value": "grep(pattern='Foo')"}],
                tmpdir,
            )
            model = self._make_model(exp)
            captured = {}

            import experience.example.code_auto_encoder.harness_model.harness_model as hm_mod
            original = hm_mod._call_llm

            async def fake_llm(system, user, method):
                captured["called"] = True
                return "CONTEXT_SUFFICIENT"

            hm_mod._call_llm = fake_llm
            try:
                fn = model._make_tool_use(wt)
                asyncio.get_event_loop().run_until_complete(fn([0, 0, 0], ""))
            finally:
                hm_mod._call_llm = original

            self.assertNotIn("called", captured, "LLM should NOT be called at bootstrap")

    def test_collect_idx_1_experience_injected(self):
        """collect_idx=1, retry_idx=0 → experience snippets prepended to user_prompt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = "class Foo:\n    def bar(self): pass\n" * 30
            _make_worktree(tmpdir, "foo.py", content, mask_line=2)
            wt = make_tensor([tmpdir], tmpdir)
            exp = _make_experience(
                [{"query": "foo bar Foo bar", "key": "foo", "value": "grep(pattern='Foo')"}],
                tmpdir,
            )
            model = self._make_model(exp, topk=1)
            _, captured = self._run_tool_use_capture_prompt(
                model, wt, [0, 1, 0], "some prior context"
            )
            self.assertIn("user", captured)
            self.assertIn("=== Similar past tool traces ===", captured["user"],
                          "experience snippets must appear in user_prompt at collect_idx=1")

    def test_retry_idx_1_uses_cached_snippets(self):
        """collect_idx=1, retry_idx=1 → no re-retrieval; cached snippets still injected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = "class Foo:\n    def bar(self): pass\n" * 30
            _make_worktree(tmpdir, "foo.py", content, mask_line=2)
            wt = make_tensor([tmpdir], tmpdir)
            exp = _make_experience(
                [{"query": "foo bar Foo bar", "key": "foo", "value": "grep(pattern='Foo')"}],
                tmpdir,
            )
            model = self._make_model(exp, topk=1)
            retrieval_calls = []
            captured_prompts = []

            import experience.example.code_auto_encoder.harness_model.harness_model as hm_mod

            original_fetch = hm_mod._fetch_experience_snippets

            def spy_fetch(experience, query_str, topk):
                retrieval_calls.append(1)
                return original_fetch(experience, query_str, topk)

            hm_mod._fetch_experience_snippets = spy_fetch
            try:
                # Use the SAME fn so both calls share the same _experience_cache
                fn = model._make_tool_use(wt)

                async def run_both():
                    orig = hm_mod._call_llm
                    async def fake_llm(system, user, method):
                        captured_prompts.append(user)
                        return "CONTEXT_SUFFICIENT"
                    hm_mod._call_llm = fake_llm
                    try:
                        await fn([0, 1, 0], "ctx")  # first: triggers retrieval + caches
                        await fn([0, 1, 1], "ctx")  # retry: uses cache
                    finally:
                        hm_mod._call_llm = orig

                asyncio.get_event_loop().run_until_complete(run_both())
            finally:
                hm_mod._fetch_experience_snippets = original_fetch

            self.assertEqual(len(retrieval_calls), 1, "retrieval should run exactly once (cache)")
            self.assertGreaterEqual(len(captured_prompts), 2)
            self.assertIn("=== Similar past tool traces ===", captured_prompts[1],
                          "cached snippets must still be injected on retry")

    def test_collect_idx_2_uses_same_cache(self):
        """collect_idx=2, retry_idx=0 → batch_idx already in cache; no second retrieval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = "class Foo:\n    def bar(self): pass\n" * 30
            _make_worktree(tmpdir, "foo.py", content, mask_line=2)
            wt = make_tensor([tmpdir], tmpdir)
            exp = _make_experience(
                [{"query": "foo bar Foo bar", "key": "foo", "value": "grep(pattern='Foo')"}],
                tmpdir,
            )
            model = self._make_model(exp, topk=1)
            retrieval_calls = []
            import experience.example.code_auto_encoder.harness_model.harness_model as hm_mod

            original_fetch = hm_mod._fetch_experience_snippets

            def spy_fetch(experience, query_str, topk):
                retrieval_calls.append(1)
                return original_fetch(experience, query_str, topk)

            hm_mod._fetch_experience_snippets = spy_fetch
            try:
                # Same model instance → same _experience_cache
                fn = model._make_tool_use(wt)

                async def run_two():
                    import experience.example.code_auto_encoder.harness_model.harness_model as m
                    orig = m._call_llm
                    m._call_llm = AsyncMock(return_value="CONTEXT_SUFFICIENT")
                    try:
                        await fn([0, 1, 0], "ctx1")
                        await fn([0, 2, 0], "ctx2")
                    finally:
                        m._call_llm = orig

                asyncio.get_event_loop().run_until_complete(run_two())
            finally:
                hm_mod._fetch_experience_snippets = original_fetch

            self.assertEqual(len(retrieval_calls), 1, "collect_idx=2 reuses cache; no second retrieval")

    def test_no_experience_no_snippet_in_prompt(self):
        """experience=None → no experience header in user_prompt at any collect_idx."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = "class Foo:\n    pass\n"
            _make_worktree(tmpdir, "foo.py", content, mask_line=1)
            wt = make_tensor([tmpdir], tmpdir)
            model = self._make_model(experience=None)
            _, captured = self._run_tool_use_capture_prompt(model, wt, [0, 1, 0], "ctx")
            self.assertNotIn("=== Similar past tool traces ===", captured.get("user", ""),
                             "no experience header when experience=None")


# ---------------------------------------------------------------------------
# Level 3: Regression test
# ---------------------------------------------------------------------------

class TestRegressionNoExperience(unittest.TestCase):

    def test_user_prompt_unchanged_vs_baseline(self):
        """experience=None: user_prompt must match exact baseline (no injection side-effects)."""
        import json
        with tempfile.TemporaryDirectory() as tmpdir:
            content = "def foo():\n    pass\n"
            _make_worktree(tmpdir, "foo.py", content, mask_line=1)
            wt = make_tensor([tmpdir], tmpdir)

            # Build expected baseline prompt manually (same logic as _make_tool_use)
            task = {"target_file": "foo.py", "mask_line": 1}
            prompt = "some prior context"
            already_read = "read(" in prompt and task["target_file"] in prompt
            if already_read:
                read_hint = (
                    f"NOTE: You have already read {task['target_file']}. "
                    "Do NOT read it again. Instead, search for related files in the same directory "
                    "or grep for function/variable names from the mask region, then read the most "
                    "relevant files. Or declare CONTEXT_SUFFICIENT if you have enough context."
                )
            else:
                read_hint = (
                    "Strategy: start by reading the target file around the mask (offset=0, limit=200). "
                    "If the mask is inside a function whose body is unclear, search for "
                    "similar patterns in other files and read the most relevant files."
                )
            expected = (
                f"Task: Recover the masked code in {task['target_file']} "
                f"(line {task['mask_line']}).\n"
                f"Current context:\n{prompt}\n\n"
                f"{read_hint}\n"
                f"Decide the next action."
            )

            captured = {}

            async def fake_llm(system, user, method):
                captured["user"] = user
                return "CONTEXT_SUFFICIENT"

            from experience.example.code_auto_encoder.harness_model.harness_model import HarnessModel
            import experience.example.code_auto_encoder.harness_model.harness_model as hm_mod

            model = HarnessModel(experience=None)
            fn = model._make_tool_use(wt)
            original = hm_mod._call_llm
            hm_mod._call_llm = fake_llm
            try:
                asyncio.get_event_loop().run_until_complete(fn([0, 1, 0], prompt))
            finally:
                hm_mod._call_llm = original

            self.assertEqual(captured.get("user"), expected,
                             "user_prompt must be identical to baseline when experience=None")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestBuildContextQuery,
        TestFetchExperienceSnippets,
        TestInjectionLogic,
        TestRegressionNoExperience,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
