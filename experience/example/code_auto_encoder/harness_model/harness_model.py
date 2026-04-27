"""Harness Model: Simulate Claude Code with SymbolicTensor Framework.

Two-stage architecture:
  1. code_context_gather: ft_recurrent with accumulate_output=concat
  2. code_gen: ft_recurrent with validation
"""

import json
import os
import sys
import tempfile
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from experience.future_tensor.future_tensor import FutureTensor, _read_element
from experience.future_tensor.function.ft_recurrent import ft_recurrent
from experience.future_tensor.status import Status
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.tensor_util.todo_tensor_like import todo_tensor_like
from experience.llm_client.raw_llm_query import raw_llm_query
from experience.llm_client.agent_config_factory import AgentConfigFactory

from experience.example.code_auto_encoder.harness_model.function import (
    ALL_OPS,
    ALL_VALIDATORS,
)
from experience.example.code_auto_encoder.harness_model.function.harness_validator_op import HarnessValidatorOp


_SYSTEM_TOOL_SCHEMA = """\
You are a coding agent that gathers context from a codebase to recover masked code.

Available tools:
- glob(pattern: str) — Find files matching a glob pattern.
- grep(pattern: str, glob: str | None = None) — Search file contents by regex.
- read(file_path: str, offset: int = 0, limit: int = 200) — Read a file with line numbers.

Call a tool by outputting EXACTLY:
  tool_name(arg1="value1", arg2="value2")

If you have gathered enough context, output:
  CONTEXT_SUFFICIENT

Important: when reading the target file, read a range that includes the masked line and
several lines before and after it (e.g., offset=mask_line-5, limit=15) to get sufficient context.
"""

_SYSTEM_GENERATION = """\
You are a coding agent that recovers masked code from a codebase.

Given the codebase context gathered from tool calls, output ONLY the missing source code
that was in the masked region. No explanations, no markdown fences, no extra comments.
Do NOT output function signatures, decorators, or docstrings unless they were in the masked region.

The masked region is marked with <AUTOENCODER-CLOZE-MASK-PLACEHOLDER> in the gathered context.
Replace that placeholder with the original code. The mask may span multiple lines — output ALL
lines that were removed, preserving original indentation exactly.
"""


async def _call_llm(system: str, user: str, llm_method: str = "raw_llm_api") -> str:
    """Async LLM call."""
    prompt = f"{system}\n\n{user}"
    if llm_method == "raw_llm_api":
        config = AgentConfigFactory.create_raw_llm_config()
        return await raw_llm_query(prompt, config=config)
    raise ValueError(f"Unsupported llm_method: {llm_method}")


def _parse_tool_call(response: str) -> Tuple[str, dict]:
    """Parse 'tool_name(arg="val")' into (tool_name, kwargs dict)."""
    import re
    response = response.strip()
    if response.startswith("CONTEXT_SUFFICIENT"):
        return ("CONTEXT_SUFFICIENT", {})
    # Match tool_name(key="value", ...)
    m = re.match(r"(\w+)\s*\((.*)\)", response)
    if not m:
        return ("", {})
    tool_name = m.group(1)
    args_str = m.group(2)
    kwargs = {}
    # Simple key="value" parser
    for key, val in re.findall(r'(\w+)\s*=\s*"([^"]*)"', args_str):
        kwargs[key] = val
    for key, val in re.findall(r"(\w+)\s*=\s*'([^']*)'", args_str):
        kwargs[key] = val
    # Numeric args
    for key, val in re.findall(r'(\w+)\s*=\s*(\d+)', args_str):
        kwargs[key] = int(val)
    return tool_name, kwargs


class HarnessModel(nn.Module):
    def __init__(
        self,
        experience: Optional[torch.Tensor] = None,
        max_codegen_steps: int = 4,
        max_tool_call_retries: int = 5,
        topk: int = 2,
        task_prompt: str = "",
        llm_method: str = "raw_llm_api",
        llm_env: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.experience = experience
        self.max_codegen_steps = max_codegen_steps
        self.max_tool_call_retries = max_tool_call_retries
        self.topk = topk
        self.task_prompt = task_prompt
        self.llm_method = llm_method
        self.llm_env = llm_env
        self.ops = ALL_OPS
        self.validators = ALL_VALIDATORS

    def forward(self, worktree_tensor: torch.Tensor) -> torch.Tensor:
        batch_size = worktree_tensor.shape[0]
        tmpdir = worktree_tensor.st_relative_to

        # ── Stage 1: code_context_gather ──
        ft_gather = FutureTensor(
            [batch_size, self.max_tool_call_retries],
            tmpdir,
            self._make_code_context_gather(worktree_tensor),
        )

        def concat_context(acc: str, cur: str) -> str:
            # Extract clean result from trace format "[header]\nresult"
            if cur.startswith("[") and "\n" in cur:
                header, result = cur.split("\n", 1)
                # Only accumulate read results (clean file contents), skip grep/glob
                if not header.startswith("[read("):
                    return acc
                if result.startswith("ERROR:") or result.startswith("(file not found") or result.startswith("(read error") or result.startswith("(no matches") or result.startswith("(empty") or result.startswith("(regex error"):
                    clean = ""
                else:
                    clean = result
            elif cur.startswith("[invalid tool"):
                clean = ""
            else:
                clean = cur
            if not clean:
                return acc
            if not acc:
                return clean
            return acc + "\n\n---\n\n" + clean

        context_ft, _ = ft_recurrent(
            ft_gather,
            accumulate_output=concat_context,
            task_prompt=self.task_prompt,
            llm_method=self.llm_method,
        )

        # Materialize context
        context_prompts = make_tensor(["gather context"] * batch_size, tmpdir)
        context_ft.ft_forward(context_prompts)
        context_tensor = context_ft._tensor

        # ── Stage 2: code_gen ──
        ft_gen = FutureTensor(
            [batch_size, self.max_codegen_steps],
            tmpdir,
            self._make_code_gen(context_tensor, worktree_tensor),
        )

        output_ft, _ = ft_recurrent(
            ft_gen,
            task_prompt=self.task_prompt,
            llm_method=self.llm_method,
        )

        # Materialize output
        gen_prompts = make_tensor(["generate code"] * batch_size, tmpdir)
        output_ft.ft_forward(gen_prompts)
        return output_ft._tensor

    def _make_code_context_gather(self, worktree_tensor: torch.Tensor):
        """Build ft_async_get for context gathering stage."""

        async def code_context_gather(coords: List[int], prompt: str) -> Tuple[str, Status]:
            batch_idx = coords[0]
            retry_idx = coords[1]

            worktree_path = _read_element(worktree_tensor, batch_idx)

            # Read task metadata
            task_json_path = os.path.join(worktree_path, ".cloze_task.json")
            with open(task_json_path) as f:
                task = json.load(f)

            # Build user prompt
            already_read_target = "read(" in prompt and task["target_file"] in prompt
            if already_read_target:
                read_hint = (
                    f"NOTE: You have already read {task['target_file']}. "
                    f"Do NOT read it again. Instead, search for related files in the same directory "
                    f"or grep for function/variable names from the mask region, then read the most "
                    f"relevant files. Or declare CONTEXT_SUFFICIENT if you have enough context."
                )
            else:
                read_hint = (
                    f"Strategy: start by reading the target file around the mask (offset=0, limit=200). "
                    f"If the mask is inside a function whose body is unclear, search for "
                    f"similar patterns in other files and read the most relevant files."
                )
            user_prompt = (
                f"Task: Recover the masked code in {task['target_file']} "
                f"(line {task['mask_line']}).\n"
                f"Current context:\n{prompt}\n\n"
                f"{read_hint}\n"
                f"Decide the next action."
            )

            # Bootstrap: on first retry, automatically read the target file with a large window
            if retry_idx == 0:
                tool_name = "read"
                kwargs = {"file_path": task["target_file"], "offset": 0, "limit": 200}
                print(f"[DEBUG b{batch_idx} r{retry_idx}] bootstrap read: {kwargs}")
            else:
                response = await _call_llm(_SYSTEM_TOOL_SCHEMA, user_prompt, self.llm_method)
                print(f"[DEBUG b{batch_idx} r{retry_idx}] LLM response: {repr(response[:120])}")
                tool_name, kwargs = _parse_tool_call(response)

            if tool_name == "CONTEXT_SUFFICIENT":
                print(f"[DEBUG b{batch_idx} r{retry_idx}] CONTEXT_SUFFICIENT")
                return ("", Status.confidence(0.9))

            if tool_name not in self.ops:
                return (
                    f"[invalid tool: {tool_name}]\n{response}",
                    Status.self_confidence_but_failed(0.1),
                )

            # Execute tool
            result = self.ops[tool_name].forward(worktree_path, **kwargs)

            # Validate tool result
            tool_validator = self.validators.get("validate_tool_result")
            if tool_validator is not None:
                ok, msg = tool_validator.validate(result)
                if not ok:
                    return (
                        f"[{tool_name}({kwargs})]\nERROR: {msg}\n",
                        Status.self_confidence_but_failed(0.2),
                    )

            trace = f"[{tool_name}({kwargs})]\n{result}"
            print(f"[DEBUG b{batch_idx} r{retry_idx}] tool trace len={len(trace)}")

            # Context validator: check if we have the target file content
            context_validator = self._context_validator(worktree_path, task)
            ok, cv_msg = context_validator.validate(prompt + "\n" + trace)
            print(f"[DEBUG b{batch_idx} r{retry_idx}] context_validator: ok={ok} msg={cv_msg}")
            if ok:
                print(f"[DEBUG b{batch_idx} r{retry_idx}] context sufficient -> confidence")
                return (trace, Status.confidence(0.9))

            return (trace, Status.self_confidence_but_failed(0.5))

        return code_context_gather

    def _context_validator(self, worktree_path: str, task: dict) -> HarnessValidatorOp:
        """Build a context validator that checks if target file was READ with mask region
        AND at least one other file was read for cross-reference."""
        target_file = task["target_file"]
        mask_line = task.get("mask_line", 0)
        target_path = os.path.join(worktree_path, target_file)
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                target_lines = [line.rstrip() for line in f.readlines()]
        except (OSError, UnicodeDecodeError):
            target_lines = []

        class ContextValidator(HarnessValidatorOp):
            name = "context_validator"
            description = "Check if accumulated context contains read result for target file around mask and at least one other file."

            def validate(self, content: str) -> tuple:
                # Must have a read() result for the target file
                if "read(" not in content or target_file not in content:
                    return (False, f"target file {target_file} not yet read")

                # Must contain lines around the mask (mask_line +/- 3)
                start = max(0, mask_line - 3)
                end = min(len(target_lines), mask_line + 4)
                found = 0
                for line in target_lines[start:end]:
                    stripped = line.strip()
                    if len(stripped) > 3 and stripped in content:
                        found += 1
                if found < 2:
                    return (False, f"read result missing mask region lines")

                return (True, "")

            def schema(self) -> str:
                return "context_validator()"

        return ContextValidator()

    def _make_code_gen(self, context_tensor: torch.Tensor, worktree_tensor: torch.Tensor):
        """Build ft_async_get for code generation stage."""

        async def code_gen(coords: List[int], prompt: str) -> Tuple[str, Status]:
            import ast

            batch_idx = coords[0]
            retry_idx = coords[1]

            context = _read_element(context_tensor, batch_idx)
            worktree_path = _read_element(worktree_tensor, batch_idx)

            # Read task metadata
            task_json_path = os.path.join(worktree_path, ".cloze_task.json")
            with open(task_json_path) as f:
                task = json.load(f)

            # Build generation prompt
            mask_start = task.get('mask_start_1idx')
            mask_end = task.get('mask_end_1idx')
            mask_line = task.get('mask_line', 0)
            if mask_start is not None and mask_end is not None:
                num_lines = mask_end - mask_start + 1
                range_hint = (
                    f"The mask originally covered lines {mask_start}-{mask_end} "
                    f"({num_lines} lines). "
                )
            else:
                range_hint = ""
            user_prompt = (
                f"Task: Recover the masked code in {task['target_file']} "
                f"(line {mask_line}).\n"
                f"{range_hint}"
                f"The mask is marked with <AUTOENCODER-CLOZE-MASK-PLACEHOLDER> in the context below. "
                f"Output EXACTLY the {num_lines if range_hint else 'original multiple'} lines of code that were removed. "
                f"Do NOT output more lines or fewer lines. "
                f"Do NOT output anything before or after the replacement code. "
                f"Preserve original indentation exactly.\n\n"
                f"Gathered context:\n{context}\n\n"
            )
            if retry_idx > 0:
                user_prompt += (
                    f"Previous attempt was wrong. "
                    f"Retry {retry_idx}. Output ONLY the exact original lines, nothing else.\n"
                )

            response = await _call_llm(_SYSTEM_GENERATION, user_prompt, self.llm_method)
            print(f"[DEBUG gen b{batch_idx} r{retry_idx}] response: {repr(response[:120])}")

            # Strip markdown fences if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned[cleaned.find("\n") + 1:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:cleaned.rfind("```")]
            cleaned = cleaned.strip()

            # Empty check
            if not cleaned:
                return (response, Status.self_confidence_but_failed(0.1))

            # In-context syntax validation: replace placeholder in the actual file and parse
            target_file = task["target_file"]
            masked_file_path = os.path.join(worktree_path, target_file)
            try:
                with open(masked_file_path, "r", encoding="utf-8") as f:
                    file_content = f.read()
            except (OSError, UnicodeDecodeError) as e:
                return (response, Status.self_confidence_but_failed(0.2))

            reconstructed = file_content.replace("<AUTOENCODER-CLOZE-MASK-PLACEHOLDER>", cleaned)
            try:
                ast.parse(reconstructed)
            except SyntaxError as e:
                # Try multiple indentation levels (0, 2, 4, 6, 8, 10, 12 spaces)
                # The masked region may need different indentation than the line before it.
                for indent_spaces in [0, 2, 4, 6, 8, 10, 12]:
                    indent = " " * indent_spaces
                    fixed_lines = []
                    for line in cleaned.splitlines():
                        if line.strip():
                            fixed_lines.append(indent + line)
                        else:
                            fixed_lines.append(line)
                    fixed = "\n".join(fixed_lines)
                    fixed_reconstructed = file_content.replace("<AUTOENCODER-CLOZE-MASK-PLACEHOLDER>", fixed)
                    try:
                        ast.parse(fixed_reconstructed)
                        print(f"[DEBUG gen b{batch_idx} r{retry_idx}] fixed indentation ({indent_spaces} spaces) and passed syntax check")
                        return (fixed, Status.confidence(0.9))
                    except SyntaxError:
                        continue
                print(f"[DEBUG gen b{batch_idx} r{retry_idx}] in-context syntax error: {e.msg} at line {e.lineno}")
                return (response, Status.self_confidence_but_failed(0.4))

            print(f"[DEBUG gen b{batch_idx} r{retry_idx}] in-context syntax OK -> confidence")
            return (cleaned, Status.confidence(0.9))

        return code_gen
