"""
test_repl_coder_simulator: Online Stage 1.

Harness model composes the static DAG and drives the REPL loop.
ClaudeCodeMock is the mock human: observes terminal, writes experience.
It knows nothing about the pipeline.
"""

import os
import shutil
import sys
import subprocess
import tempfile

import libtmux

import sympy
import torch

from experience.future_tensor.future_tensor import FutureTensor
from experience.future_tensor.status import Status
from experience.future_tensor.function.tmux_session import tmux_session_prefix
from experience.future_tensor.function.ft_make_forwarded import ft_make_forwarded
from experience.future_tensor.function.ft_tmux_create_session import ft_tmux_create_session
from experience.future_tensor.function.ft_tmux_send_text import ft_tmux_send_text
from experience.future_tensor.function.ft_tmux_send_ctrl import ft_tmux_send_ctrl
from experience.future_tensor.function.ft_tmux_capture_pane import ft_tmux_capture_pane
from experience.future_tensor.function.ft_sleep import ft_sleep
from experience.future_tensor.function.ft_sequential import ft_sequential
from experience.future_tensor.function.ft_recurrent import ft_recurrent
from experience.future_tensor.function.ft_expert import ft_expert
from experience.future_tensor.function.ft_switch import ft_switch
from experience.future_tensor.function.ft_expand import ft_expand
from experience.future_tensor.function.ft_mean import ft_mean
from experience.future_tensor.function.ft_first_line import ft_first_line
from experience.future_tensor.function.ft_terminal_idle_gate import ft_terminal_idle_gate
from experience.future_tensor.function.ft_validate_ctrl import ft_validate_ctrl
from experience.future_tensor.backward_dispatch import (
    dispatch_policy,
    need_reflection,
    ft_reflection_starter,
    TracePolicy,
)
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor as st_make_tensor
from experience.symbolic_tensor.tensor_util.st_setitem import st_setitem


# ─── Source LLM env ───

result = subprocess.run(
    ["bash", "-c", "source ~/.anthropic.sh && env"],
    capture_output=True, text=True,
)
for line in result.stdout.splitlines():
    if "=" in line:
        key, _, val = line.partition("=")
        os.environ[key] = val
os.environ.pop("CLAUDECODE", None)


MAX_ITERS = 30
N_EXPERIENCE = 8


# ─── ClaudeCodeMock: the mock human ───

class ClaudeCodeMock:
    """Mock human operator. Observes a tmux session, writes experience.

    Knows nothing about the pipeline, tensors, autograd, or ft_forward.
    Only provides observe_and_update() to be called from the harness loop.

    NEVER interacts with the terminal directly — only observes and writes
    experience entries. All terminal actions come through the pipeline.
    """

    def __init__(self, session_name, decision_exp, cmd_exp, n_experience=8):
        self.session_name = session_name
        self.decision_exp = decision_exp
        self.cmd_exp = cmd_exp
        self.n_experience = n_experience
        self._di = 0
        self._ci = 0

        server = libtmux.Server()
        self._pane = None
        for s in server.sessions:
            if s.session_name == session_name:
                self._pane = s.active_window.active_pane
                break

    def observe_and_update(self):
        """Look at the terminal. Write experience based on what we see.

        Only writes experience entries — NEVER sends keys or interacts
        with the terminal directly.
        """
        captured = self._pane.capture_pane()
        lines = [l for l in captured if l.strip()]
        if not lines:
            return
        last_line = lines[-1]
        has_prompt, has_cmd = self._parse(last_line)

        if has_cmd:
            # Reinforce ctrl heavily: write 3 entries per observation so
            # retrieval (topk=2) overwhelmingly selects ctrl examples.
            for _ in range(min(3, self.n_experience - self._di)):
                self._teach_decision(last_line, "命令已输入\n提示符后有命令", "ctrl:按回车执行")
            for _ in range(min(3, self.n_experience - self._ci)):
                self._teach_cmd("ctrl:按回车执行", "ctrl\nEnter", "Enter")
        elif has_prompt:
            self._teach_decision(last_line, "空提示符\n没有命令", "text:输入shell命令")
            self._teach_cmd("text:输入shell命令", "text\nshell命令", "echo hello world")

        print(f"    cc: prompt={has_prompt} cmd={has_cmd} | {last_line}")

    def _parse(self, line):
        if "λ" in line:
            after = line.split("λ", 1)[1].strip()
            parts = after.split()
            pi = next((i for i, p in enumerate(parts) if p.startswith(("/", "~"))), -1)
            if pi == -1:
                return True, len(parts) > 0
            return True, len(parts) > pi + 1
        if "$ " in line:
            return True, line.split("$ ", 1)[1].strip() != ""
        if line.rstrip().endswith("$"):
            return True, False
        return False, False

    def _teach_decision(self, q, k, v):
        if self._di < self.n_experience:
            self._write(self.decision_exp, self._di, q, k, v)
            self._di += 1

    def _teach_cmd(self, q, k, v):
        if self._ci < self.n_experience:
            self._write(self.cmd_exp, self._ci, q, k, v)
            self._ci += 1

    def _write(self, tensor, idx, q, k, v):
        st_setitem(tensor, [idx, 0], q)
        st_setitem(tensor, [idx, 1], k)
        st_setitem(tensor, [idx, 2], v)
        cache = os.path.join(tensor.st_relative_to, tensor.st_tensor_uid, "qkv_data_view")
        if os.path.isdir(cache):
            shutil.rmtree(cache)


# ─── Terminal parsing (for validator) ───

def parse_terminal(text):
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return False, False
    last = lines[-1]
    if "λ" in last:
        after = last.split("λ", 1)[1].strip()
        parts = after.split()
        pi = next((i for i, p in enumerate(parts) if p.startswith(("/", "~"))), -1)
        if pi == -1:
            return True, len(parts) > 0
        return True, len(parts) > pi + 1
    if "$ " in last:
        return True, last.split("$ ", 1)[1].strip() != ""
    if last.rstrip().endswith("$"):
        return True, False
    return False, False


# ─── Validator ───

def ft_coder_validator(body, max_iters=30):
    relative_to = body.ft_static_tensor.st_relative_to

    async def _get(coords, prompt):
        i = coords[-1]
        text, _ = await body.ft_async_get(coords, prompt)
        has_prompt, has_cmd = parse_terminal(text)
        lines = [l for l in text.split("\n") if l.strip()]
        # Check that "hello" appears in a non-prompt output line (not garbage in prompt)
        output_lines = [l for l in lines if "λ" not in l and "$ " not in l and not l.rstrip().endswith("$")]
        has_hello_output = any("hello" in l.lower() for l in output_lines)
        if i >= 1 and has_prompt and not has_cmd and len(lines) >= 3 and has_hello_output:
            return (text, Status.confidence(1.0))
        return ("", Status.self_confidence_but_failed(0.9))

    v = FutureTensor(relative_to, _get,
                     [sympy.Integer(1), sympy.Integer(max_iters)])
    v.ft_capacity_shape = [1, max_iters]
    v.requires_grad_(True)
    return v


def read_ft_element(ft):
    path = os.path.join(
        ft.ft_static_tensor.st_relative_to, ft.ft_static_tensor.st_tensor_uid,
        "storage", "0", "data",
    )
    return open(path).read() if os.path.isfile(path) else None


# ─── Inline output_prompt: reads files, embeds content in prompt text ───
# raw_llm_api cannot read file paths from the default prompt.
# This callable reads workspace files and puts content inline so the LLM
# actually sees the experience entries and input text.

def _read_exp_entries(exp_view_dir):
    """Read experience QKV entries from disk. Returns list of (query, key, value) tuples."""
    if not os.path.isdir(exp_view_dir):
        return []
    # Experience layout: <coord>/0/data.txt=query, <coord>/1/data.txt=key, <coord>/2/data.txt=value
    entries = {}
    for root, dirs, files in os.walk(exp_view_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, exp_view_dir)
            parts = rel.replace("\\", "/").split("/")
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read().strip()
            except Exception:
                content = ""
            # parts like ["0", "0", "data.txt"] → coord="0", qkv_idx=0
            if len(parts) >= 3:
                coord = parts[0]
                qkv_idx = int(parts[1])
                if coord not in entries:
                    entries[coord] = ["", "", ""]
                entries[coord][qkv_idx] = content
    return [(e[0], e[1], e[2]) for e in entries.values() if any(e)]


def _read_input_text(input_view_dir):
    """Read input text from the workspace."""
    if not os.path.isdir(input_view_dir):
        return ""
    for root, dirs, files in os.walk(input_view_dir):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception:
                pass
    return ""


def inline_output_prompt(task_prompt, workspace_dir, exp_view_dir, input_view_dir, output_dir):
    """Build LLM prompt with experience and input content inline."""
    entries = _read_exp_entries(exp_view_dir)
    input_text = _read_input_text(input_view_dir)

    # Build experience examples: show query → value (input-like → output)
    if entries:
        examples = []
        for q, k, v in entries:
            if v:
                # Show the query (which resembles the input) → value (the target output)
                label = q.replace("\n", " ") if q else k.replace("\n", " ")
                examples.append(f"  INPUT: {label}\n  OUTPUT: {v}")
        exp_text = "\n\n".join(examples) if examples else "(no examples)"
    else:
        exp_text = "(no examples)"

    return (
        f"{task_prompt}\n\n"
        f"Examples of correct input→output:\n{exp_text}\n\n"
        f"Now for this input:\n{input_text}\n\n"
        "Reply with ONLY the output. One line. No explanation. No prefix."
    )


# ─── Harness model + REPL loop ───

print("Running test_repl_coder_simulator (online Stage 1)...\n")

INSTANCE_ID = "repl_coder_test"
SESSION_NAME = f"{tmux_session_prefix}{INSTANCE_ID}"

with tempfile.TemporaryDirectory() as tmpdir:
    # Workspace
    workspace = ft_make_forwarded(tmpdir, [1], [INSTANCE_ID])
    setup = ft_sequential(ft_tmux_create_session(workspace), ft_sleep(workspace, 0.5))
    setup.ft_forward(st_make_tensor(["启动终端会话"], tmpdir))

    expanded = ft_expand(workspace, [1, MAX_ITERS])
    reflection_starter = ft_reflection_starter()
    capture = need_reflection(ft_tmux_capture_pane(expanded), reflection_starter)

    # Experience (TODO — cold-start)
    decision_exp = st_make_tensor([["", "", ""]] * N_EXPERIENCE, tmpdir)
    cmd_exp = st_make_tensor([["", "", ""]] * N_EXPERIENCE, tmpdir)

    # Expert chain → filter → gate → switch → body → validator → recurrent
    decision_raw = ft_expert(capture, decision_exp,
        output_prompt=inline_output_prompt,
        task_prompt="观察终端最后一行。提示符后只有路径→text:输入shell命令。提示符后有命令→ctrl:按回车执行。只输出一行。",
        topk=2, skip_query_gen=True)
    decision = ft_first_line(decision_raw)
    cmd = ft_expert(decision, cmd_exp,
        output_prompt=inline_output_prompt,
        task_prompt="text:开头→输出shell命令。ctrl:开头→输出Enter。只输出一行。",
        topk=2, skip_query_gen=True)
    cmd_clean = ft_first_line(cmd)
    cmd_gated = ft_terminal_idle_gate(cmd_clean, expanded)
    cmd_ctrl = ft_validate_ctrl(cmd_clean)
    switched = ft_switch(decision, [
        ("text", "type", "send text", ft_tmux_send_text(cmd_gated, expanded)),
        ("ctrl", "ctrl", "send ctrl", ft_tmux_send_ctrl(cmd_ctrl, expanded)),
    ])
    body = ft_sequential(switched, ft_sleep(expanded, 0.5), capture)
    validator = ft_coder_validator(body, max_iters=MAX_ITERS)

    output = ft_recurrent(validator, step_budget=1)
    prompt = st_make_tensor(["在终端中输出问候语hello world"], tmpdir)

    # Stage 1 operator (knows nothing about the pipeline above)
    cc = ClaudeCodeMock(SESSION_NAME, decision_exp, cmd_exp, N_EXPERIENCE)

    # REPL loop
    for step in range(MAX_ITERS):
        output.ft_forward(prompt)
        if output.ft_forwarded:
            print(f"  step {step}: completed")
            break

        loss = ft_mean(output)
        records = []
        with dispatch_policy(TracePolicy(records)):
            loss.backward(create_graph=True, retain_graph=True)
        print(f"  step {step}: trace={len(records)}")
        records.clear()

        cc.observe_and_update()

    # Verify — require "hello" in command output, not just anywhere in pane
    content = read_ft_element(output)
    if content:
        content_lines = [l for l in content.split("\n") if l.strip()]
        output_lines = [l for l in content_lines if "λ" not in l and "$ " not in l and not l.rstrip().endswith("$")]
        ok = any("hello" in l.lower() for l in output_lines)
    else:
        ok = False
    print(f"\n  {'✓' if ok else '✗'} output lines contain 'hello' (exp: d={cc._di} c={cc._ci})")
    if content:
        print(f"  {content.strip()}")

if not ok:
    sys.exit(1)
print("\ntest_repl_coder_simulator passed.")
