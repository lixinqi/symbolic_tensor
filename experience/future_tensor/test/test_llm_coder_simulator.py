"""
test_llm_coder_simulator: LLM-powered coder simulator using ft_expert + ft_recurrent.

Architecture (all composed at pipeline level, no ops inside ft_async_get):

  workspace_ft[1]             — tmux instance ID (broadcasts to [1, max_iters])
  decision_input[1, 30]      — builds context per iteration, writes to disk
  decision_output[1, 30]     = ft_expert(decision_input, experience)
  action_condition[1, 30]    — returns "text"|"ctrl" per iteration
  payload_slot[1, 30]        — returns payload per iteration
  send_text_op[1, 30]        = ft_tmux_send_text(payload_slot, ft_expand(workspace_ft, [1, 30]))
  send_ctrl_op[1, 30]        = ft_tmux_send_ctrl(payload_slot, ft_expand(workspace_ft, [1, 30]))
  switched_send[1, 30]       = ft_switch(action_condition, [("text",send_text_op),("ctrl",send_ctrl_op)])
  capture_op[1, 30]          = ft_tmux_capture_pane(ft_expand(workspace_ft, [1, 30]))
  validator[30]              = ft_coder_validator(decision_input, decision_output, switched_send, capture_op)
  output[1]                  = ft_recurrent(validator[30])

  ft_recurrent calls validator.ft_async_get([i], prompt).
  Validator calls child ops at [0, i] — each iteration has its own coordinate slot.
  No shared mutable state — coordination is through tensor coordinates.

Only Chinese prompts allowed for ft_forward (anti-hack).
No CLI commands exposed in prompts — LLM decides everything.
"""

import os
import sys
import subprocess
import tempfile

import sympy

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
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor as st_make_tensor


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


# ─── Termination check ───

def check_terminator_last_line(captured_text: str) -> bool:
    """Check if terminal shows command completion: fresh EMPTY prompt on last line.

    Returns True only when the prompt is empty (no command typed after it).
    Prompt format: (base) λ <hostname> <path> [<command>...]
    """
    lines = [l for l in captured_text.split("\n") if l.strip()]
    if len(lines) < 3:
        return False
    last_line = lines[-1]
    if "λ" in last_line:
        after_lambda = last_line.split("λ", 1)[1].strip()
        parts = after_lambda.split()
        # Find path token (starts with / or ~), then check if anything follows
        path_idx = -1
        for idx, p in enumerate(parts):
            if p.startswith("/") or p.startswith("~"):
                path_idx = idx
                break
        if path_idx == -1:
            # No path found — unusual prompt, not termination
            return False
        # Anything after the path is a typed command
        return len(parts) == path_idx + 1
    if "$ " in last_line:
        after_dollar = last_line.split("$ ", 1)[1].strip()
        return after_dollar == ""
    if last_line.rstrip().endswith("$"):
        return True
    return False


# ─── Experience ───

def make_experience(tmpdir: str):
    """Experience for terminal decisions — concrete task→command examples."""
    return st_make_tensor([
        ["任务\n列出当前目录文件\n终端\n命令行为空",
         "任务是列出文件，对应的shell命令是ls",
         "text:ls"],
        ["任务\n在终端中打印一段文字\n终端\n命令行为空",
         "任务是打印文字，对应的shell命令是echo加内容",
         "text:echo hello"],
        ["任务\n执行命令\n终端\n命令行已有内容",
         "命令行已有内容，只需按回车执行",
         "ctrl:Enter"],
    ], tmpdir)


# ─── Helpers ───

def get_current_cmdline_from_text(pane_text: str) -> str:
    """Extract what's currently typed on the command line (after last prompt)."""
    if not pane_text:
        return ""
    lines = [l for l in pane_text.split("\n") if l.strip()]
    if not lines:
        return ""
    last_line = lines[-1]
    if "λ" in last_line:
        after_lambda = last_line.split("λ", 1)[1].strip()
        parts = after_lambda.split()
        path_ended = False
        result_parts = []
        for p in parts:
            if path_ended:
                result_parts.append(p)
            elif p.startswith("/") or p.startswith("~"):
                path_ended = True
        return " ".join(result_parts)
    elif "$ " in last_line:
        return last_line.split("$ ", 1)[1].strip()
    return ""


# ─── Parse ───

def _parse_decision(raw: str) -> str:
    """Extract first text:/ctrl: line from LLM output."""
    if not raw:
        return ""
    for line in raw.strip().split("\n"):
        line = line.strip()
        if line.startswith("text:") or line.startswith("ctrl:"):
            return line
    for line in raw.strip().split("\n"):
        line = line.strip()
        if line:
            return line
    return ""


def parse_action(decision_result: str):
    """Parse decision string into (action_type, payload)."""
    if not decision_result:
        return ("text", " ")
    if ":" in decision_result:
        parts = decision_result.split(":", 1)
        kind = parts[0].strip().lower()
        payload = parts[1].strip()
        if kind == "ctrl":
            return ("ctrl", payload if payload else "Enter")
        else:
            return ("text", payload if payload else " ")
    if decision_result.lower() in ("enter", "tab", "escape", "backspace"):
        return ("ctrl", decision_result.capitalize())
    return ("text", decision_result if decision_result else " ")


# ─── Pipeline builders ───

def make_decision_input(tmpdir: str, capture_op: FutureTensor, max_iters: int) -> FutureTensor:
    """FutureTensor[1, max_iters]: builds context per iteration, writes to disk.

    ft_capacity_shape: [1, max_iters]
    Captures the live pane via capture_op to show current terminal state to LLM.
    Write-through: materializes to disk so ft_expert can read it.
    """
    ft = FutureTensor(tmpdir, None, [sympy.Integer(1), sympy.Integer(max_iters)])
    ft.ft_capacity_shape = [1, max_iters]

    async def _get(coords, prompt):
        # Capture current terminal state
        pane_text, _ = await capture_op.ft_async_get(coords, prompt)
        cmdline = get_current_cmdline_from_text(pane_text)

        context = (
            f"任务目标：{prompt}\n"
            f"当前终端内容：\n{pane_text}\n"
            f"当前命令行已输入的内容：「{cmdline}」\n"
            f"你是一个终端用户，只能通过text:和ctrl:两种方式操作终端。\n"
            f"text:后面必须是一个合法的shell命令（如echo、ls、cat等），绝对不能是任务描述。\n"
        )
        if cmdline:
            context += f"命令行已经有内容「{cmdline}」，不要重复输入！直接回答：ctrl:Enter\n"
        else:
            context += f"命令行为空。根据任务目标，输入对应的shell命令。只回答一行，格式：text:shell命令\n"

        # Write-through to disk so ft_expert can read it
        flat_idx = sum(c * s for c, s in zip(coords, ft.ft_static_tensor.stride()))
        digits = list(str(flat_idx))
        path = os.path.join(
            ft.ft_static_tensor.st_relative_to, ft.ft_static_tensor.st_tensor_uid,
            "storage", os.path.join(*digits), "data",
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(context)
        ft.ft_static_tensor.data[tuple(coords)] = 1.0

        return (context, Status.confidence(1.0))

    ft.ft_async_get = _get
    return ft


def make_parsed_decision(
    decision_input: FutureTensor,
    decision_output: FutureTensor,
    capture_op: FutureTensor,
    max_iters: int,
) -> FutureTensor:
    """FutureTensor[1, max_iters]: calls decision_input + decision_output, parses, caches.

    ft_capacity_shape: [1, max_iters]
    Returns parsed string "text:payload" or "ctrl:payload" per iteration.
    Caches results so action_condition and payload_slot can read without re-calling LLM.
    """
    relative_to = decision_input.ft_static_tensor.st_relative_to
    cache = {}  # cache[i] = (action_type, payload)

    async def _get(coords, prompt):
        # ft_switch passes dict prompt; extract string
        if isinstance(prompt, dict):
            prompt = prompt.get("prompt", "")
        i = coords[-1]
        if i in cache:
            action_type, payload = cache[i]
            return (f"{action_type}:{payload}", Status.confidence(1.0))

        # Materialize decision_input (writes to disk for ft_expert)
        await decision_input.ft_async_get(coords, prompt)

        # Call ft_expert
        raw_result, _ = await decision_output.ft_async_get(coords, prompt)
        decision_result = _parse_decision(raw_result)
        action_type, payload = parse_action(decision_result)
        cache[i] = (action_type, payload)
        print(f"  [step {i}] {action_type}:{repr(payload)}")
        return (f"{action_type}:{payload}", Status.confidence(1.0))

    ft = FutureTensor(relative_to, _get, [sympy.Integer(1), sympy.Integer(max_iters)])
    ft.ft_capacity_shape = [1, max_iters]
    return ft


def make_action_condition(parsed_decision: FutureTensor, max_iters: int) -> FutureTensor:
    """FutureTensor[1, max_iters]: pulls parsed_decision, returns "text"|"ctrl".

    ft_capacity_shape: [1, max_iters]
    """
    relative_to = parsed_decision.ft_static_tensor.st_relative_to

    async def _get(coords, prompt):
        result, _ = await parsed_decision.ft_async_get(coords, prompt)
        action_type = result.split(":", 1)[0] if ":" in result else "text"
        return (action_type, Status.confidence(1.0))

    ft = FutureTensor(relative_to, _get, [sympy.Integer(1), sympy.Integer(max_iters)])
    ft.ft_capacity_shape = [1, max_iters]
    return ft


def make_payload_slot(parsed_decision: FutureTensor, max_iters: int) -> FutureTensor:
    """FutureTensor[1, max_iters]: pulls parsed_decision, returns payload.

    ft_capacity_shape: [1, max_iters]
    """
    relative_to = parsed_decision.ft_static_tensor.st_relative_to

    async def _get(coords, prompt):
        result, _ = await parsed_decision.ft_async_get(coords, prompt)
        payload = result.split(":", 1)[1] if ":" in result else ""
        return (payload if payload else " ", Status.confidence(1.0))

    ft = FutureTensor(relative_to, _get, [sympy.Integer(1), sympy.Integer(max_iters)])
    ft.ft_capacity_shape = [1, max_iters]
    return ft


# ─── ft_coder_validator: termination check ───

def ft_coder_validator(
    iteration_body: FutureTensor,
    max_iters: int = 30,
) -> FutureTensor:
    """Validator: run iteration body, check if terminal shows command completion.

    ft_capacity_shape: [1, max_iters]

    At coords [0, i]: pull iteration_body → check_terminator_last_line on result.
    """
    relative_to = iteration_body.ft_static_tensor.st_relative_to

    async def _validator_async_get(coords, prompt):
        i = coords[-1]
        captured_text, _ = await iteration_body.ft_async_get(coords, prompt)

        if i >= 1 and check_terminator_last_line(captured_text):
            return (captured_text, Status.confidence(1.0))
        else:
            return ("", Status.self_confidence_but_failed(0.9))

    validator = FutureTensor(relative_to, _validator_async_get,
                             [sympy.Integer(1), sympy.Integer(max_iters)])
    validator.ft_capacity_shape = [1, max_iters]
    return validator


# ─── Helpers ───

def read_ft_element(ft, flat_index):
    digits = list(str(flat_index))
    path = os.path.join(
        ft.ft_static_tensor.st_relative_to, ft.ft_static_tensor.st_tensor_uid,
        "storage", os.path.join(*digits), "data",
    )
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return f.read()


# ─── Test ───

print("Running test_llm_coder_simulator...\n")

INSTANCE_ID = "llm_coder_test"
SESSION_NAME = f"{tmux_session_prefix}{INSTANCE_ID}"

with tempfile.TemporaryDirectory() as tmpdir:
    workspace_ft = ft_make_forwarded(tmpdir, [1], [INSTANCE_ID])  # [1]

    # Setup: create session + delay
    setup = ft_sequential(
        ft_tmux_create_session(workspace_ft),  # [1]
        ft_sleep(workspace_ft, 0.5))           # [1]
    setup.ft_forward(st_make_tensor(["启动终端会话"], tmpdir))

    # ── Compose entire pipeline ──

    # 1. Expand workspace_ft[1] → [1, 30] for all ops
    workspace_expanded = ft_expand(workspace_ft, [1, MAX_ITERS])  # [1, 30]

    # 2. capture_op[1, 30]: live pane observation (used by decision + iteration body)
    capture_op = ft_tmux_capture_pane(workspace_expanded)  # [1, 30]

    # 3. decision_input[1, 30]: captures pane, builds context, writes to disk
    decision_input = make_decision_input(tmpdir, capture_op, MAX_ITERS)

    # 4. decision_output[1, 30]: ft_expert reads from disk, asks LLM
    experience = make_experience(tmpdir)
    decision_output = ft_expert(
        decision_input, experience,
        task_prompt="你是终端操作员。根据任务目标生成对应的shell命令。只输出一行：text:shell命令 或 ctrl:键名。text:后面必须是合法shell命令（如echo、ls、cat），绝对不能把任务描述当命令输入。",
        topk=2,
    )

    # 5. parsed_decision[1, 30]: drives decision_input + decision_output, parses, caches
    parsed_decision = make_parsed_decision(
        decision_input, decision_output, capture_op, MAX_ITERS,
    )

    # 6. action_condition[1, 30] + payload_slot[1, 30]: derive from parsed_decision
    action_condition = make_action_condition(parsed_decision, MAX_ITERS)
    payload_slot = make_payload_slot(parsed_decision, MAX_ITERS)

    # 7. send ops[1, 30]: payload_slot[1,30] broadcasts with workspace[1,30]
    send_text_op = ft_tmux_send_text(payload_slot, workspace_expanded)   # [1, 30]
    send_ctrl_op = ft_tmux_send_ctrl(payload_slot, workspace_expanded)   # [1, 30]

    # 8. switched_send[1, 30]: routes based on action_condition
    switched_send = ft_switch(action_condition, [
        ("text", "type", "send text to terminal", send_text_op),
        ("ctrl", "ctrl", "send control key to terminal", send_ctrl_op),
    ])

    # 9. sleep_op[1, 30]
    sleep_op = ft_sleep(workspace_expanded, 0.5)  # [1, 30]

    # 10. iteration_body[1, 30] = ft_sequential(switched_send, sleep, capture)
    iteration_body = ft_sequential(switched_send, sleep_op, capture_op)  # [1, 30]

    # 11. validator[1, 30]: check termination on iteration_body result
    validator = ft_coder_validator(iteration_body, max_iters=MAX_ITERS)
    output = ft_recurrent(validator)  # output[1]

    # ONE ft_forward — the prompt IS the task
    output.ft_forward(st_make_tensor(["在终端中输出问候语hello world"], tmpdir))

    # Verify
    content = read_ft_element(output, 0)
    ok = content is not None and "hello" in content.lower()
    print(f"\n  {'✓' if ok else '✗'} pane contains 'hello'")
    print(f"\n  Pane:\n{content}\n")

if not ok:
    sys.exit(1)
print("test_llm_coder_simulator passed.")
