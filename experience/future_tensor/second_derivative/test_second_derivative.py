"""
Tests for experience.future_tensor.second_derivative.

Tests are grouped by scenario:
  1. Module imports and structure
  2. need_2nd_derivative: requires_grad behaviour
  3. dispatch_policy context manager + PolicyConflictError
  4. TracePolicy (default): records collected without LLM
  5. Custom policy: selective dispatch
  6. recurrent_2nd_backward: TracePolicy collects correct inputs
  7. moe_2nd_backward: TracePolicy collects correct inputs
  8. Integration: recurrent_backward dispatches into active TracePolicy
  9. Integration: st_moe_backward dispatches into active TracePolicy
 10. ReflectionRecord fields

Run:
    python -m experience.future_tensor.second_derivative.test_second_derivative
"""

import os
import sys
import tempfile
import time

import torch

passed = 0
failed = 0


def run_test(name: str, condition: bool, expected=None, actual=None):
    global passed, failed
    if condition:
        passed += 1
        print(f"  \u2713 {name}")
    else:
        failed += 1
        print(f"  \u2717 {name}")
        if expected is not None:
            print(f"    expected: {expected}")
            print(f"    actual:   {actual}")


# ── Group 1: Module imports ──────────────────────────────────────────────────
print("Group 1: Module imports")

from experience.future_tensor.second_derivative import (
    need_2nd_derivative,
    get_2nd_dispatcher,
    dispatch_policy,
    TracePolicy,
    Policy,
    ReflectionRecord,
    PolicyConflictError,
)
from experience.future_tensor.second_derivative.function.recurrent_2nd import (
    recurrent_2nd_backward,
)
from experience.future_tensor.second_derivative.function.moe_2nd import (
    moe_2nd_backward,
)

run_test("need_2nd_derivative callable", callable(need_2nd_derivative))
run_test("get_2nd_dispatcher callable", callable(get_2nd_dispatcher))
run_test("dispatch_policy callable", callable(dispatch_policy))
run_test("TracePolicy is Policy subclass", issubclass(TracePolicy, Policy))
run_test("ReflectionRecord importable", ReflectionRecord is not None)
run_test("PolicyConflictError is RuntimeError subclass",
         issubclass(PolicyConflictError, RuntimeError))
run_test("recurrent_2nd_backward callable", callable(recurrent_2nd_backward))
run_test("moe_2nd_backward callable", callable(moe_2nd_backward))

# ── Group 2: need_2nd_derivative ─────────────────────────────────────────────
print("\nGroup 2: need_2nd_derivative")

anchor = torch.nn.Parameter(torch.ones(()))

t = torch.zeros(3)
result = need_2nd_derivative(t, anchor)
run_test("returns same tensor object", result is t)
run_test("requires_grad set to True", result.requires_grad is True)

t2 = torch.ones(2, 3)
result2 = need_2nd_derivative(t2, anchor)
run_test("works on 2D tensor", result2 is t2 and result2.requires_grad)

# Already requires_grad — idempotent
t3 = torch.zeros(4, requires_grad=True)
result3 = need_2nd_derivative(t3, anchor)
run_test("idempotent when already requires_grad", result3.requires_grad is True)

# ── Group 3: dispatch_policy context manager ─────────────────────────────────
print("\nGroup 3: dispatch_policy context manager")

collector = []
with dispatch_policy(TracePolicy(collector)):
    # Active inside block — verify by dispatching directly
    from experience.future_tensor.function.ft_recurrent_backward import recurrent_backward
    d = get_2nd_dispatcher(recurrent_backward)
    out = d({"x": 1})

run_test("dispatch inside block appends record", len(collector) == 1)
run_test("record fn matches", collector[0].fn is recurrent_backward)
run_test("record inputs match", collector[0].inputs == {"x": 1})
run_test("record output is scalar tensor", collector[0].output.shape == torch.Size([]))

# PolicyConflictError on nesting
try:
    with dispatch_policy(TracePolicy([])):
        with dispatch_policy(TracePolicy([])):
            pass
    run_test("nested dispatch_policy raises PolicyConflictError", False)
except PolicyConflictError:
    run_test("nested dispatch_policy raises PolicyConflictError", True)

# Policy clears after block
collector2 = []
with dispatch_policy(TracePolicy(collector2)):
    pass
# dispatch outside block goes to default collector, not collector2
from experience.future_tensor.second_derivative.context import _default_collector
pre_len = len(_default_collector)
d2 = get_2nd_dispatcher(recurrent_backward)
d2({"outside": True})
run_test("outside block goes to module default (not collector2)",
         len(collector2) == 0 and len(_default_collector) > pre_len)

# ── Group 4: TracePolicy default ─────────────────────────────────────────────
print("\nGroup 4: TracePolicy — default, non-destructive")

from experience.symbolic_tensor.function.st_moe_backward import st_moe_backward

coll = []
tp = TracePolicy(coll)

dummy_inputs = {"grad_output": torch.zeros(2), "input": torch.ones(2)}

out1 = tp.dispatch(recurrent_backward, dummy_inputs)
out2 = tp.dispatch(st_moe_backward, {"a": 1, "b": 2})

run_test("TracePolicy dispatch returns scalar 1", out1.item() == 1.0)
run_test("TracePolicy appends two records", len(coll) == 2)
run_test("record 0 fn is recurrent_backward", coll[0].fn is recurrent_backward)
run_test("record 1 fn is st_moe_backward", coll[1].fn is st_moe_backward)
run_test("record 0 inputs preserved", coll[0].inputs is dummy_inputs)
run_test("record timestamp is float", isinstance(coll[0].timestamp, float))
run_test("record timestamp recent", abs(coll[0].timestamp - time.monotonic()) < 5.0)

# ── Group 5: Custom policy ───────────────────────────────────────────────────
print("\nGroup 5: Custom policy")

class SelectivePolicy(Policy):
    """Only dispatches recurrent_backward; traces the rest."""
    def __init__(self):
        self.recurrent_called = []
        self.other_trace = []

    def dispatch(self, fn, arg_name2inputs):
        if fn is recurrent_backward:
            self.recurrent_called.append(arg_name2inputs)
        else:
            self.other_trace.append((fn, arg_name2inputs))
        return torch.ones(())

sp = SelectivePolicy()
with dispatch_policy(sp):
    get_2nd_dispatcher(recurrent_backward)({"r": 1})
    get_2nd_dispatcher(st_moe_backward)({"m": 2})

run_test("SelectivePolicy: recurrent_called has 1 entry", len(sp.recurrent_called) == 1)
run_test("SelectivePolicy: other_trace has 1 entry", len(sp.other_trace) == 1)
run_test("SelectivePolicy: recurrent inputs correct", sp.recurrent_called[0] == {"r": 1})
run_test("SelectivePolicy: other fn is st_moe_backward", sp.other_trace[0][0] is st_moe_backward)

# ── Group 6: recurrent_2nd_backward ──────────────────────────────────────────
print("\nGroup 6: recurrent_2nd_backward")

with tempfile.TemporaryDirectory() as tmpdir:
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
    from experience.future_tensor.status import Status

    grad_out = make_tensor(["diff text"], tmpdir)
    inp = make_tensor(["original"], tmpdir)
    out = make_tensor(["result"], tmpdir)
    pmt = make_tensor(["prompt"], tmpdir)

    coll6 = []
    with dispatch_policy(TracePolicy(coll6)):
        ret = recurrent_2nd_backward(grad_out, inp, out, pmt, task_prompt="test")

    run_test("returns scalar 1", ret.item() == 1.0 and ret.shape == torch.Size([]))
    run_test("one record collected", len(coll6) == 1)
    run_test("fn is recurrent_backward", coll6[0].fn is recurrent_backward)
    run_test("grad_output in inputs", coll6[0].inputs["grad_output"] is grad_out)
    run_test("input in inputs", coll6[0].inputs["input"] is inp)
    run_test("output in inputs", coll6[0].inputs["output"] is out)
    run_test("prompt_tensor in inputs", coll6[0].inputs["prompt_tensor"] is pmt)
    run_test("kwargs forwarded (task_prompt)", coll6[0].inputs.get("task_prompt") == "test")

# ── Group 7: moe_2nd_backward ────────────────────────────────────────────────
print("\nGroup 7: moe_2nd_backward")

with tempfile.TemporaryDirectory() as tmpdir:
    grad_out7 = make_tensor(["moe diff"], tmpdir)
    inp7 = make_tensor(["moe input"], tmpdir)
    out7 = make_tensor(["moe output"], tmpdir)
    exp7 = make_tensor([["q", "k", "v"]], tmpdir)
    idx7 = [[torch.tensor([0])]]

    coll7 = []
    with dispatch_policy(TracePolicy(coll7)):
        ret7 = moe_2nd_backward(grad_out7, inp7, out7, exp7, idx7, context=None, task_prompt="moe")

    run_test("returns scalar 1", ret7.item() == 1.0 and ret7.shape == torch.Size([]))
    run_test("one record collected", len(coll7) == 1)
    run_test("fn is st_moe_backward", coll7[0].fn is st_moe_backward)
    run_test("grad_output in inputs", coll7[0].inputs["grad_output"] is grad_out7)
    run_test("experience in inputs", coll7[0].inputs["experience"] is exp7)
    run_test("indexes in inputs", coll7[0].inputs["selected_experience_qkv_indexes_list"] is idx7)
    run_test("kwargs forwarded (task_prompt)", coll7[0].inputs.get("task_prompt") == "moe")

# ── Group 8: Integration — recurrent_backward dispatches ─────────────────────
print("\nGroup 8: Integration — recurrent_backward → TracePolicy")

# Use kContextOverflow for all elements so numeric channel zeroes all coeffs
# → no AgentTask fires → no LLM API key needed → 2nd derivative dispatch still runs
with tempfile.TemporaryDirectory() as tmpdir:
    from experience.future_tensor.function.ft_recurrent_backward import recurrent_backward

    input8 = make_tensor(["overflow0", "overflow1"], tmpdir)
    input8.data[0] = Status.convert_status_to_float(Status.kContextOverflow)
    input8.data[1] = Status.convert_status_to_float(Status.kContextOverflow)

    output8 = make_tensor(["overflow0"], tmpdir)
    output8.data[0] = Status.convert_status_to_float(Status.kContextOverflow)

    grad_output8 = make_tensor(["improved text"], tmpdir)
    grad_output8.data[0] = 1.0

    prompt8 = make_tensor(["translate", "unused"], tmpdir)
    prompt8.data.fill_(1.0)

    coll8 = []
    with dispatch_policy(TracePolicy(coll8)):
        gi8 = recurrent_backward(
            grad_output8, input8, output8, prompt8,
            task_prompt="no llm needed",
            llm_method="raw_llm_api",
        )

    run_test("recurrent_backward dispatched 2nd derivative", len(coll8) == 1)
    run_test("record fn is recurrent_backward", coll8[0].fn is recurrent_backward)
    run_test("grad_output passed through", coll8[0].inputs["grad_output"] is grad_output8)
    run_test("input passed through", coll8[0].inputs["input"] is input8)
    run_test("output passed through", coll8[0].inputs["output"] is output8)
    run_test("prompt_tensor passed through", coll8[0].inputs["prompt_tensor"] is prompt8)
    run_test("task_prompt forwarded", coll8[0].inputs.get("task_prompt") == "no llm needed")

# ── Group 9: Integration — st_moe_backward dispatches ────────────────────────
print("\nGroup 9: Integration — st_moe_backward → TracePolicy")

# st_moe_backward always fires an LLM task for grad_experience (padding logic).
# Load API credentials so it can run.
import subprocess as _sp
_env_result = _sp.run(
    ["bash", "-c", "source ~/.anthropic.sh && env"],
    capture_output=True, text=True,
)
for _line in _env_result.stdout.splitlines():
    if "=" in _line:
        _k, _, _v = _line.partition("=")
        os.environ[_k] = _v
os.environ.pop("CLAUDECODE", None)

with tempfile.TemporaryDirectory() as tmpdir:
    from experience.symbolic_tensor.function.st_moe_backward import st_moe_backward

    inp9 = make_tensor(["hello world"], tmpdir)
    inp9.data[0] = Status.convert_status_to_float(Status.confidence(0.9))
    inp9.requires_grad_(True)

    out9 = make_tensor(["Bonjour le monde"], tmpdir)
    out9.data[0] = Status.convert_status_to_float(Status.confidence(0.9))

    exp9 = make_tensor([
        ["greeting\nhello", "Hello in English", "Bonjour en francais"],
        ["farewell\nbye",   "Goodbye in English", "Au revoir en francais"],
    ], tmpdir)
    exp9.data.fill_(1.0)

    grad_out9 = make_tensor(["better translation"], tmpdir)
    grad_out9.data[0] = 1.0

    # For experience shape [2, 3]: leaf = [dim0_indices, dim1_qkv_indices].
    # Select row 0 only: dim0=[0], dim1=[0] (qkv col placeholder, replaced by slice).
    idx9 = [[torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long)]]

    coll9 = []
    with dispatch_policy(TracePolicy(coll9)):
        gi9, ge9 = st_moe_backward(
            grad_out9, inp9, out9, exp9, idx9,
            task_prompt="Translate English to French.",
            llm_method="raw_llm_api",
        )

    run_test("st_moe_backward dispatched 2nd derivative", len(coll9) == 1)
    run_test("record fn is st_moe_backward", coll9[0].fn is st_moe_backward)
    run_test("grad_output in inputs", coll9[0].inputs["grad_output"] is grad_out9)
    run_test("experience in inputs", coll9[0].inputs["experience"] is exp9)
    run_test("task_prompt forwarded", coll9[0].inputs.get("task_prompt") == "Translate English to French.")

# ── Group 10: ReflectionRecord fields ────────────────────────────────────────
print("\nGroup 10: ReflectionRecord fields")

rec = ReflectionRecord(
    fn=recurrent_backward,
    inputs={"a": 1},
    output=torch.ones(()),
)
run_test("fn stored", rec.fn is recurrent_backward)
run_test("inputs stored", rec.inputs == {"a": 1})
run_test("output stored", rec.output.item() == 1.0)
run_test("timestamp auto-set", isinstance(rec.timestamp, float) and rec.timestamp > 0)
run_test("fn.__name__ accessible", rec.fn.__name__ == "recurrent_backward")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n  Passed: {passed}, Failed: {failed}, Total: {passed + failed}")
if failed == 0:
    print("All second_derivative tests passed.")
else:
    sys.exit(1)
