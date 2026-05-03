"""
Tests for experience.future_tensor.second_derivative.

Tests are grouped by scenario:
  1. Module imports and structure
  2. need_2nd_derivative: requires_grad behaviour
  3. dispatch_policy context manager + PolicyConflictError
  4. TracePolicy (default): records collected without LLM
  5. Custom policy: selective dispatch
  6. RecurrentGradFn: autograd.Function structure
  7. MoeGradFn: autograd.Function structure
  8. Integration: RecurrentGradFn.backward() dispatches into active TracePolicy
  9. Integration: MoeGradFn.backward() dispatches into active TracePolicy
 10. ReflectionRecord fields

Run:
    python -m experience.future_tensor.second_derivative.test.test_second_derivative
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
from experience.future_tensor.function.recurrent_2nd import RecurrentGradFn
from experience.future_tensor.function.moe_2nd import MoeGradFn

run_test("need_2nd_derivative callable", callable(need_2nd_derivative))
run_test("get_2nd_dispatcher callable", callable(get_2nd_dispatcher))
run_test("dispatch_policy callable", callable(dispatch_policy))
run_test("TracePolicy is Policy subclass", issubclass(TracePolicy, Policy))
run_test("ReflectionRecord importable", ReflectionRecord is not None)
run_test("PolicyConflictError is RuntimeError subclass",
         issubclass(PolicyConflictError, RuntimeError))
run_test("RecurrentGradFn is autograd.Function subclass",
         issubclass(RecurrentGradFn, torch.autograd.Function))
run_test("MoeGradFn is autograd.Function subclass",
         issubclass(MoeGradFn, torch.autograd.Function))

# ── Group 2: need_2nd_derivative ─────────────────────────────────────────────
print("\nGroup 2: need_2nd_derivative")

anchor = torch.nn.Parameter(torch.ones(()))

t = torch.zeros(())
result = need_2nd_derivative(t, anchor)
run_test("returns tensor with same value", result.item() == t.item())
run_test("input is scalar", result.shape == torch.Size([]))
run_test("requires_grad set to True", result.requires_grad is True)

# Already requires_grad — idempotent
t2 = torch.zeros((), requires_grad=True)
result2 = need_2nd_derivative(t2, anchor)
run_test("idempotent when already requires_grad", result2.requires_grad is True)

# Non-scalar second_derivative_start raises
try:
    need_2nd_derivative(torch.zeros(()), torch.nn.Parameter(torch.ones(3)))
    run_test("non-scalar second_derivative_start raises AssertionError", False)
except AssertionError:
    run_test("non-scalar second_derivative_start raises AssertionError", True)

# Non-scalar input raises
try:
    need_2nd_derivative(torch.zeros(3), anchor)
    run_test("non-scalar input raises AssertionError", False)
except AssertionError:
    run_test("non-scalar input raises AssertionError", True)

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

# ── Group 6: RecurrentGradFn autograd.Function structure ──────────────────────
print("\nGroup 6: RecurrentGradFn — autograd.Function structure")

# RecurrentGradFn.forward() = recurrent_backward (1st derivative).
# RecurrentGradFn.backward() = 2nd derivative dispatch.
# We verify the structure by calling RecurrentGradFn.apply() and then
# triggering .backward() on a scalar derived from the result.

with tempfile.TemporaryDirectory() as tmpdir:
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
    from experience.future_tensor.status import Status

    # All kContextOverflow → numeric channel zeros all coefficients
    # → no AgentTask fires → no LLM API key needed for the 1st backward
    grad_out6 = make_tensor(["diff text"], tmpdir)
    grad_out6.data[0] = 1.0
    grad_out6.requires_grad_(True)

    inp6 = make_tensor(["original"], tmpdir)
    inp6.data[0] = Status.convert_status_to_float(Status.kContextOverflow)

    out6 = make_tensor(["result"], tmpdir)
    out6.data[0] = Status.convert_status_to_float(Status.kContextOverflow)

    pmt6 = make_tensor(["prompt"], tmpdir)
    pmt6.data[0] = Status.convert_status_to_float(Status.kContextOverflow)

    coll6 = []
    with dispatch_policy(TracePolicy(coll6)):
        # Call the 1st backward via RecurrentGradFn.apply()
        result6 = RecurrentGradFn.apply(
            grad_out6, inp6, out6, pmt6,
            8, None, "test", "raw_llm_api", None,
        )
        # result6 has RecurrentGradFnBackward as grad_fn.
        # Calling .backward() on it triggers RecurrentGradFn.backward() = 2nd derivative.
        result6.sum().backward()

    run_test("RecurrentGradFn is autograd.Function subclass",
             issubclass(RecurrentGradFn, torch.autograd.Function))
    run_test("2nd derivative dispatched via backward()", len(coll6) >= 1)
    if len(coll6) >= 1:
        run_test("record fn is recurrent_backward", coll6[0].fn is recurrent_backward)
        run_test("grad_output in inputs", "grad_output" in coll6[0].inputs)

# ── Group 7: MoeGradFn autograd.Function structure ────────────────────────────
print("\nGroup 7: MoeGradFn — autograd.Function structure")

run_test("MoeGradFn is autograd.Function subclass",
         issubclass(MoeGradFn, torch.autograd.Function))
from experience.symbolic_tensor.function.st_moe_backward import st_moe_backward as _st_moe_backward
run_test("st_moe_backward callable", callable(_st_moe_backward))

# ── Group 8: Integration — RecurrentGradFn.backward() dispatches ───────────────
print("\nGroup 8: Integration — RecurrentGradFn.backward() → TracePolicy")

# When FtRecurrent.backward() calls RecurrentGradFn.apply(...), and the caller
# later calls second_derivative_start.grad.backward(), PyTorch will invoke
# RecurrentGradFn.backward() which dispatches the 2nd derivative.
# Here we simulate this by calling RecurrentGradFn.apply() on leaf tensors
# and calling .backward() on a scalar to trigger the 2nd derivative.

with tempfile.TemporaryDirectory() as tmpdir:
    input8 = make_tensor(["overflow0", "overflow1"], tmpdir)
    input8.data[0] = Status.convert_status_to_float(Status.kContextOverflow)
    input8.data[1] = Status.convert_status_to_float(Status.kContextOverflow)

    output8 = make_tensor(["overflow0"], tmpdir)
    output8.data[0] = Status.convert_status_to_float(Status.kContextOverflow)

    grad_output8 = make_tensor(["improved text"], tmpdir)
    grad_output8.data[0] = 1.0
    grad_output8.requires_grad_(True)

    prompt8 = make_tensor(["translate", "unused"], tmpdir)
    prompt8.data.fill_(1.0)

    coll8 = []
    with dispatch_policy(TracePolicy(coll8)):
        gi8 = RecurrentGradFn.apply(
            grad_output8, input8, output8, prompt8,
            8, None, "no llm needed", "raw_llm_api", None,
        )
        # gi8 has RecurrentGradFnBackward as grad_fn.
        # Calling .backward() on it triggers RecurrentGradFn.backward() = 2nd derivative.
        gi8.sum().backward()

    run_test("RecurrentGradFn.backward() dispatched 2nd derivative", len(coll8) >= 1)
    if len(coll8) >= 1:
        run_test("record fn is recurrent_backward", coll8[0].fn is recurrent_backward)
        run_test("grad_output passed through",
                 coll8[0].inputs.get("grad_output") is grad_output8)
        run_test("input passed through", coll8[0].inputs.get("input") is input8)
        run_test("output passed through", coll8[0].inputs.get("output") is output8)
        run_test("prompt_tensor passed through",
                 coll8[0].inputs.get("prompt_tensor") is prompt8)
        run_test("task_prompt forwarded",
                 coll8[0].inputs.get("task_prompt") == "no llm needed")
        run_test("grad_input in inputs", "grad_input" in coll8[0].inputs)
        run_test("topk_self_confidence_but_failed in inputs",
                 "topk_self_confidence_but_failed" in coll8[0].inputs)

# ── Group 9: Integration — MoeGradFn.backward() dispatches ────────────────────
print("\nGroup 9: Integration — MoeGradFn.backward() → TracePolicy")

# Load API credentials (st_moe_backward fires LLM for grad_experience)
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
    grad_out9.requires_grad_(True)

    idx9 = [[torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long)]]

    coll9 = []
    with dispatch_policy(TracePolicy(coll9)):
        gi9, ge9 = MoeGradFn.apply(
            grad_out9, inp9, out9, exp9,
            "Translate English to French.",  # task_prompt
            1,                               # topk
            "raw_llm_api",                   # llm_method
            None,                            # llm_env
            None,                            # context
            None,                            # grad_input_prompt
            None,                            # grad_exp_key_prompt
            None,                            # grad_exp_value_prompt
            idx9,                            # selected_experience_qkv_indexes_list
        )
        # Trigger 2nd derivative by calling .backward() on output of MoeGradFn
        if gi9 is not None and gi9.requires_grad:
            gi9.sum().backward()
        elif ge9 is not None and ge9.requires_grad:
            ge9.sum().backward()

    run_test("MoeGradFn.backward() dispatched 2nd derivative", len(coll9) >= 1)
    if len(coll9) >= 1:
        run_test("record fn is st_moe_backward", coll9[0].fn is st_moe_backward)
        run_test("grad_output in inputs",
                 coll9[0].inputs.get("grad_output") is grad_out9)
        run_test("experience in inputs", coll9[0].inputs.get("experience") is exp9)
        run_test("task_prompt forwarded",
                 coll9[0].inputs.get("task_prompt") == "Translate English to French.")
        run_test("grad_input in inputs", "grad_input" in coll9[0].inputs)
        run_test("grad_experience in inputs", "grad_experience" in coll9[0].inputs)
        run_test("context in inputs", "context" in coll9[0].inputs)
        run_test("topk in inputs", "topk" in coll9[0].inputs)

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
