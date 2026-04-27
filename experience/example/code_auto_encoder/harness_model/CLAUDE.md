# Harness Model: Simulate Claude Code with SymbolicTensor Framework

## Goal

Build a model that **simulates Claude Code's agentic tool-use loop** entirely within
the SymbolicTensor/FutureTensor framework. No direct use of Claude Code. No loading
all file contents into LLM context. The LLM only sees what it explicitly requests
through HarnessOp tool calls.

## Philosophy: grep is attention

HarnessOps (grep, read, glob) are **attention over the codebase**. The agent runs in
two sequential stages: gather clean context via tool calls, then generate and validate
answers. Execution trace (raw tool history) is stored in the experience tensor for
learning; it never pollutes the generation input.

## Core abstraction mapping

| Claude Code concept | Framework equivalent |
|---|---|
| Tool (Read, Grep, Glob, Write) | `HarnessOp` subclass |
| Tool execution + validation | `ft_recurrent[tool_use]` |
| Context gathering loop | `ft_recurrent[code_context_gather]` with `accumulate_output` |
| Code generation loop | `ft_recurrent[code_gen]` |
| Experience retrieval | `ft_moe` via `select_qkv_indexes` |
| Learning from mistakes | `StSGD` updating experience |

## Architecture

```
worktree_tensor (batch,)
        │
        ▼
FutureTensor (batch, R)
        │
        │  ft_recurrent[code_context_gather]
        │  accumulate_output = concat clean results
        │  ├── ft_recurrent[tool_use] per step
        │  └── Context validator OK → confidence
        ▼
context_tensor (batch,)
        │
        ▼
   broadcast → (batch, L)
        │
        │  ft_recurrent[code_gen]
        │  Generate answer → HmValidateResultGather → pass/fail
        ▼
output._tensor (batch,)
```

L = `max_codegen_steps`, R = `max_tool_call_retries`.

### Shape arithmetic

```
(batch, R) ──code_context_gather──▶ (batch,) ──broadcast──▶ (batch, L) ──code_gen──▶ (batch,)
```

### Trajectory regex

```
tool       := glob | grep | read | write
context_ok := context_validator_pass

gather_ok   := (tool → scbf){0,R-1} (context_ok → confidence)
gather_fail := (tool → scbf){R}

gen_ok      := generate → validate → confidence
gen_fail    := generate → validate → scbf
             | gather_fail → scbf

trajectory  := (gather_ok → gen_ok)
             | (gen_fail){0,L-1} (gather_ok → gen_ok)
             | (gen_fail){L}
```

## `ft_recurrent` extension

`ft_recurrent` accepts `accumulate_output: Callable[[acc, cur], acc]`.
Default is identity (`None`): each iteration's output replaces the previous.
`code_context_gather` sets it to concatenate clean tool results. When set, the
returned output is the accumulator across all iterations up to exit. When `None`,
the all-fail path falls back to the best single iteration (backward compatible).

## Operators

### Attention operators (`function/`)

| File | Role |
|---|---|
| `harness_op.py` | Base class |
| `hm_glob.py` | Find files by pattern |
| `hm_grep.py` | Search contents by regex |
| `hm_read.py` | Read file with offset/limit |
| `hm_write.py` | Write file |

### Validators (`function/`)

| File | Role |
|---|---|
| `harness_validator_op.py` | Base class |
| `hm_validate_tool_result.py` | Check tool output (empty, error) |
| `hm_validate_syntax.py` | `ast.parse` check |
| `hm_validate_empty.py` | Empty/TODO check |
| `hm_validate_length.py` | Line-count check |
| `hm_validate_balance.py` | Brace/paren/bracket balance |
| `hm_validate_result_gather.py` | Composite gather (takes leafs as `__init__` args) |

## Stage 1: code_context_gather

`ft_recurrent` on `(batch, R)` with `accumulate_output = concat`. Each step invokes
`ft_recurrent[tool_use]` to execute one tool call and validate the result. Valid
tool outputs are appended to the accumulator. A context validator checks whether
the accumulated context is sufficient. Sufficient → `confidence` (exit with context).
Budget exhausted → `scbf`. Raw trace is written to the experience tensor.

## Stage 2: code_gen

Broadcast context tensor to `(batch, L)`. `ft_recurrent` with default
`accumulate_output`. At each step: generate answer from context, run
`HmValidateResultGather`. Pass → `confidence`. Fail → `scbf` (retry with failure
description injected into prompt).

## Experience integration

Queried inside `code_context_gather` at `tool_step_idx=0`. Top-k traces prepended as
few-shot examples. Uses `select_qkv_indexes`, not a separate tensor layer.

## Files to create

| File | Role |
|---|---|
| `function/*.py` | HarnessOps + Validators (listed above) |
| `function/__init__.py` | `ALL_OPS` + `ALL_VALIDATORS` |
| `prepare_worktrees.viba/.py` | Create worktree dirs with masks applied |
| `prepare_experience.viba/.py` | Build ExperienceTensor from codebase |
| `harness_model.viba/.py` | Orchestrator `HarnessModel(nn.Module)` |
| `test_harness.viba/.py` | CLI test runner |

## HarnessModel(nn.Module)

```python
class HarnessModel(nn.Module):
    def __init__(self, experience, max_codegen_steps=8, max_tool_call_retries=5,
                 topk=4, task_prompt="", llm_method="raw_llm_api", llm_env=None):
        ...

    def forward(self, worktree_tensor):
        # Stage 1: gather context
        ft_gather = FutureTensor(..., ft_async_get=code_context_gather)
        context, _ = ft_recurrent(ft_gather, accumulate_output=concat_context)

        # Stage 2: generate code
        ft_gen = context.broadcast_to(...).future_tensor(
            ..., ft_async_get=code_gen)
        ft_final, _ = ft_recurrent(ft_gen)
        ft_final.ft_forward(trigger)
        return ft_final._tensor
```

## Key design decisions

1. **`ft_recurrent` accumulation**: `accumulate_output` parameter keeps clean context
   separate from noisy execution trace. Trace goes to experience; accumulator feeds
   generation.
2. **Two sequential stages**: `code_context_gather` (double-nested) produces context;
   `code_gen` (single-nested) consumes it. No mixed adaptive step.
3. **Context validator decides sufficiency**: deterministic gate, not LLM self-report.
4. **Composite answer validator**: `HmValidateResultGather(*leafs)` takes validators
   as `__init__` arguments. Returns rich failure text for precise retry prompts.
5. **Tool result validator**: every HarnessOp output is checked for emptiness/errors.

## Verification

```bash
python -m experience.example.code_auto_encoder.harness_model.test_harness \
    --total-batch-size 1 --seed 42 --llm-method raw_llm_api \
    --max-codegen-steps 8 --max-tool-call-retries 5 --topk 2
```

Loss metric: `get_edit_distance_ratio`. Target: <= baseline loss.
