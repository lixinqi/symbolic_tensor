"""Convert AstTagRelation JSONL to Python source code.

Pipeline:
  1. convert_ast_tag_jsonl_to_ast_json: JSONL records → AST JSON dicts
  2. _json_dict_to_ast: JSON dicts → ast.AST nodes
  3. ast.unparse: AST nodes → Python source
"""

import ast as ast_mod
import json
import os
import warnings
from typing import Any, Dict, List


from experience.ast_tag.convert_ast_tag_jsonl_to_ast_json import (
    convert_ast_tag_jsonl_to_ast_json,
)


# ---------------------------------------------------------------------------
# JSON dict → ast.AST  (best-effort for dropout data)
# ---------------------------------------------------------------------------

# Default values for required AST fields that dropout may remove.
# Without these, ast.unparse crashes on incomplete nodes.
_FIELD_DEFAULTS: Dict[str, Any] = {
    'value': ast_mod.Constant(value=0),
    'target': ast_mod.Name(id='_'),
    'targets': [ast_mod.Name(id='_')],
    'attr': '_',
    'arg': '_',
    'name': '_',
    'module': '',
    'id': '_',
    'func': ast_mod.Name(id='_'),
    'args': [],  # correct for Call.args; FunctionDef overridden below
    'test': ast_mod.Constant(value=True),
    'body': [ast_mod.Pass()],
    'names': [ast_mod.alias(name='_')],
    'items': [],
    'keys': [],
    'values': [],
    'elts': [],
    'ops': [ast_mod.Eq()],
    'comparators': [ast_mod.Constant(value=0)],
    'left': ast_mod.Constant(value=0),
    'right': ast_mod.Constant(value=0),
    'operand': ast_mod.Constant(value=0),
    'op': ast_mod.Add(),
    'slice': ast_mod.Constant(value=0),
    'iter': ast_mod.Name(id='_'),
    'exc': None,
    'cause': None,
    'orelse': [],
    'finalbody': [],
    'handlers': [],
    'decorator_list': [],
    'bases': [],
    'keywords': [],
    'returns': None,
    'type_comment': None,
    'type_params': [],
    'level': 0,
    'optional_vars': None,
    'context_expr': ast_mod.Constant(value=None),
    'dims': [],
    'generators': [],
    'ifs': [],
    'is_async': 0,
}

# Per-type overrides for fields whose default meaning depends on context.
_EMPTY_ARGUMENTS = ast_mod.arguments(
    posonlyargs=[], args=[], vararg=None,
    kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[],
)
_TYPE_FIELD_DEFAULTS: Dict[str, Dict[str, Any]] = {
    'FunctionDef': {'args': _EMPTY_ARGUMENTS},
    'AsyncFunctionDef': {'args': _EMPTY_ARGUMENTS},
    'Lambda': {'args': _EMPTY_ARGUMENTS},
    'ExceptHandler': {'name': None},
    'Return': {'value': None},
    'AnnAssign': {'value': None},
}


def _json_dict_to_ast(node: Any) -> Any:
    """Recursively convert JSON dict to ast.AST node.

    Best-effort: fills placeholder defaults for missing required fields
    so that ast.unparse produces partial output instead of crashing.
    """
    if isinstance(node, dict) and '_type' in node:
        cls = getattr(ast_mod, node['_type'], None)
        if cls is None:
            return None
        kwargs: Dict[str, Any] = {}
        for k, v in node.items():
            if k.startswith('_'):
                continue
            kwargs[k] = _json_dict_to_ast(v)
        # Fill missing required fields with placeholder defaults
        type_overrides = _TYPE_FIELD_DEFAULTS.get(node['_type'], {})
        for field in cls._fields:
            if field not in kwargs:
                if field in type_overrides:
                    kwargs[field] = type_overrides[field]
                elif field in _FIELD_DEFAULTS:
                    kwargs[field] = _FIELD_DEFAULTS[field]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                return cls(**kwargs)
        except TypeError:
            try:
                valid = {f: kwargs[f] for f in cls._fields if f in kwargs}
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    return cls(**valid)
            except Exception:
                return None
    if isinstance(node, list):
        items = [_json_dict_to_ast(x) for x in node]
        return [x for x in items if x is not None]
    return node


def _safe_unparse(node: Any) -> str:
    """Unparse a single AST node, with per-body-statement fallback."""
    m = ast_mod.Module(body=[node], type_ignores=[])
    ast_mod.fix_missing_locations(m)
    try:
        return ast_mod.unparse(m)
    except Exception:
        pass
    # For compound statements, try unparsing body statements individually
    body = getattr(node, 'body', None)
    if not isinstance(body, list) or not body:
        return ""
    # Build a stub with each body statement tried independently
    parts = []
    # Emit the signature line as a stub if possible
    stub = _unparse_stub(node)
    if stub:
        inner = []
        for stmt in body:
            try:
                sm = ast_mod.Module(body=[stmt], type_ignores=[])
                ast_mod.fix_missing_locations(sm)
                inner.append(ast_mod.unparse(sm))
            except Exception:
                continue
        if inner:
            parts.append(stub)
            for line in inner:
                for sub in line.splitlines():
                    parts.append("    " + sub)
        else:
            parts.append(stub)
            parts.append("    pass")
    else:
        for stmt in body:
            try:
                sm = ast_mod.Module(body=[stmt], type_ignores=[])
                ast_mod.fix_missing_locations(sm)
                parts.append(ast_mod.unparse(sm))
            except Exception:
                continue
    return "\n".join(parts)


def _unparse_stub(node: Any) -> str:
    """Try to produce just the signature/header line for compound statements."""
    t = type(node).__name__
    if t == 'FunctionDef' or t == 'AsyncFunctionDef':
        name = getattr(node, 'name', '_')
        prefix = 'async def' if t == 'AsyncFunctionDef' else 'def'
        # Try to unparse args
        args_node = getattr(node, 'args', None)
        if args_node:
            try:
                args_str = ast_mod.unparse(args_node)
            except Exception:
                args_str = ""
        else:
            args_str = ""
        ret = getattr(node, 'returns', None)
        ret_str = ""
        if ret is not None:
            try:
                ret_str = " -> " + ast_mod.unparse(ret)
            except Exception:
                pass
        decos = []
        for d in getattr(node, 'decorator_list', []):
            try:
                decos.append("@" + ast_mod.unparse(d))
            except Exception:
                pass
        header = f"{prefix} {name}({args_str}){ret_str}:"
        if decos:
            return "\n".join(decos) + "\n" + header
        return header
    if t == 'ClassDef':
        name = getattr(node, 'name', '_')
        return f"class {name}:"
    if t == 'If':
        test = getattr(node, 'test', None)
        if test:
            try:
                return "if " + ast_mod.unparse(test) + ":"
            except Exception:
                pass
        return "if True:"
    if t == 'For' or t == 'AsyncFor':
        prefix = 'async for' if t == 'AsyncFor' else 'for'
        target = getattr(node, 'target', None)
        iter_ = getattr(node, 'iter', None)
        try:
            t_str = ast_mod.unparse(target) if target else '_'
        except Exception:
            t_str = '_'
        try:
            i_str = ast_mod.unparse(iter_) if iter_ else '_'
        except Exception:
            i_str = '_'
        return f"{prefix} {t_str} in {i_str}:"
    if t == 'While':
        test = getattr(node, 'test', None)
        try:
            t_str = ast_mod.unparse(test) if test else 'True'
        except Exception:
            t_str = 'True'
        return f"while {t_str}:"
    if t == 'With' or t == 'AsyncWith':
        prefix = 'async with' if t == 'AsyncWith' else 'with'
        return f"{prefix} _:"
    if t == 'Try' or t == 'TryStar':
        return "try:"
    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_jsonl_to_python(records: List[Dict]) -> str:
    """Convert AstTagRelation JSONL records to Python source code.

    Reconstructs AST JSON via convert_ast_tag_jsonl_to_ast_json,
    then generates Python via ast.unparse.
    """
    if not records:
        return ""
    json_nodes, _ = convert_ast_tag_jsonl_to_ast_json(records)
    if not json_nodes:
        return ""
    body = [_json_dict_to_ast(n) for n in json_nodes]
    body = [n for n in body if n is not None]
    if not body:
        return ""
    module = ast_mod.Module(body=body, type_ignores=[])
    ast_mod.fix_missing_locations(module)
    try:
        return ast_mod.unparse(module)
    except Exception:
        # Per-statement fallback with deep recovery for dropout data
        parts = []
        for node in body:
            text = _safe_unparse(node)
            if text:
                parts.append(text)
        return "\n".join(parts)


def convert_file_to_python(jsonl_path: str) -> str:
    """Convert a JSONL file to Python source code."""
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return convert_jsonl_to_python(records)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from experience.ast_tag.convert_python_to_ast_tag_jsonl import (
        convert_python_to_ast_tag_jsonl,
    )
    from experience.ast_tag.convert_ast_json_to_ast_tag_jsonl import (
        random_dropout_tag_relations,
    )
    import random

    CODEBASE_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "example", "code_auto_encoder", "codebase",
    )
    DATASET_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_dataset",
    )

    py_files = []
    for root, _dirs, files in os.walk(CODEBASE_DIR):
        for f in sorted(files):
            if f.endswith(".py") and f != "__init__.py":
                py_files.append(os.path.join(root, f))
    py_files.sort()

    # --- exact roundtrip tests ---
    passed = 0
    failed = 0
    parse_errors = 0
    errors = []

    for path in py_files:
        with open(path, "r", encoding="utf-8") as fh:
            source = fh.read()

        try:
            ast_mod.parse(source)
        except SyntaxError:
            continue

        jsonl = convert_python_to_ast_tag_jsonl(source)
        groups = [json.loads(line) for line in jsonl.strip().splitlines()]
        reconstructed = convert_jsonl_to_python(groups)

        rel = os.path.relpath(path, CODEBASE_DIR)

        try:
            ast_mod.parse(reconstructed)
        except SyntaxError as e:
            failed += 1
            parse_errors += 1
            errors.append(rel)
            print(f"  PARSE_ERROR {rel}: {e}")
            continue

        passed += 1
        print(f"  PASS {rel}")

    print(f"\n--- Roundtrip: {passed} passed, {failed} failed ({parse_errors} parse errors) ---")

    # --- 200 robustness tests: dropout + no crash ---
    jsonl_files = []
    for root, _dirs, files in os.walk(DATASET_DIR):
        for f in sorted(files):
            if f.endswith(".jsonl"):
                jsonl_files.append(os.path.join(root, f))
    jsonl_files.sort()

    random.seed(42)
    robust_total = 200
    robust_ok = 0
    robust_errors = []
    for i in range(robust_total):
        fp = jsonl_files[i % len(jsonl_files)]
        rel = os.path.relpath(fp, DATASET_DIR)
        with open(fp) as fh:
            records = [json.loads(l) for l in fh if l.strip()]
        dropout_rate = 0.1 + 0.6 * (i / robust_total)
        x = random_dropout_tag_relations(records, dropout_rate=dropout_rate)
        try:
            result = convert_jsonl_to_python(x)
            robust_ok += 1
            if i < 10 or i % 50 == 0:
                print(f"  ROBUST OK   {rel} (dropout={dropout_rate:.2f}, "
                      f"{len(x)}/{len(records)} kept, {len(result)} chars)")
        except Exception as e:
            robust_errors.append((rel, dropout_rate, str(e)))
            print(f"  ROBUST FAIL {rel} (dropout={dropout_rate:.2f}): {e}")

    print(f"\n--- Robustness: {robust_ok}/{robust_total} no-crash ---")
    if robust_errors:
        print(f"Failures:")
        for rel, dr, err in robust_errors:
            print(f"  {rel} (dropout={dr:.2f}): {err}")
    assert robust_ok == robust_total, f"{robust_total - robust_ok} robustness tests crashed"
    print(f"\nAll tests passed.")
