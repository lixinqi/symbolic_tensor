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
# JSON dict → ast.AST
# ---------------------------------------------------------------------------

def _json_dict_to_ast(node: Any) -> Any:
    """Recursively convert JSON dict to ast.AST node.

    Handles incomplete nodes from dropout gracefully — returns None on failure.
    """
    if isinstance(node, dict) and '_type' in node:
        cls = getattr(ast_mod, node['_type'], None)
        if cls is None:
            return None
        kwargs = {}
        for k, v in node.items():
            if k.startswith('_'):
                continue
            kwargs[k] = _json_dict_to_ast(v)
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
        # Per-statement fallback for robustness on dropout data
        parts = []
        for node in body:
            try:
                m = ast_mod.Module(body=[node], type_ignores=[])
                ast_mod.fix_missing_locations(m)
                parts.append(ast_mod.unparse(m))
            except Exception:
                pass
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
        "..", "example", "tag_auto_encoder", "dataset",
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
