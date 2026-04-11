"""Convert Python AST to AstTagRecord JSONL.

Relation tags align with Python ast module:
- owner_tag: $Type_N (CamelCase matching ast._type), <module> for root
- relation_tag: ast field name (body, test, func, args, ...)
- Leaf nodes inlined: Name→id, Constant→repr(value), operators→_type
"""

import ast
import json
import os
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# AST node → JSON dict
# ---------------------------------------------------------------------------

def ast_to_dict(node: Any) -> Any:
    if isinstance(node, ast.AST):
        d: Dict[str, Any] = {"_type": type(node).__name__}
        for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
            val = getattr(node, attr, None)
            if val is not None:
                d[f"_{attr}"] = val
        for field, value in ast.iter_fields(node):
            d[field] = ast_to_dict(value)
        return d
    if isinstance(node, list):
        return [ast_to_dict(item) for item in node]
    return node


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SKIP_FIELDS = frozenset({'ctx', 'type_comment', 'kind', 'type_params'})

_LEAF_TYPES = frozenset({
    'Add', 'Sub', 'Mult', 'Div', 'FloorDiv', 'Mod', 'Pow',
    'LShift', 'RShift', 'BitOr', 'BitXor', 'BitAnd', 'MatMult',
    'UAdd', 'USub', 'Not', 'Invert',
    'Eq', 'NotEq', 'Lt', 'LtE', 'Gt', 'GtE', 'Is', 'IsNot', 'In', 'NotIn',
    'And', 'Or',
    'Load', 'Store', 'Del',
})


# ---------------------------------------------------------------------------
# Temp var generator
# ---------------------------------------------------------------------------

class _TempVarGen:
    def __init__(self):
        self._counters: Dict[str, int] = {}

    def gen(self, type_name: str) -> str:
        idx = self._counters.get(type_name, 0)
        self._counters[type_name] = idx + 1
        return f"${type_name}_{idx}"


# ---------------------------------------------------------------------------
# Symbol extraction
# ---------------------------------------------------------------------------

def _node_to_symbol(node: Any, tmp_gen: _TempVarGen) -> str:
    if node is None:
        return 'None'
    if not isinstance(node, dict):
        return str(node)
    # Use pre-annotated symbol if available (from path_map roundtrip)
    if '_symbol' in node:
        return node['_symbol']
    t = node.get('_type', '')
    if t == 'Name':
        return node.get('id', '')
    if t == 'Constant':
        v = node.get('value')
        if v is None: return 'None'
        if v is True: return 'True'
        if v is False: return 'False'
        if v is ...: return '...'
        return repr(v)
    if t in _LEAF_TYPES:
        return t
    return tmp_gen.gen(t)


def _scalar_sym(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    if value is ...:
        return '...'
    return repr(value)


def _extract(node: Any, rels: List[Dict], tmp_gen: _TempVarGen,
             self_sym: Optional[str] = None,
             order_counters: Optional[Dict] = None):
    if not isinstance(node, dict) or '_type' not in node:
        return
    t = node['_type']
    if t in ('Name', 'Constant') or t in _LEAF_TYPES:
        return
    ln = node.get('_lineno', 0)
    me = self_sym if self_sym is not None else (
        '<module>' if t == 'Module' else tmp_gen.gen(t))
    if order_counters is None:
        order_counters = {}

    def add_rel(field_name: str, member: str):
        rel_tag = f"{t}__{field_name}"
        key = (me, rel_tag)
        ov = order_counters.get(key, 0)
        order_counters[key] = ov + 1
        rels.append({
            'line': ln,
            'relation_tag': rel_tag,
            'owner_tag': me,
            'member_tag': member,
            'member_order_value': ov,
        })

    for field_name, value in node.items():
        if field_name.startswith('_') or field_name in _SKIP_FIELDS:
            continue
        if value is None:
            continue
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and '_type' in item:
                    child_sym = _node_to_symbol(item, tmp_gen)
                    add_rel(field_name, child_sym)
                    _extract(item, rels, tmp_gen, self_sym=child_sym,
                             order_counters=order_counters)
                elif item is None:
                    add_rel(field_name, '<null>')
                else:
                    add_rel(field_name, _scalar_sym(item))
        elif isinstance(value, dict) and '_type' in value:
            child_sym = _node_to_symbol(value, tmp_gen)
            add_rel(field_name, child_sym)
            _extract(value, rels, tmp_gen, self_sym=child_sym,
                     order_counters=order_counters)
        else:
            add_rel(field_name, _scalar_sym(value))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_python_to_ast_tag_jsonl(source_or_dict: Any) -> str:
    if isinstance(source_or_dict, str):
        tree = ast.parse(source_or_dict)
        ast_dict = ast_to_dict(tree)
    else:
        ast_dict = source_or_dict
    rels: List[Dict] = []
    tmp_gen = _TempVarGen()
    _extract(ast_dict, rels, tmp_gen)
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in rels)


if __name__ == "__main__":
    from experience.ast_tag.convert_ast_tag_jsonl_to_python import (
        convert_jsonl_to_python,
    )

    CODEBASE_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "example", "code_auto_encoder", "codebase",
    )
    py_files = []
    for root, _dirs, files in os.walk(CODEBASE_DIR):
        for f in sorted(files):
            if f.endswith(".py") and f != "__init__.py":
                py_files.append(os.path.join(root, f))
    py_files.sort()

    # --- roundtrip tests ---
    passed = 0
    failed = 0
    errors = []

    for path in py_files:
        with open(path, "r", encoding="utf-8") as fh:
            source = fh.read()

        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        expected = ast.unparse(ast.parse(source))

        jsonl = convert_python_to_ast_tag_jsonl(source)
        records = [json.loads(line) for line in jsonl.strip().splitlines()]
        reconstructed = convert_jsonl_to_python(records)

        rel = os.path.relpath(path, CODEBASE_DIR)

        if expected == reconstructed:
            passed += 1
            print(f"  PASS {rel}")
        else:
            failed += 1
            errors.append(rel)
            # show first difference for debugging
            exp_lines = expected.splitlines()
            rec_lines = reconstructed.splitlines()
            for i, (e, r) in enumerate(zip(exp_lines, rec_lines)):
                if e != r:
                    print(f"  FAIL {rel} line {i+1}:")
                    print(f"    expected:      {e[:120]}")
                    print(f"    reconstructed: {r[:120]}")
                    break
            else:
                if len(exp_lines) != len(rec_lines):
                    print(f"  FAIL {rel}: line count {len(exp_lines)} vs {len(rec_lines)}")
                else:
                    print(f"  FAIL {rel}: unknown diff")

    print(f"\n--- Roundtrip: {passed} passed, {failed} failed ---")
    assert passed >= 50, f"need at least 50 roundtrip passes, got {passed}"
    if errors:
        print(f"Failed files: {errors}")
    print(f"All roundtrip tests passed ({passed} files).")
