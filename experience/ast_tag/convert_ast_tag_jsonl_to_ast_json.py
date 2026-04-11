"""Convert AST Tag JSONL relations back to AST JSON tree.

Since relation_tags align with ast field names:
- Extract _type from owner_tag prefix ($Type_N → Type)
- relation_tag is the field name
- Reconstruction is a generic lookup — no per-type case analysis
"""

import ast
import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# AST field metadata (which fields are lists, identifiers, ints)
# ---------------------------------------------------------------------------

_LIST_FIELDS: Set[Tuple[str, str]] = set()
for _name in dir(ast):
    _cls = getattr(ast, _name)
    if isinstance(_cls, type) and issubclass(_cls, ast.AST):
        _ft = getattr(_cls, '_field_types', None)
        if _ft:
            for _f, _t in _ft.items():
                if '*' in str(_t):
                    _LIST_FIELDS.add((_name, _f))

# Hardcoded fallback for Python < 3.13
_LIST_FIELDS |= {
    ('Module', 'body'),
    ('FunctionDef', 'body'), ('FunctionDef', 'decorator_list'),
    ('AsyncFunctionDef', 'body'), ('AsyncFunctionDef', 'decorator_list'),
    ('ClassDef', 'body'), ('ClassDef', 'bases'), ('ClassDef', 'keywords'),
    ('ClassDef', 'decorator_list'),
    ('Delete', 'targets'), ('Assign', 'targets'),
    ('For', 'body'), ('For', 'orelse'),
    ('AsyncFor', 'body'), ('AsyncFor', 'orelse'),
    ('While', 'body'), ('While', 'orelse'),
    ('If', 'body'), ('If', 'orelse'),
    ('With', 'items'), ('With', 'body'),
    ('AsyncWith', 'items'), ('AsyncWith', 'body'),
    ('Try', 'body'), ('Try', 'handlers'), ('Try', 'orelse'), ('Try', 'finalbody'),
    ('TryStar', 'body'), ('TryStar', 'handlers'), ('TryStar', 'orelse'),
    ('TryStar', 'finalbody'),
    ('Import', 'names'), ('ImportFrom', 'names'),
    ('Global', 'names'), ('Nonlocal', 'names'),
    ('arguments', 'posonlyargs'), ('arguments', 'args'), ('arguments', 'kwonlyargs'),
    ('arguments', 'kw_defaults'), ('arguments', 'defaults'),
    ('BoolOp', 'values'), ('Compare', 'ops'), ('Compare', 'comparators'),
    ('Call', 'args'), ('Call', 'keywords'),
    ('JoinedStr', 'values'), ('Dict', 'keys'), ('Dict', 'values'),
    ('Set', 'elts'), ('List', 'elts'), ('Tuple', 'elts'),
    ('ListComp', 'generators'), ('SetComp', 'generators'),
    ('DictComp', 'generators'), ('GeneratorExp', 'generators'),
    ('comprehension', 'ifs'),
    ('ExceptHandler', 'body'),
    ('Match', 'cases'),
    ('match_case', 'body'),
    ('MatchSequence', 'patterns'), ('MatchMapping', 'keys'),
    ('MatchMapping', 'patterns'), ('MatchClass', 'patterns'),
    ('MatchClass', 'kwd_attrs'), ('MatchClass', 'kwd_patterns'),
    ('MatchOr', 'patterns'),
}

_IDENTIFIER_FIELDS = frozenset({
    ('FunctionDef', 'name'), ('AsyncFunctionDef', 'name'), ('ClassDef', 'name'),
    ('arg', 'arg'), ('keyword', 'arg'),
    ('alias', 'name'), ('alias', 'asname'),
    ('ImportFrom', 'module'), ('ExceptHandler', 'name'),
    ('Attribute', 'attr'),
    ('Global', 'names'), ('Nonlocal', 'names'),
})

_INT_FIELDS = frozenset({
    ('ImportFrom', 'level'), ('AnnAssign', 'simple'),
    ('comprehension', 'is_async'), ('FormattedValue', 'conversion'),
})

_OPERATOR_TYPES = frozenset({
    'Add', 'Sub', 'Mult', 'Div', 'FloorDiv', 'Mod', 'Pow',
    'LShift', 'RShift', 'BitOr', 'BitXor', 'BitAnd', 'MatMult',
    'UAdd', 'USub', 'Not', 'Invert',
    'Eq', 'NotEq', 'Lt', 'LtE', 'Gt', 'GtE', 'Is', 'IsNot', 'In', 'NotIn',
    'And', 'Or',
})


# ---------------------------------------------------------------------------
# Relation index
# ---------------------------------------------------------------------------

class _RelationIndex:
    def __init__(self, records: List[Dict]):
        self._fwd: Dict[Tuple[str, str], List[Tuple[str, int, int]]] = {}
        self._owner_tags: Dict[str, set] = {}
        for r in records:
            owner = r["owner_tag"]
            rel = r["relation_tag"]
            member = r["member_tag"]
            order = r["member_order_value"]
            line = r.get("line", 0)
            self._fwd.setdefault((owner, rel), []).append((member, order, line))
            self._owner_tags.setdefault(owner, set()).add(rel)
        for key in self._fwd:
            self._fwd[key].sort(key=lambda x: x[1])

    def get(self, owner: str, rel_tag: str) -> List[str]:
        return [m for m, _, _ in self._fwd.get((owner, rel_tag), [])]

    def tags(self, owner: str) -> set:
        return self._owner_tags.get(owner, set())

    def get_line(self, owner: str) -> int:
        tags = self._owner_tags.get(owner, set())
        min_ln = 0
        for tag in tags:
            for _, _, ln in self._fwd.get((owner, tag), []):
                if ln > 0 and (min_ln == 0 or ln < min_ln):
                    min_ln = ln
        return min_ln


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

_TYPE_RE = re.compile(r"^\$(.+?)_\d+$")


def _extract_type(sym: str) -> str:
    m = _TYPE_RE.match(sym)
    return m.group(1) if m else ""


def _get_ast_fields(type_name: str) -> Tuple[str, ...]:
    cls = getattr(ast, type_name, None)
    if cls and hasattr(cls, '_fields'):
        return cls._fields
    return ()


def _leaf_to_node(sym: str) -> Any:
    if sym in _OPERATOR_TYPES:
        return {"_type": sym}
    if sym == 'None':
        return {"_type": "Constant", "value": None}
    if sym == 'True':
        return {"_type": "Constant", "value": True}
    if sym == 'False':
        return {"_type": "Constant", "value": False}
    if sym == '...':
        return {"_type": "Constant", "value": ...}
    if sym.startswith("'") or sym.startswith('"') or sym.startswith("b'") or sym.startswith('b"'):
        try:
            return {"_type": "Constant", "value": eval(sym)}
        except Exception:
            pass
    try:
        return {"_type": "Constant", "value": int(sym)}
    except ValueError:
        pass
    try:
        return {"_type": "Constant", "value": float(sym)}
    except ValueError:
        pass
    return {"_type": "Name", "id": sym}


def _reconstruct(sym: str, idx: _RelationIndex, path: str,
                  path_map: Dict[str, str]) -> Any:
    if not sym.startswith('$') and sym != '<module>':
        return _leaf_to_node(sym)

    path_map[path] = sym
    type_name = 'Module' if sym == '<module>' else _extract_type(sym)
    node: Dict[str, Any] = {"_type": type_name}

    ln = idx.get_line(sym)
    if ln > 0:
        node["_lineno"] = ln

    for field_name in _get_ast_fields(type_name):
        members = idx.get(sym, f"{type_name}__{field_name}")
        if not members:
            continue

        is_list = (type_name, field_name) in _LIST_FIELDS
        is_ident = (type_name, field_name) in _IDENTIFIER_FIELDS
        is_int = (type_name, field_name) in _INT_FIELDS

        if is_list:
            node[field_name] = [
                _reconstruct_member(m, is_ident, is_int, idx,
                                     f"{path}/{field_name}/{i}", path_map)
                for i, m in enumerate(members)
            ]
        else:
            node[field_name] = _reconstruct_member(
                members[0], is_ident, is_int, idx,
                f"{path}/{field_name}", path_map)

    return node


def _reconstruct_member(sym: str, is_ident: bool, is_int: bool,
                         idx: _RelationIndex, path: str,
                         path_map: Dict[str, str]) -> Any:
    if sym == '<null>':
        return None
    if is_ident:
        return sym
    if is_int:
        try:
            return int(sym)
        except ValueError:
            return sym
    if sym.startswith('$'):
        return _reconstruct(sym, idx, path, path_map)
    return _leaf_to_node(sym)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_ast_tag_jsonl_to_ast_json(
    records: List[Dict],
) -> Tuple[List[Dict], Dict[str, str]]:
    idx = _RelationIndex(records)
    path_map: Dict[str, str] = {}
    body_members = idx.get("<module>", "Module__body")
    json_nodes = [
        _reconstruct(sym, idx, f"/body/{i}", path_map)
        for i, sym in enumerate(body_members)
    ]
    return json_nodes, path_map


if __name__ == "__main__":
    import os
    from experience.ast_tag.convert_python_to_ast_tag_jsonl import (
        convert_python_to_ast_tag_jsonl, ast_to_dict,
    )

    SKIP = {'type_comment', 'kind', 'type_params', 'ctx'}

    def strip(node):
        if isinstance(node, dict):
            return {k: strip(v) for k, v in node.items()
                    if (not k.startswith('_') or k == '_type') and k not in SKIP
                    and v is not None and v != []}
        if isinstance(node, list):
            return [strip(x) for x in node]
        return node

    CODEBASE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "example", "code_auto_encoder", "codebase",
    )
    py_files = []
    for root, _dirs, files in os.walk(CODEBASE):
        for f in sorted(files):
            if f.endswith(".py") and f != "__init__.py":
                py_files.append(os.path.join(root, f))
    py_files.sort()

    passed = 0
    for fp in py_files:
        rel = os.path.relpath(fp, CODEBASE)
        with open(fp) as fh:
            source = fh.read()
        import ast as _ast
        tree = _ast.parse(source)
        original = strip(ast_to_dict(tree)['body'])

        jsonl_str = convert_python_to_ast_tag_jsonl(source)
        records = [json.loads(l) for l in jsonl_str.strip().splitlines()]
        recon, _ = convert_ast_tag_jsonl_to_ast_json(records)
        recon_clean = strip(recon)

        if original == recon_clean:
            passed += 1
        else:
            print(f"DIFF {rel}")
    print(f"\nRoundtrip: {passed}/{len(py_files)}")
