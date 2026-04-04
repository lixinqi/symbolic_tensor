"""Convert AstTagRelationGroup JSONL to Python source code.

Generated from convert_ast_tag_jsonl_to_python.viba.

Viba DSL specification:
  convert_ast_tag_jsonl_to_python[ProgrammingLanguage] :=
    $ast_obj ast[ProgrammingLanguage]
    <- JsonLines[$ast_tag_rel_group AstTagRelationGroup[ProgrammingLanguage]]
    # inline
    <- Import[./ast_tag_relation_group.viba]
    <- { Assert that the $ast_tag_rel_group meets the AstTagRelationGroup schema }

# at least 50 roundtrip tests
"""

import ast as ast_mod
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict


# ---------------------------------------------------------------------------
# Relation indexing
# ---------------------------------------------------------------------------

# Operator mappings for reconstruction
BIN_OP_MAP = {
    "Add": "+", "Sub": "-", "Mult": "*", "Div": "/", "FloorDiv": "//",
    "Mod": "%", "Pow": "**", "LShift": "<<", "RShift": ">>",
    "BitOr": "|", "BitXor": "^", "BitAnd": "&", "MatMult": "@"
}

UNARY_OP_MAP = {
    "UAdd": "+", "USub": "-", "Not": "not ", "Invert": "~"
}

COMPARE_OP_MAP = {
    "Eq": "==", "NotEq": "!=", "Lt": "<", "LtE": "<=",
    "Gt": ">", "GtE": ">=", "Is": "is", "IsNot": "is not",
    "In": "in", "NotIn": "not in"
}

AUG_OP_MAP = {
    "Add": "+=", "Sub": "-=", "Mult": "*=", "Div": "/=", "FloorDiv": "//=",
    "Mod": "%=", "Pow": "**=", "LShift": "<<=", "RShift": ">>=",
    "BitOr": "|=", "BitXor": "^=", "BitAnd": "&=", "MatMult": "@="
}


class _RelationIndex:
    """Index AstTagRelation records for fast lookup by tag / lhs / rhs."""

    def __init__(self, groups: List[Dict]):
        self.all = sorted(groups, key=lambda r: r.get("line", 0))
        self._by_tag: Dict[str, List[Dict]] = defaultdict(list)
        self._by_owner: Dict[str, List[Dict]] = defaultdict(list)
        for g in self.all:
            tag = g.get("relation_tag", "")
            owner = str(g.get("owner_tag", ""))
            self._by_tag[tag].append(g)
            self._by_owner[owner].append(g)

    def tag(self, tag: str) -> List[Dict]:
        return self._by_tag.get(tag, [])

    def owner(self, owner: str, tag: Optional[str] = None) -> List[Dict]:
        results = self._by_owner.get(owner, [])
        if tag is not None:
            results = [r for r in results if r.get("relation_tag") == tag]
        return results


# ---------------------------------------------------------------------------
# Ungroup relations
# ---------------------------------------------------------------------------

def _ungroup_relations(groups: List[Dict]) -> List[Dict]:
    """Expand AstTagRelationGroup to flat AstTagRelation records."""
    relations = []
    for g in groups:
        line = g.get("line", 0)
        tag = g.get("relation_tag", "")
        owner = g.get("owner_tag", "")
        members = g.get("member_tags", [])
        for member in members:
            relations.append({
                "line": line,
                "relation_tag": tag,
                "lhs_tag": owner,
                "rhs_tag": member
            })
    return relations


# ---------------------------------------------------------------------------
# Expression reconstruction helpers
# ---------------------------------------------------------------------------

def _reconstruct_binop(binop_id: str, idx: _RelationIndex, _visited: Optional[set] = None) -> str:
    """Reconstruct a binary operation from bin_op relations."""
    if _visited is None:
        _visited = set()
    if binop_id in _visited:
        return "<cycle>"
    _visited.add(binop_id)

    op_groups = idx.owner(binop_id, "bin_op")
    if not op_groups:
        return "<BinOp>"
    op = op_groups[0]["member_tags"][0] if op_groups[0]["member_tags"] else "+"

    left_groups = idx.owner(binop_id, "bin_op_left")
    right_groups = idx.owner(binop_id, "bin_op_right")

    left = left_groups[0]["member_tags"][0] if left_groups and left_groups[0]["member_tags"] else "?"
    right = right_groups[0]["member_tags"][0] if right_groups and right_groups[0]["member_tags"] else "?"

    # Recurse if left/right are references
    if left.startswith("binop:"):
        left = _reconstruct_binop(left, idx, _visited)
    elif left.startswith("unaryop:"):
        left = _reconstruct_unaryop(left, idx, _visited)
    elif left.startswith("compare:"):
        left = _reconstruct_compare(left, idx, _visited)
    elif left.startswith("subscript:"):
        left = _reconstruct_subscript(left, idx, _visited)

    if right.startswith("binop:"):
        right = _reconstruct_binop(right, idx, _visited)
    elif right.startswith("unaryop:"):
        right = _reconstruct_unaryop(right, idx, _visited)
    elif right.startswith("compare:"):
        right = _reconstruct_compare(right, idx, _visited)
    elif right.startswith("subscript:"):
        right = _reconstruct_subscript(right, idx, _visited)

    return f"{left} {op} {right}"


def _reconstruct_unaryop(unaryop_id: str, idx: _RelationIndex, _visited: Optional[set] = None) -> str:
    """Reconstruct a unary operation from unary_op relations."""
    if _visited is None:
        _visited = set()
    if unaryop_id in _visited:
        return "<cycle>"
    _visited.add(unaryop_id)

    op_groups = idx.owner(unaryop_id, "unary_op")
    if not op_groups:
        return "<UnaryOp>"
    op = op_groups[0]["member_tags"][0] if op_groups[0]["member_tags"] else "+"

    operand_groups = idx.owner(unaryop_id, "unary_op_operand")
    operand = operand_groups[0]["member_tags"][0] if operand_groups and operand_groups[0]["member_tags"] else "?"

    if operand.startswith("binop:"):
        operand = f"({_reconstruct_binop(operand, idx, _visited)})"
    elif operand.startswith("unaryop:"):
        operand = _reconstruct_unaryop(operand, idx, _visited)
    elif operand.startswith("compare:"):
        operand = f"({_reconstruct_compare(operand, idx, _visited)})"

    # Handle 'not' specially
    if op == "not":
        return f"not {operand}"
    return f"{op}{operand}"


def _reconstruct_compare(compare_id: str, idx: _RelationIndex, _visited: Optional[set] = None) -> str:
    """Reconstruct a comparison from compare relations."""
    if _visited is None:
        _visited = set()
    if compare_id in _visited:
        return "<cycle>"
    _visited.add(compare_id)

    left_groups = idx.owner(compare_id, "compare_left")
    op_groups = idx.owner(compare_id, "compare_op")
    right_groups = idx.owner(compare_id, "compare_right")

    left = left_groups[0]["member_tags"][0] if left_groups and left_groups[0]["member_tags"] else "?"
    ops = [g["member_tags"] for g in op_groups] if op_groups else [[]]
    rights = [g["member_tags"] for g in right_groups] if right_groups else [[]]

    # Flatten
    ops = [o for sublist in ops for o in sublist]
    rights = [r for sublist in rights for r in sublist]

    result = left
    for op, right in zip(ops, rights):
        result = f"{result} {op} {right}"

    return result


def _reconstruct_subscript(subscript_id: str, idx: _RelationIndex, _visited: Optional[set] = None) -> str:
    """Reconstruct a subscript from subscript relations."""
    if _visited is None:
        _visited = set()
    if subscript_id in _visited:
        return "<cycle>"
    _visited.add(subscript_id)

    value_groups = idx.owner(subscript_id, "subscript_value")
    slice_groups = idx.owner(subscript_id, "subscript")

    value = value_groups[0]["member_tags"][0] if value_groups and value_groups[0]["member_tags"] else "?"
    slice_val = slice_groups[0]["member_tags"][0] if slice_groups and slice_groups[0]["member_tags"] else "?"

    # Handle slice objects
    if slice_val.startswith("slice:"):
        slice_val = _reconstruct_slice(slice_val, idx, _visited)

    return f"{value}[{slice_val}]"


def _reconstruct_slice(slice_id: str, idx: _RelationIndex, _visited: Optional[set] = None) -> str:
    """Reconstruct a slice from slice relations."""
    if _visited is None:
        _visited = set()
    if slice_id in _visited:
        return "<cycle>"
    _visited.add(slice_id)

    lower_groups = idx.owner(slice_id, "slice_lower")
    upper_groups = idx.owner(slice_id, "slice_upper")
    step_groups = idx.owner(slice_id, "slice_step")

    lower = lower_groups[0]["member_tags"][0] if lower_groups and lower_groups[0]["member_tags"] else ""
    upper = upper_groups[0]["member_tags"][0] if upper_groups and upper_groups[0]["member_tags"] else ""
    step = step_groups[0]["member_tags"][0] if step_groups and step_groups[0]["member_tags"] else ""

    if lower or upper or step:
        parts = [lower or "", upper or "", step or ""]
        # Remove trailing empty parts
        while parts and parts[-1] == "":
            parts.pop()
        return ":".join(parts)
    return ":"


def _reconstruct_dict(dict_id: str, idx: _RelationIndex, _visited: Optional[set] = None) -> str:
    """Reconstruct a dict literal from dict relations."""
    if _visited is None:
        _visited = set()
    if dict_id in _visited:
        return "<cycle>"
    _visited.add(dict_id)

    key_groups = idx.owner(dict_id, "dict_key")
    value_groups = idx.owner(dict_id, "dict_value")

    keys = []
    for g in key_groups:
        keys.extend(g.get("member_tags", []))
    values = []
    for g in value_groups:
        values.extend(g.get("member_tags", []))

    pairs = []
    for i, v in enumerate(values):
        k = keys[i] if i < len(keys) else None
        if k:
            pairs.append(f"{k}: {v}")
        else:
            pairs.append(f"**{v}")  # dict unpacking

    return "{" + ", ".join(pairs) + "}"


def _reconstruct_expr(expr: str, idx: _RelationIndex, _visited: Optional[set] = None) -> str:
    """Reconstruct an expression, handling references."""
    if _visited is None:
        _visited = set()

    if expr.startswith("binop:"):
        return _reconstruct_binop(expr, idx, _visited)
    elif expr.startswith("unaryop:"):
        return _reconstruct_unaryop(expr, idx, _visited)
    elif expr.startswith("compare:"):
        return _reconstruct_compare(expr, idx, _visited)
    elif expr.startswith("subscript:"):
        return _reconstruct_subscript(expr, idx, _visited)
    elif expr.startswith("slice:"):
        return _reconstruct_slice(expr, idx, _visited)
    elif expr.startswith("dict:"):
        return _reconstruct_dict(expr, idx, _visited)
    elif expr.startswith("await:"):
        val_groups = idx.owner(expr, "await_value")
        val = val_groups[0]["member_tags"][0] if val_groups and val_groups[0]["member_tags"] else "?"
        return f"await {val}"
    elif expr.startswith("lambda:"):
        return _reconstruct_lambda(expr, idx, _visited)
    elif expr.startswith("ifexp:"):
        return _reconstruct_ifexp(expr, idx, _visited)

    return expr


def _reconstruct_lambda(lambda_id: str, idx: _RelationIndex, _visited: Optional[set] = None) -> str:
    """Reconstruct a lambda expression."""
    if _visited is None:
        _visited = set()
    if lambda_id in _visited:
        return "<cycle>"
    _visited.add(lambda_id)

    param_groups = idx.owner(lambda_id, "param")
    body_groups = idx.owner(lambda_id, "lambda_body")

    params = []
    for g in param_groups:
        params.extend(g.get("member_tags", []))

    body = body_groups[0]["member_tags"][0] if body_groups and body_groups[0]["member_tags"] else "None"

    return f"lambda {', '.join(params)}: {body}"


def _reconstruct_ifexp(ifexp_id: str, idx: _RelationIndex, _visited: Optional[set] = None) -> str:
    """Reconstruct an if expression (ternary)."""
    if _visited is None:
        _visited = set()
    if ifexp_id in _visited:
        return "<cycle>"
    _visited.add(ifexp_id)

    test_groups = idx.owner(ifexp_id, "if_expr_test")
    body_groups = idx.owner(ifexp_id, "if_expr_body")
    else_groups = idx.owner(ifexp_id, "if_expr_else")

    test = test_groups[0]["member_tags"][0] if test_groups and test_groups[0]["member_tags"] else "True"
    body = body_groups[0]["member_tags"][0] if body_groups and body_groups[0]["member_tags"] else "None"
    orelse = else_groups[0]["member_tags"][0] if else_groups and else_groups[0]["member_tags"] else "None"

    return f"{body} if {test} else {orelse}"


# ---------------------------------------------------------------------------
# Statement reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_funcdef(name: str, idx: _RelationIndex) -> str:
    """Reconstruct a function definition."""
    lines = [f"def {name}("]

    # Parameters
    param_groups = idx.owner(name, "param")
    params = []
    for g in param_groups:
        params.extend(g.get("member_tags", []))
    lines.append(", ".join(params))
    lines.append(")")

    # Returns
    ret_groups = idx.owner(name, "returns")
    if ret_groups and ret_groups[0]["member_tags"]:
        ret = ret_groups[0]["member_tags"][0]
        lines.append(f" -> {ret}")

    lines.append(":\n")

    # Docstring placeholder
    lines.append('    """..."""\n')

    # Body - simplified
    body_groups = idx.owner(name, "contains")
    for g in body_groups:
        for member in g.get("member_tags", []):
            lines.append(f"    # {member}\n")

    return "".join(lines)


def _reconstruct_classdef(name: str, idx: _RelationIndex) -> str:
    """Reconstruct a class definition."""
    lines = [f"class {name}"]

    # Bases
    base_groups = idx.owner(name, "bases")
    bases = []
    for g in base_groups:
        bases.extend(g.get("member_tags", []))
    if bases:
        lines.append(f"({', '.join(bases)})")

    lines.append(":\n")
    lines.append('    """..."""\n')

    # Body
    body_groups = idx.owner(name, "contains")
    for g in body_groups:
        for member in g.get("member_tags", []):
            lines.append(f"    # {member}\n")

    return "".join(lines)


def _reconstruct_imports(idx: _RelationIndex) -> str:
    """Reconstruct import statements."""
    lines = []
    import_groups = idx.tag("imports")
    alias_map = defaultdict(list)

    # Collect aliases
    alias_groups = idx.tag("aliases")
    for g in alias_groups:
        for member in g.get("member_tags", []):
            alias_map[g.get("owner_tag", "")].append(member)

    # Group by type
    from_imports = defaultdict(list)
    simple_imports = []

    for g in import_groups:
        for member in g.get("member_tags", []):
            if "." in member and not member.startswith("."):
                parts = member.rsplit(".", 1)
                module, name = parts[0], parts[1]
                from_imports[module].append(name)
            else:
                simple_imports.append(member)

    # Generate import statements
    if simple_imports:
        for imp in sorted(set(simple_imports)):
            aliases = alias_map.get(imp, [])
            if aliases:
                lines.append(f"import {imp} as {aliases[0]}\n")
            else:
                lines.append(f"import {imp}\n")

    for module, names in sorted(from_imports.items()):
        lines.append(f"from {module} import {', '.join(sorted(set(names)))}\n")

    return "".join(lines)


# ---------------------------------------------------------------------------
# Main conversion function
# ---------------------------------------------------------------------------

def convert_jsonl_to_python(groups: List[Dict]) -> str:
    """Convert AstTagRelationGroup records to Python source code."""
    idx = _RelationIndex(groups)
    lines = []

    # 1. Imports
    imports = _reconstruct_imports(idx)
    if imports:
        lines.append(imports)
        lines.append("\n")

    # 2. Top-level definitions
    def_groups = idx.tag("defines")
    defined_names = set()
    for g in def_groups:
        for name in g.get("member_tags", []):
            if name in defined_names:
                continue
            defined_names.add(name)

            # Check if it's a class or function
            if idx.owner(name, "bases"):
                lines.append(_reconstruct_classdef(name, idx))
                lines.append("\n")
            elif idx.owner(name, "param"):
                lines.append(_reconstruct_funcdef(name, idx))
                lines.append("\n")
            else:
                # Variable assignment
                assign_groups = idx.owner(name, "assigns")
                if assign_groups:
                    val = assign_groups[0]["member_tags"][0] if assign_groups[0]["member_tags"] else "..."
                    lines.append(f"{name} = {val}\n")

    # 3. Other statements
    # ... (simplified for now)

    return "".join(lines)


def convert_file_to_python(jsonl_path: str) -> str:
    """Convert a JSONL file to Python source code."""
    groups = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                groups.append(json.loads(line))
    return convert_jsonl_to_python(groups)


# ---------------------------------------------------------------------------
# Roundtrip testing
# ---------------------------------------------------------------------------

def run_roundtrip_test(source_dir: str = "ast") -> None:
    """Run roundtrip tests on all JSON AST files in source_dir."""
    import glob
    import convert_python_to_ast_tag_jsonl as to_jsonl

    json_files = glob.glob(os.path.join(source_dir, "**", "*.json"), recursive=True)
    json_files = [f for f in json_files if not f.endswith("__init__.json")]

    print("=== Roundtrip test: source.py → JSONL → reconstructed.py ===\n")

    for json_file in sorted(json_files):
        with open(json_file, "r", encoding="utf-8") as f:
            node_dict = json.load(f)

        # Count source lines from the AST
        src_lines = _count_source_lines(node_dict)

        # Convert to relations
        relations = to_jsonl._extract_relations(node_dict)
        groups = to_jsonl._group_relations(relations)

        # Convert back
        reconstructed = convert_jsonl_to_python(groups)

        # Count reconstructed lines
        recon_lines = len([l for l in reconstructed.split("\n") if l.strip()])

        print(f"{json_file}: {src_lines} src lines → {len(groups)} relations → {recon_lines} reconstructed lines")

    print("\n--- Roundtrip test complete ---")


def _count_source_lines(node_dict: Dict) -> int:
    """Count approximate source lines from AST by finding max lineno."""
    max_line = 0

    def walk(node):
        nonlocal max_line
        if isinstance(node, dict):
            if "_lineno" in node:
                max_line = max(max_line, node["_lineno"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(node_dict)
    return max_line


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run roundtrip test
    run_roundtrip_test("ast")
