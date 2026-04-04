"""Convert AstTagRelation JSONL to Python source code.

Generated from convert_ast_tag_jsonl_to_python.viba.

Viba DSL specification:
  convert_ast_tag_jsonl_to_python[ProgrammingLanguage] :=
    JsonLines[AstTagRelation[ProgrammingLanguage]]
    <- $ast_obj ast[ProgrammingLanguage]
    # inline
    <- Import[./ast_tag_relation_group.viba]
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

    def __init__(self, relations: List[Dict]):
        self.all = sorted(relations, key=lambda r: r.get("line", 0))
        self._by_tag: Dict[str, List[Dict]] = defaultdict(list)
        self._by_lhs: Dict[str, List[Dict]] = defaultdict(list)
        self._by_rhs: Dict[str, List[Dict]] = defaultdict(list)
        for r in self.all:
            self._by_tag[r["relation_tag"]].append(r)
            self._by_lhs[str(r["lhs_tag"])].append(r)
            self._by_rhs[str(r["rhs_tag"])].append(r)

    def tag(self, tag: str, lhs: Optional[str] = None,
            rhs: Optional[str] = None) -> List[Dict]:
        results = self._by_tag.get(tag, [])
        if lhs is not None:
            results = [r for r in results if str(r["lhs_tag"]) == lhs]
        if rhs is not None:
            results = [r for r in results if str(r["rhs_tag"]) == rhs]
        return results

    def lhs(self, lhs: str, tag: Optional[str] = None) -> List[Dict]:
        results = self._by_lhs.get(lhs, [])
        if tag is not None:
            results = [r for r in results if r["relation_tag"] == tag]
        return results

    def rhs(self, rhs: str, tag: Optional[str] = None) -> List[Dict]:
        results = self._by_rhs.get(rhs, [])
        if tag is not None:
            results = [r for r in results if r["relation_tag"] == tag]
        return results


# ---------------------------------------------------------------------------
# Expression reconstruction helpers
# ---------------------------------------------------------------------------

def _reconstruct_binop(binop_id: str, idx: _RelationIndex, _visited: Optional[set] = None) -> str:
    """Reconstruct a binary operation from bin_op relations."""
    if _visited is None:
        _visited = set()

    op_rel = idx.tag("bin_op", rhs=binop_id)
    if not op_rel:
        return "<BinOp>"
    op = BIN_OP_MAP.get(op_rel[0]["lhs_tag"], op_rel[0]["lhs_tag"])

    left_rel = idx.tag("bin_op_left", lhs=binop_id)
    right_rel = idx.tag("bin_op_right", lhs=binop_id)

    left = _try_reconstruct_expr(left_rel[0]["rhs_tag"] if left_rel else "?", idx, _visited.copy())
    right = _try_reconstruct_expr(right_rel[0]["rhs_tag"] if right_rel else "?", idx, _visited.copy())

    return f"{left} {op} {right}"


def _reconstruct_unaryop(unary_id: str, idx: _RelationIndex, _visited: Optional[set] = None) -> str:
    """Reconstruct a unary operation from unary_op relations."""
    if _visited is None:
        _visited = set()

    op_rel = idx.tag("unary_op", rhs=unary_id)
    if not op_rel:
        return "<UnaryOp>"
    op = UNARY_OP_MAP.get(op_rel[0]["lhs_tag"], op_rel[0]["lhs_tag"])

    operand_rel = idx.tag("unary_op_operand", lhs=unary_id)
    operand = _try_reconstruct_expr(operand_rel[0]["rhs_tag"] if operand_rel else "?", idx, _visited.copy())

    return f"{op}{operand}"


def _reconstruct_compare(compare_id: str, idx: _RelationIndex, _visited: Optional[set] = None) -> str:
    """Reconstruct a comparison expression from compare relations."""
    if _visited is None:
        _visited = set()

    left_rel = idx.tag("compare_left", lhs=compare_id)
    if not left_rel:
        return "<Compare>"

    left = _try_reconstruct_expr(left_rel[0]["rhs_tag"], idx, _visited.copy())
    ops = idx.tag("compare_op", lhs=compare_id)
    rights = idx.tag("compare_right", lhs=compare_id)

    parts = [left]
    for op_r, right_r in zip(ops, rights):
        op = COMPARE_OP_MAP.get(op_r["rhs_tag"], op_r["rhs_tag"])
        parts.append(op)
        rhs = _try_reconstruct_expr(right_r["rhs_tag"], idx, _visited.copy())
        parts.append(rhs)

    return " ".join(parts)


def _reconstruct_boolop(boolop_id: str, idx: _RelationIndex, _visited: Optional[set] = None) -> str:
    """Reconstruct a boolean operation from bool_op relations."""
    if _visited is None:
        _visited = set()

    op_rel = idx.tag("bool_op", rhs=boolop_id)
    if not op_rel:
        return "<BoolOp>"
    op = "and" if op_rel[0]["lhs_tag"] == "And" else "or"

    operands = idx.tag("bool_op_operand", lhs=boolop_id)
    vals = [_try_reconstruct_expr(r["rhs_tag"], idx, _visited.copy()) for r in operands]

    return f" {op} ".join(vals)


def _reconstruct_subscript(subscript_id: str, idx: _RelationIndex, _visited: Optional[set] = None) -> str:
    """Reconstruct a subscript expression."""
    if _visited is None:
        _visited = set()

    val_rel = idx.tag("subscript_value", lhs=subscript_id)
    if not val_rel:
        return "<Subscript>"

    value = _try_reconstruct_expr(str(val_rel[0]["rhs_tag"]), idx, _visited.copy())

    # Get the subscript relation (index/slice)
    sub_rel = idx.tag("subscript", lhs=subscript_id)
    if not sub_rel:
        return f"{value}[...]"

    index = str(sub_rel[0]["rhs_tag"])

    # Resolve any embedded references in the index (without adding this subscript_id to visited yet)
    index = _try_reconstruct_expr(index, idx, _visited.copy())

    # Check if it's a slice
    slice_rels = idx.tag("slice", lhs=subscript_id)
    if slice_rels:
        # It's a slice - get lower, upper, step
        lower_rel = idx.tag("slice_lower", lhs=subscript_id)
        upper_rel = idx.tag("slice_upper", lhs=subscript_id)
        step_rel = idx.tag("slice_step", lhs=subscript_id)

        lower = _try_reconstruct_expr(lower_rel[0]["rhs_tag"], idx, _visited.copy()) if lower_rel else ""
        upper = _try_reconstruct_expr(upper_rel[0]["rhs_tag"], idx, _visited.copy()) if upper_rel else ""
        step = _try_reconstruct_expr(step_rel[0]["rhs_tag"], idx, _visited.copy()) if step_rel else ""

        if step:
            return f"{value}[{lower}:{upper}:{step}]"
        return f"{value}[{lower}:{upper}]"

    # Handle tuple index - remove outer parentheses for cleaner output
    # e.g., Dict[(str,int)] -> Dict[str, int]
    if index.startswith("(") and index.endswith(")"):
        # This is a tuple - extract elements
        inner = index[1:-1]
        # The tuple elements are comma-separated
        return f"{value}[{inner}]"

    return f"{value}[{index}]"


def _reconstruct_comprehension(comp_id: str, idx: _RelationIndex, _visited: Optional[set] = None) -> str:
    """Reconstruct a comprehension expression."""
    if _visited is None:
        _visited = set()

    body_rel = idx.tag("comprehension_body", lhs=comp_id)
    target_rel = idx.tag("comprehension_target", lhs=comp_id)
    iter_rel = idx.tag("comprehension_iter", lhs=comp_id)
    if_rels = idx.tag("comprehension_if", lhs=comp_id)

    # Resolve the body expression
    body = "?"
    if body_rel:
        body = _try_reconstruct_expr(str(body_rel[0]["rhs_tag"]), idx, _visited)

    target = target_rel[0]["rhs_tag"] if target_rel else "?"
    it = _try_reconstruct_expr(str(iter_rel[0]["rhs_tag"]) if iter_rel else "?", idx, _visited)

    comp = f"{body} for {target} in {it}"
    for if_r in if_rels:
        cond = _try_reconstruct_expr(str(if_r['rhs_tag']), idx, _visited)
        comp += f" if {cond}"

    # Determine if it's a list/set/generator/dict comprehension
    if ":" in body and body.count(":") == 1 and not body.startswith("{"):
        return "{" + comp + "}"  # DictComp
    # Check if it should be a set or list based on context
    # Default to list comprehension
    return "[" + comp + "]"


def _reconstruct_ifexp(ifexp_id: str, idx: _RelationIndex, _visited: Optional[set] = None) -> str:
    """Reconstruct an if expression (ternary)."""
    if _visited is None:
        _visited = set()

    body_rel = idx.tag("if_expr_body", lhs=ifexp_id)
    test_rel = idx.tag("if_expr_test", lhs=ifexp_id)
    else_rel = idx.tag("if_expr_else", lhs=ifexp_id)

    body = _try_reconstruct_expr(body_rel[0]["rhs_tag"] if body_rel else "?", idx, _visited.copy())
    test = _try_reconstruct_expr(test_rel[0]["rhs_tag"] if test_rel else "?", idx, _visited.copy())
    else_ = _try_reconstruct_expr(else_rel[0]["rhs_tag"] if else_rel else "?", idx, _visited.copy())

    return f"{body} if {test} else {else_}"


def _reconstruct_lambda(lambda_id: str, idx: _RelationIndex, _visited: Optional[set] = None) -> str:
    """Reconstruct a lambda expression."""
    if _visited is None:
        _visited = set()

    body_rel = idx.tag("lambda_body", lhs=lambda_id)
    body = _try_reconstruct_expr(body_rel[0]["rhs_tag"] if body_rel else "?", idx, _visited.copy())
    return f"lambda: {body}"


def _reconstruct_dict(dict_id: str, idx: _RelationIndex, _visited: Optional[set] = None) -> str:
    """Reconstruct a dict literal."""
    if _visited is None:
        _visited = set()

    keys = idx.tag("dict_key", lhs=dict_id)
    values = idx.tag("dict_value", lhs=dict_id)

    pairs = []
    for k, v in zip(keys, values):
        k_val = _try_reconstruct_expr(k['rhs_tag'], idx, _visited.copy())
        v_val = _try_reconstruct_expr(v['rhs_tag'], idx, _visited.copy())
        pairs.append(f"{k_val}: {v_val}")

    return "{" + ", ".join(pairs) + "}" if pairs else "{}"


def _reconstruct_await(await_id: str, idx: _RelationIndex, _visited: Optional[set] = None) -> str:
    """Reconstruct an await expression."""
    if _visited is None:
        _visited = set()

    val_rel = idx.tag("await_value", lhs=await_id)
    if val_rel:
        val = _try_reconstruct_expr(str(val_rel[0]["rhs_tag"]), idx, _visited.copy())
        return f"await {val}"
    return "await ..."


def _try_reconstruct_expr(rhs: str, idx: _RelationIndex, _visited: Optional[set] = None) -> str:
    """Try to reconstruct a complex expression from relations."""
    if _visited is None:
        _visited = set()

    # Prevent infinite recursion
    if rhs in _visited:
        return rhs
    _visited.add(rhs)

    import re
    # Pattern for exact reference match (e.g., "binop:16:21" not "binop:16:21.long")
    exact_ref_pattern = r'^(comp|binop|unary|compare|boolop|subscript|ifexp|lambda|dict|await):\d+:\d+:\d+$'

    # Check if rhs is an exact reference to a complex expression
    if re.match(exact_ref_pattern, rhs):
        if rhs.startswith("binop:"):
            return _reconstruct_binop(rhs, idx, _visited)
        if rhs.startswith("unary:"):
            return _reconstruct_unaryop(rhs, idx, _visited)
        if rhs.startswith("compare:"):
            return _reconstruct_compare(rhs, idx, _visited)
        if rhs.startswith("boolop:"):
            return _reconstruct_boolop(rhs, idx, _visited)
        if rhs.startswith("subscript:"):
            return _reconstruct_subscript(rhs, idx, _visited)
        if rhs.startswith("comp:"):
            return _reconstruct_comprehension(rhs, idx, _visited)
        if rhs.startswith("ifexp:"):
            return _reconstruct_ifexp(rhs, idx, _visited)
        if rhs.startswith("lambda:"):
            return _reconstruct_lambda(rhs, idx, _visited)
        if rhs.startswith("dict:"):
            return _reconstruct_dict(rhs, idx, _visited)
        if rhs.startswith("await:"):
            return _reconstruct_await(rhs, idx, _visited)
        if rhs.startswith("await:"):
            return _reconstruct_await(rhs, idx, _visited)

    # Handle embedded references in calls and other expressions
    # e.g., sum(comp:1:7) should resolve the comprehension
    # e.g., binop:16:21.long should resolve the binop
    def replace_ref(match):
        ref = match.group(0)
        return _try_reconstruct_expr(ref, idx, _visited.copy())

    # Match any embedded reference patterns
    pattern = r'(comp:\d+:\d+:\d+|binop:\d+:\d+:\d+|unary:\d+:\d+:\d+|compare:\d+:\d+:\d+|boolop:\d+:\d+:\d+|subscript:\d+:\d+:\d+|ifexp:\d+:\d+:\d+|lambda:\d+:\d+:\d+|dict:\d+:\d+:\d+|await:\d+:\d+:\d+)'
    if re.search(pattern, rhs):
        return re.sub(pattern, replace_ref, rhs)

    return rhs


# ---------------------------------------------------------------------------
# Import reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_imports(idx: _RelationIndex) -> List[str]:
    """Reconstruct import statements, grouping imports from the same module."""
    # Group imports by (line, module) to combine multiple imports from same module
    import_groups: Dict[Tuple[int, str], List[Tuple[str, Optional[str]]]] = {}

    for r in sorted(idx.tag("imports"), key=lambda r: r["line"]):
        module = str(r["lhs_tag"])
        name = str(r["rhs_tag"])
        line = r["line"]

        # Find alias for this import
        alias = None
        for a in idx.tag("aliases", lhs=name):
            if a["line"] == line:
                alias = str(a["rhs_tag"])
                break

        key = (line, module)
        if key not in import_groups:
            import_groups[key] = []
        import_groups[key].append((name, alias))

    lines: List[str] = []
    for (line, module), names in sorted(import_groups.items()):
        # Check if this is a bare import: module == name for all names
        # For "import torch": module="torch", name="torch"
        # For "from typing import List": module="typing", name="List"
        is_bare_import = all(n == module for n, _ in names)

        if is_bare_import:
            # bare "import X"
            for name, alias in names:
                stmt = f"import {name}"
                if alias:
                    stmt += f" as {alias}"
                lines.append(stmt)
        else:
            # "from module import A, B, C"
            parts = []
            for name, alias in names:
                if alias:
                    parts.append(f"{name} as {alias}")
                else:
                    parts.append(name)
            stmt = f"from {module} import {', '.join(parts)}"
            lines.append(stmt)
    return lines


# ---------------------------------------------------------------------------
# Function reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_function(name: str, idx: _RelationIndex, indent: int,
                          define_line: int) -> List[str]:
    pad = "    " * indent
    out: List[str] = []

    # Decorators
    for r in idx.tag("decorates", rhs=name):
        out.append(f"{pad}@{r['lhs_tag']}")

    # Async?
    is_async = bool(idx.tag("defines", rhs=f"async:{name}")) or \
               bool(idx.tag("async_def", lhs=name))
    async_kw = "async " if is_async else ""

    # Build parameter list in emission order
    param_rels = sorted(idx.tag("param", lhs=name), key=lambda r: r["line"])
    star = idx.tag("star_param", lhs=name)
    dstar = idx.tag("double_star_param", lhs=name)
    star_name = str(star[0]["rhs_tag"]) if star else None
    dstar_name = str(dstar[0]["rhs_tag"]) if dstar else None

    ann_map: Dict[str, str] = {}
    for r in idx.tag("annotation"):
        # Resolve the annotation type (e.g., subscript:1 -> List[int])
        ann_val = _try_reconstruct_expr(str(r["rhs_tag"]), idx)
        ann_map[str(r["lhs_tag"])] = ann_val
    def_map: Dict[str, str] = {}
    for r in idx.tag("default_value"):
        def_val = _try_reconstruct_expr(str(r["rhs_tag"]), idx)
        def_map[str(r["lhs_tag"])] = def_val

    params: List[str] = []
    for r in param_rels:
        p = str(r["rhs_tag"])
        ann = f": {ann_map[p]}" if p in ann_map else ""
        # Only use default_value if it's from the same function (same line as param)
        dv_val = None
        for dv in idx.tag("default_value", lhs=p):
            if dv["line"] == r["line"]:
                dv_val = _try_reconstruct_expr(str(dv["rhs_tag"]), idx)
                break
        dv = f" = {dv_val}" if dv_val else ""
        params.append(f"{p}{ann}{dv}")

    if star_name:
        ann = f": {ann_map[star_name]}" if star_name in ann_map else ""
        params.append(f"*{star_name}{ann}")
    if dstar_name:
        ann = f": {ann_map[dstar_name]}" if dstar_name in ann_map else ""
        params.append(f"**{dstar_name}{ann}")

    # Return type annotation (only the one co-located with the definition)
    ret_ann = ""
    for r in idx.tag("returns", lhs=name):
        if r["line"] == define_line:
            ret_type = _try_reconstruct_expr(str(r["rhs_tag"]), idx)
            ret_ann = f" -> {ret_type}"
            break

    out.append(f"{pad}{async_kw}def {name}({', '.join(params)}){ret_ann}:")

    # Ellipsis body?
    if idx.tag("ellipsis_body", lhs=name):
        out.append(f"{pad}    ...")
        return out

    # Docstring?
    doc_rels = idx.tag("docstring", lhs=name)
    if doc_rels:
        doc_sym = doc_rels[0]["rhs_tag"]
        # The docstring is stored as a truncated symbol; just emit placeholder
        inner_pad = "    " * (indent + 1)
        out.append(f'{inner_pad}"""..."""')

    # Body
    body = _reconstruct_body(name, idx, indent + 1, define_line)
    if body:
        out.extend(body)
    elif not doc_rels:
        out.append(f"{pad}    ...")
    return out


# ---------------------------------------------------------------------------
# Class reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_class(name: str, idx: _RelationIndex, indent: int,
                       define_line: int) -> List[str]:
    pad = "    " * indent
    out: List[str] = []

    for r in idx.tag("decorates", rhs=name):
        out.append(f"{pad}@{r['lhs_tag']}")

    bases = [str(r["rhs_tag"]) for r in idx.tag("bases", lhs=name)]
    mc = idx.tag("metaclass", lhs=name)
    # Only include keyword_arg from the same line as the class definition
    # to avoid picking up keyword arguments from function calls inside the class
    kw_args = [str(r["rhs_tag"]) for r in idx.tag("keyword_arg", lhs=name)
               if r["line"] == define_line]

    parts = bases[:]
    if mc:
        parts.append(f"metaclass={mc[0]['rhs_tag']}")
    parts.extend(kw_args)
    base_str = f"({', '.join(parts)})" if parts else ""
    out.append(f"{pad}class {name}{base_str}:")

    # Class-level annotations / assignments
    inner_pad = "    " * (indent + 1)
    body_lines: List[Tuple[int, str]] = []

    # Docstring?
    doc_rels = idx.tag("docstring", lhs=name)
    if doc_rels:
        body_lines.append((doc_rels[0]["line"], f'{inner_pad}"""..."""'))

    # Annotations that belong to this class (not to child functions)
    child_names = set()
    for d in idx.tag("defines", lhs=name):
        n = str(d["rhs_tag"])
        child_names.add(n.replace("async:", ""))

    for r in idx.tag("annotation"):
        lhs = str(r["lhs_tag"])
        if lhs in child_names:
            continue
        # Check if this annotation is inside this class via "contains"
        if idx.tag("contains", lhs=name, rhs=f"stmt:{r['line']}"):
            ann = _try_reconstruct_expr(str(r["rhs_tag"]), idx)
            val_rel = [a for a in idx.tag("assigns", lhs=lhs)
                       if a["line"] == r["line"]]
            if val_rel:
                val = _try_reconstruct_expr(str(val_rel[0]['rhs_tag']), idx)
                body_lines.append((r["line"],
                    f"{inner_pad}{lhs}: {ann} = {val}"))
            else:
                body_lines.append((r["line"], f"{inner_pad}{lhs}: {ann}"))

    # Child definitions
    for d in sorted(idx.tag("defines", lhs=name), key=lambda r: r["line"]):
        child = str(d["rhs_tag"])
        is_async_child = child.startswith("async:")
        real_name = child[6:] if is_async_child else child
        fn_lines = _reconstruct_function(real_name, idx, indent + 1,
                                         d["line"])
        if body_lines or out[-1].endswith(":"):
            body_lines.append((d["line"], ""))
        for fl in fn_lines:
            body_lines.append((d["line"], fl))

    if body_lines:
        for _, line in sorted(body_lines, key=lambda x: x[0]):
            out.append(line)
    else:
        out.append(f"{inner_pad}...")
    return out


# ---------------------------------------------------------------------------
# Body reconstruction — statements inside a scope
# ---------------------------------------------------------------------------

def _reconstruct_body(scope: str, idx: _RelationIndex, indent: int,
                      define_line: int) -> List[str]:
    pad = "    " * indent
    out: List[Tuple[int, int, str]] = []  # (line, seq, statement)
    emitted_lines: set = set()

    # Collect stmt_seq for ordering
    stmt_seqs = {}
    for r in idx.tag("stmt_seq", lhs=scope):
        rhs = str(r["rhs_tag"])
        # Format: "seq:line"
        if ":" in rhs:
            seq_str, line_str = rhs.split(":", 1)
            try:
                seq = int(seq_str)
                ln = int(line_str)
                stmt_seqs[ln] = seq
            except ValueError:
                pass

    def get_seq(ln: int) -> int:
        return stmt_seqs.get(ln, 999999)

    scope_rels = sorted(idx.lhs(scope), key=lambda r: (get_seq(r["line"]), r["line"]))

    for r in scope_rels:
        tag = r["relation_tag"]
        ln = r["line"]

        # Skip structural / already-handled tags
        if tag in ("defines", "contains", "param", "star_param",
                    "double_star_param", "annotation", "default_value",
                    "decorates", "bases", "metaclass", "keyword_arg",
                    "ellipsis_body", "docstring", "async_def",
                    "static_method", "class_method", "property_def",
                    "stmt_seq",
                    # Body structure tags (these are scope markers, not statements)
                    "if_body", "for_body", "while_body", "try_body",
                    "except_body", "with_body", "else_body", "finally_body",
                    # Helper tags for expression reconstruction
                    "bin_op", "bin_op_left", "bin_op_right",
                    "unary_op", "unary_op_operand",
                    "compare_left", "compare_op", "compare_right",
                    "bool_op", "bool_op_operand",
                    "subscript_value", "slice", "slice_lower", "slice_upper", "slice_step",
                    "dict_key", "dict_value",
                    "comprehension_target", "comprehension_body",
                    "lambda_body", "if_expr_body", "if_expr_test", "if_expr_else",
                    "call_arg", "for_tuple_target"):
            continue

        # Return annotation is on the define_line — skip it
        if tag == "returns" and ln == define_line:
            continue

        if ln in emitted_lines:
            continue

        if tag == "returns":
            emitted_lines.add(ln)
            rhs = _try_reconstruct_expr(str(r['rhs_tag']), idx)
            out.append((ln, get_seq(ln), f"{pad}return {rhs}"))

        elif tag == "calls":
            emitted_lines.add(ln)
            # Check if there's an assign on the same line
            assign = [a for a in idx.tag("assigns") if a["line"] == ln
                      and idx.tag("contains", lhs=scope,
                                  rhs=f"stmt:{ln}")]
            callee = _try_reconstruct_expr(str(r['rhs_tag']), idx)
            if assign:
                lhs = _try_reconstruct_expr(str(assign[0]['lhs_tag']), idx)
                out.append((ln, get_seq(ln), f"{pad}{lhs} = "
                                f"{callee}(...)"))
            else:
                out.append((ln, get_seq(ln), f"{pad}{callee}(...)"))

        elif tag == "if_test":
            emitted_lines.add(ln)
            rhs = _try_reconstruct_expr(str(r['rhs_tag']), idx)
            out.append((ln, get_seq(ln), f"{pad}if {rhs}:"))
            # Find the if_body scope and reconstruct it
            body_rels = idx.tag("if_body", lhs=scope)
            body_scope = None
            for br in body_rels:
                if br["line"] == ln:
                    body_scope = str(br["rhs_tag"])
                    break
            if body_scope:
                body_stmts = _reconstruct_body(body_scope, idx, indent + 1, ln)
                for bs in body_stmts:
                    out.append((ln, get_seq(ln), bs))
            else:
                out.append((ln, get_seq(ln), f"{pad}    ..."))
            # Now look for elif_test relations that belong to this if
            # The elif_test should be in the same scope and the line should be
            # the next line after the if_test line
            elif_rels = [er for er in idx.tag("elif_test")
                        if er["line"] > ln and str(er.get("lhs_tag")) == scope]
            for er in elif_rels:
                e_ln = er["line"]
                e_rhs = _try_reconstruct_expr(str(er['rhs_tag']), idx)
                out.append((e_ln, get_seq(ln) + 0.1, f"{pad}elif {e_rhs}:"))
                # Find the if_body for this elif
                if_body_rels = idx.tag("if_body")
                elif_body_scope = None
                for ibr in if_body_rels:
                    if ibr["line"] == e_ln:
                        elif_body_scope = str(ibr["rhs_tag"])
                        break
                if elif_body_scope:
                    elif_stmts = _reconstruct_body(elif_body_scope, idx, indent + 1, e_ln)
                    for es in elif_stmts:
                        out.append((e_ln, get_seq(ln) + 0.1, es))
                else:
                    out.append((e_ln, get_seq(ln) + 0.1, f"{pad}    ..."))
                emitted_lines.add(e_ln)

        elif tag == "elif_test":
            # Skip - handled as part of if_test
            pass

        elif tag == "for_target":
            emitted_lines.add(ln)
            iter_rel = next((x for x in idx.tag("for_iter")
                            if x["line"] == ln), None)
            it = _try_reconstruct_expr(str(iter_rel["rhs_tag"]), idx) if iter_rel else "..."
            out.append((ln, get_seq(ln), f"{pad}for {r['rhs_tag']} in {it}:"))
            # Find the for_body scope and reconstruct it
            body_rels = idx.tag("for_body", lhs=scope)
            body_scope = None
            for br in body_rels:
                if br["line"] == ln:
                    body_scope = str(br["rhs_tag"])
                    break
            if body_scope:
                body_stmts = _reconstruct_body(body_scope, idx, indent + 1, ln)
                for bs in body_stmts:
                    out.append((ln, get_seq(ln), bs))
            else:
                out.append((ln, get_seq(ln), f"{pad}    ..."))

        elif tag == "while_test":
            emitted_lines.add(ln)
            rhs = _try_reconstruct_expr(str(r['rhs_tag']), idx)
            out.append((ln, get_seq(ln), f"{pad}while {rhs}:"))
            # Find the while_body scope and reconstruct it
            body_rels = idx.tag("while_body", lhs=scope)
            body_scope = None
            for br in body_rels:
                if br["line"] == ln:
                    body_scope = str(br["rhs_tag"])
                    break
            if body_scope:
                body_stmts = _reconstruct_body(body_scope, idx, indent + 1, ln)
                for bs in body_stmts:
                    out.append((ln, get_seq(ln), bs))
            else:
                out.append((ln, get_seq(ln), f"{pad}    ..."))

        elif tag == "with_context":
            emitted_lines.add(ln)
            as_rel = next((x for x in idx.tag("with_as")
                          if x["line"] == ln
                          and str(x["lhs_tag"]) == str(r["rhs_tag"])),
                         None)
            as_part = f" as {as_rel['rhs_tag']}" if as_rel else ""
            out.append((ln, get_seq(ln), f"{pad}with {r['rhs_tag']}{as_part}:"))
            # Find the with_body scope and reconstruct it
            body_rels = idx.tag("with_body")
            body_scope = None
            for br in body_rels:
                if br["line"] == ln:
                    body_scope = str(br["rhs_tag"])
                    break
            if body_scope:
                body_stmts = _reconstruct_body(body_scope, idx, indent + 1, ln)
                for bs in body_stmts:
                    out.append((ln, get_seq(ln), bs))
            else:
                out.append((ln, get_seq(ln), f"{pad}    ..."))

        elif tag == "handles":
            # Skip - handled as part of try_start
            pass

        elif tag == "try_start":
            emitted_lines.add(ln)
            out.append((ln, get_seq(ln), f"{pad}try:"))
            # Find the try_body scope and reconstruct it
            body_rels = idx.tag("try_body")
            body_scope = None
            for br in body_rels:
                if br["line"] == ln:
                    body_scope = str(br["rhs_tag"])
                    break
            if body_scope:
                body_stmts = _reconstruct_body(body_scope, idx, indent + 1, ln)
                for bs in body_stmts:
                    out.append((ln, get_seq(ln), bs))
            else:
                out.append((ln, get_seq(ln), f"{pad}    ..."))
            # Now find and emit all handlers for this try block
            # Look for handles relations that are in the same parent scope
            handles_rels = idx.tag("handles")
            for h in handles_rels:
                # Check if this handler belongs to this try
                # by checking if the handles line is greater than the try line
                # and there's no other try_start between them
                h_ln = h["line"]
                if h_ln > ln:
                    # Check if this is the next handler after this try
                    as_rel = next((x for x in idx.tag("except_as")
                                  if x["line"] == h_ln), None)
                    as_part = f" as {as_rel['rhs_tag']}" if as_rel else ""
                    out.append((h_ln, get_seq(ln) + 0.5, f"{pad}except {h['rhs_tag']}{as_part}:"))
                    # Find the except_body scope and reconstruct it
                    except_body_rels = idx.tag("except_body")
                    except_scope = None
                    for ebr in except_body_rels:
                        if ebr["line"] == h_ln:
                            except_scope = str(ebr["rhs_tag"])
                            break
                    if except_scope:
                        except_stmts = _reconstruct_body(except_scope, idx, indent + 1, h_ln)
                        for es in except_stmts:
                            out.append((h_ln, get_seq(ln) + 0.5, es))
                    else:
                        out.append((h_ln, get_seq(ln) + 0.5, f"{pad}    ..."))
                    emitted_lines.add(h_ln)
            # Check for finally_body at the same line as try_start
            finally_rels = idx.tag("finally_body")
            for fr in finally_rels:
                if fr["line"] == ln:
                    finally_scope = str(fr["rhs_tag"])
                    out.append((ln, get_seq(ln) + 0.6, f"{pad}finally:"))
                    if finally_scope:
                        finally_stmts = _reconstruct_body(finally_scope, idx, indent + 1, ln)
                        for fs in finally_stmts:
                            out.append((ln, get_seq(ln) + 0.6, fs))
                    else:
                        out.append((ln, get_seq(ln) + 0.6, f"{pad}    ..."))
                    break

        elif tag == "try_else":
            emitted_lines.add(ln)
            out.append((ln, get_seq(ln), f"{pad}else:"))
            # Find the else_body scope and reconstruct it
            body_rels = idx.tag("else_body")
            body_scope = None
            for br in body_rels:
                if br["line"] == ln:
                    body_scope = str(br["rhs_tag"])
                    break
            if body_scope:
                body_stmts = _reconstruct_body(body_scope, idx, indent + 1, ln)
                for bs in body_stmts:
                    out.append((ln, get_seq(ln), bs))
            else:
                out.append((ln, get_seq(ln), f"{pad}    ..."))

        elif tag == "try_finally":
            emitted_lines.add(ln)
            out.append((ln, get_seq(ln), f"{pad}finally:"))
            # Find the finally_body scope and reconstruct it
            body_rels = idx.tag("finally_body")
            body_scope = None
            for br in body_rels:
                if br["line"] == ln:
                    body_scope = str(br["rhs_tag"])
                    break
            if body_scope:
                body_stmts = _reconstruct_body(body_scope, idx, indent + 1, ln)
                for bs in body_stmts:
                    out.append((ln, get_seq(ln), bs))
            else:
                out.append((ln, get_seq(ln), f"{pad}    ..."))

        elif tag == "raises":
            emitted_lines.add(ln)
            from_rel = next((x for x in idx.tag("raises_from")
                           if x["line"] == ln), None)
            fr = f" from {from_rel['rhs_tag']}" if from_rel else ""
            out.append((ln, get_seq(ln), f"{pad}raise {r['rhs_tag']}{fr}"))

        elif tag == "yields":
            emitted_lines.add(ln)
            rhs = _try_reconstruct_expr(str(r['rhs_tag']), idx)
            out.append((ln, get_seq(ln), f"{pad}yield {rhs}"))

        elif tag == "yields_from":
            emitted_lines.add(ln)
            rhs = _try_reconstruct_expr(str(r['rhs_tag']), idx)
            out.append((ln, get_seq(ln), f"{pad}yield from {rhs}"))

        elif tag == "global_decl":
            emitted_lines.add(ln)
            out.append((ln, get_seq(ln), f"{pad}global {r['rhs_tag']}"))

        elif tag == "nonlocal_decl":
            emitted_lines.add(ln)
            out.append((ln, get_seq(ln), f"{pad}nonlocal {r['rhs_tag']}"))

        elif tag == "assert_test":
            emitted_lines.add(ln)
            msg_rel = next((x for x in idx.tag("assert_msg", lhs=scope)
                          if x["line"] == ln), None)
            msg = f", {msg_rel['rhs_tag']}" if msg_rel else ""
            rhs = _try_reconstruct_expr(str(r['rhs_tag']), idx)
            out.append((ln, get_seq(ln), f"{pad}assert {rhs}{msg}"))

        elif tag == "compare":
            # This is a standalone compare expression
            emitted_lines.add(ln)
            compare_id = f"compare:{ln}"
            expr = _reconstruct_compare(compare_id, idx)
            out.append((ln, get_seq(ln), f"{pad}{expr}"))

        elif tag == "unary_op":
            emitted_lines.add(ln)
            unary_id = f"unary:{ln}"
            expr = _reconstruct_unaryop(unary_id, idx)
            out.append((ln, get_seq(ln), f"{pad}{expr}"))

        elif tag == "bin_op":
            emitted_lines.add(ln)
            binop_id = f"binop:{ln}"
            expr = _reconstruct_binop(binop_id, idx)
            out.append((ln, get_seq(ln), f"{pad}{expr}"))

        elif tag == "if_expr":
            emitted_lines.add(ln)
            ifexp_id = f"ifexp:{ln}"
            expr = _reconstruct_ifexp(ifexp_id, idx)
            out.append((ln, get_seq(ln), f"{pad}{expr}"))

        elif tag == "lambda":
            emitted_lines.add(ln)
            lambda_id = f"lambda:{ln}"
            expr = _reconstruct_lambda(lambda_id, idx)
            out.append((ln, get_seq(ln), f"{pad}{expr}"))

        elif tag == "subscript":
            emitted_lines.add(ln)
            subscript_id = f"subscript:{ln}"
            expr = _reconstruct_subscript(subscript_id, idx)
            out.append((ln, get_seq(ln), f"{pad}{expr}"))

        elif tag == "dict_literal":
            emitted_lines.add(ln)
            dict_id = f"dict:{ln}"
            expr = _reconstruct_dict(dict_id, idx)
            # Check if there's an assignment
            assign = [a for a in idx.tag("assigns") if a["line"] == ln]
            if assign:
                lhs = _try_reconstruct_expr(str(assign[0]['lhs_tag']), idx)
                out.append((ln, get_seq(ln), f"{pad}{lhs} = {expr}"))

        elif tag == "set_literal":
            emitted_lines.add(ln)
            set_id = f"set:{ln}"
            # Check if there's an assignment
            assign = [a for a in idx.tag("assigns") if a["line"] == ln]
            if assign:
                lhs = _try_reconstruct_expr(str(assign[0]['lhs_tag']), idx)
                out.append((ln, get_seq(ln), f"{pad}{lhs} = {{...}}"))

        elif tag == "list_literal":
            emitted_lines.add(ln)
            list_id = f"list:{ln}"
            # Check if there's an assignment
            assign = [a for a in idx.tag("assigns") if a["line"] == ln]
            if assign:
                lhs = _try_reconstruct_expr(str(assign[0]['lhs_tag']), idx)
                out.append((ln, get_seq(ln), f"{pad}{lhs} = [...]"))

        elif tag == "tuple_literal":
            emitted_lines.add(ln)
            tuple_id = f"tuple:{ln}"
            # Check if there's an assignment
            assign = [a for a in idx.tag("assigns") if a["line"] == ln]
            if assign:
                lhs = _try_reconstruct_expr(str(assign[0]['lhs_tag']), idx)
                out.append((ln, get_seq(ln), f"{pad}{lhs} = (...)"))

        elif tag == "named_expr":
            emitted_lines.add(ln)
            # NamedExpr is usually inside an if_test, skip separate emission
            pass

        elif tag == "pass_stmt":
            emitted_lines.add(ln)
            out.append((ln, get_seq(ln), f"{pad}pass"))

        elif tag == "break_stmt":
            emitted_lines.add(ln)
            out.append((ln, get_seq(ln), f"{pad}break"))

        elif tag == "continue_stmt":
            emitted_lines.add(ln)
            out.append((ln, get_seq(ln), f"{pad}continue"))

        elif tag == "del_target":
            emitted_lines.add(ln)
            out.append((ln, get_seq(ln), f"{pad}del {r['rhs_tag']}"))

        elif tag == "expr_stmt":
            emitted_lines.add(ln)
            rhs = _try_reconstruct_expr(str(r['rhs_tag']), idx)
            out.append((ln, get_seq(ln), f"{pad}{rhs}"))

    # Assignments inside this scope not yet emitted
    for r in idx.tag("assigns"):
        ln = r["line"]
        if ln in emitted_lines:
            continue
        if idx.tag("contains", lhs=scope, rhs=f"stmt:{ln}"):
            emitted_lines.add(ln)
            lhs = _try_reconstruct_expr(str(r['lhs_tag']), idx)
            rhs = _try_reconstruct_expr(str(r['rhs_tag']), idx)
            out.append((ln, get_seq(ln), f"{pad}{lhs} = {rhs}"))

    # Augmented assignments
    for r in idx.tag("aug_assigns"):
        ln = r["line"]
        if ln in emitted_lines:
            continue
        if idx.tag("contains", lhs=scope, rhs=f"stmt:{ln}"):
            emitted_lines.add(ln)
            op_rel = idx.tag("aug_op", lhs=str(r['lhs_tag']))
            op = AUG_OP_MAP.get(op_rel[0]["rhs_tag"], op_rel[0]["rhs_tag"]) if op_rel else "+="
            lhs = _try_reconstruct_expr(str(r['lhs_tag']), idx)
            rhs = _try_reconstruct_expr(str(r['rhs_tag']), idx)
            out.append((ln, get_seq(ln), f"{pad}{lhs} {op} {rhs}"))

    out.sort(key=lambda x: (x[1], x[0]))  # Sort by seq, then by line
    return [line for _, _, line in out]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_ast_tag_jsonl_to_python(jsonl: str) -> str:
    """Convert JSONL of AstTagRelation records to Python source code.

    <- $jsonl JsonLines[AstTagRelation[Python]]
    -> source_code[Python]
    """
    relations = []
    for line in jsonl.strip().splitlines():
        line = line.strip()
        if line:
            relations.append(json.loads(line))
    if not relations:
        return ""

    idx = _RelationIndex(relations)
    output: List[str] = []

    # Module-level docstring
    module_doc = idx.tag("docstring", lhs="<module>")
    if module_doc:
        output.append('"""..."""')
        output.append("")

    # Imports
    import_lines = _reconstruct_imports(idx)
    if import_lines:
        output.extend(import_lines)
        output.append("")

    # Collect all top-level items with their lines
    top_level_items: List[Tuple[int, str, str]] = []  # (line, type, content)

    # Top-level definitions (sorted by line)
    for d in sorted(idx.tag("defines", lhs="<module>"), key=lambda r: r["line"]):
        name = str(d["rhs_tag"])
        is_async = name.startswith("async:")
        real_name = name[6:] if is_async else name

        # Check if it's a class: has bases, metaclass, OR has annotations but no params
        is_class = bool(idx.tag("bases", lhs=real_name)
                        or idx.tag("metaclass", lhs=real_name))
        if not is_class:
            # Heuristic: has child defines but no params → class
            child_defs = idx.tag("defines", lhs=real_name)
            child_params = idx.tag("param", lhs=real_name)
            if child_defs and not child_params:
                is_class = True
            # Also check: has annotations but no params (dataclass-style class)
            if not child_params:
                # Check for class-level annotations via contains
                class_contains = idx.tag("contains", lhs=real_name)
                for c in class_contains:
                    # Check if there are annotation relations for this scope
                    ann_rels = [r for r in idx.tag("annotation")
                               if idx.tag("contains", lhs=real_name, rhs=f"stmt:{r['line']}")]
                    if ann_rels:
                        is_class = True
                        break

        if is_class:
            lines = _reconstruct_class(real_name, idx, 0, d["line"])
        else:
            lines = _reconstruct_function(real_name, idx, 0, d["line"])

        for line in lines:
            top_level_items.append((d["line"], "def", line))

    # Top-level assignments not inside any scope
    contained = set()
    for r in idx.tag("contains"):
        contained.add(r["rhs_tag"])

    emitted_lines = set()

    for r in idx.tag("assigns"):
        ln = r["line"]
        if f"stmt:{ln}" in contained:
            continue
        if ln in emitted_lines:
            continue
        emitted_lines.add(ln)
        rhs = _try_reconstruct_expr(str(r['rhs_tag']), idx)
        top_level_items.append((ln, "assign", f"{r['lhs_tag']} = {rhs}"))

    # Top-level augmented assignments
    for r in idx.tag("aug_assigns"):
        ln = r["line"]
        if f"stmt:{ln}" in contained:
            continue
        if ln in emitted_lines:
            continue
        emitted_lines.add(ln)
        op_rel = idx.tag("aug_op", lhs=str(r['lhs_tag']))
        op = AUG_OP_MAP.get(op_rel[0]["rhs_tag"], op_rel[0]["rhs_tag"]) if op_rel else "+="
        rhs = _try_reconstruct_expr(str(r['rhs_tag']), idx)
        top_level_items.append((ln, "aug_assign", f"{r['lhs_tag']} {op} {rhs}"))

    # Top-level if statements
    for r in idx.tag("if_test", lhs="<module>"):
        ln = r["line"]
        if ln in emitted_lines:
            continue
        emitted_lines.add(ln)
        rhs = _try_reconstruct_expr(str(r['rhs_tag']), idx)
        top_level_items.append((ln, "if", f"if {rhs}:"))
        top_level_items.append((ln, "if", "    ..."))

    # Top-level for statements
    for r in idx.tag("for_target", lhs="<module>"):
        ln = r["line"]
        if ln in emitted_lines:
            continue
        emitted_lines.add(ln)
        iter_rel = next((x for x in idx.tag("for_iter", lhs="<module>")
                        if x["line"] == ln), None)
        it = _try_reconstruct_expr(str(iter_rel["rhs_tag"]), idx) if iter_rel else "..."
        top_level_items.append((ln, "for", f"for {r['rhs_tag']} in {it}:"))
        top_level_items.append((ln, "for", "    ..."))

    # Top-level while statements
    for r in idx.tag("while_test", lhs="<module>"):
        ln = r["line"]
        if ln in emitted_lines:
            continue
        emitted_lines.add(ln)
        rhs = _try_reconstruct_expr(str(r['rhs_tag']), idx)
        top_level_items.append((ln, "while", f"while {rhs}:"))
        top_level_items.append((ln, "while", "    ..."))

    # Top-level calls (expression statements)
    for r in idx.tag("calls", lhs="<module>"):
        ln = r["line"]
        if ln in emitted_lines:
            continue
        if f"stmt:{ln}" in contained:
            continue
        emitted_lines.add(ln)
        top_level_items.append((ln, "call", f"{r['rhs_tag']}(...)"))

    # Sort by line and emit
    top_level_items.sort(key=lambda x: (x[0], 0 if x[1] == "def" else 1))

    # Group consecutive items from the same definition
    prev_type = None
    for ln, item_type, content in top_level_items:
        if item_type == "def" and prev_type != "def":
            output.append("")
        output.append(content)
        prev_type = item_type

    return "\n".join(output) + "\n"


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    from convert_python_to_ast_tag_jsonl import (
        convert_python_to_ast_tag_jsonl, ast_to_dict,
    )

    CODEBASE_DIR = os.path.join(here, "..", "code_auto_encoder", "codebase")

    # Collect all .py files
    py_files = []
    for root, _dirs, files in os.walk(CODEBASE_DIR):
        for f in sorted(files):
            if f.endswith(".py") and f != "__init__.py":
                py_files.append(os.path.join(root, f))
    py_files.sort()

    print("=== Roundtrip test: source.py → JSONL → reconstructed.py ===\n")
    for path in py_files:
        with open(path, "r", encoding="utf-8") as fh:
            source = fh.read()
        rel = os.path.relpath(path, CODEBASE_DIR)

        # Step 1: source → AST dict → JSONL
        tree = ast_mod.parse(source, filename=path)
        ast_obj = ast_to_dict(tree)
        jsonl = convert_python_to_ast_tag_jsonl(ast_obj)
        n = len(jsonl.strip().splitlines()) if jsonl.strip() else 0

        # Step 2: JSONL → Python
        reconstructed = convert_ast_tag_jsonl_to_python(jsonl)
        r_lines = len(reconstructed.strip().splitlines()) if reconstructed.strip() else 0
        s_lines = len(source.strip().splitlines())

        print(f"{rel}: {s_lines} src lines → {n} relations → {r_lines} reconstructed lines")

    # Show one detailed example
    sample = py_files[0] if py_files else None
    if sample:
        with open(sample, "r", encoding="utf-8") as fh:
            source = fh.read()
        tree = ast_mod.parse(source, filename=sample)
        ast_obj = ast_to_dict(tree)
        jsonl = convert_python_to_ast_tag_jsonl(ast_obj)
        reconstructed = convert_ast_tag_jsonl_to_python(jsonl)
        print(f"\n--- Detail: {os.path.relpath(sample, CODEBASE_DIR)} ---")
        print("Original:")
        print(source)
        print("Reconstructed:")
        print(reconstructed)
