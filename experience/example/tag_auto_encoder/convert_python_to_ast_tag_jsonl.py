"""Convert Python AST JSON to AstTagRelation JSONL.

Generated from convert_python_to_ast_tag_jsonl.viba.

Viba DSL specification:
  convert_python_to_ast_tag_jsonl[ProgrammingLanguage] :=
    JsonLines[AstTagRelation[ProgrammingLanguage]]
    <- $ast_obj ast[ProgrammingLanguage]
    # inline
    <- Import[./ast_tag_relation_group.viba]
"""

import ast
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# ast.AST → JSON dict conversion
# ---------------------------------------------------------------------------

def ast_to_dict(node: Any) -> Any:
    """Convert an ast.AST tree to the JSON dict format used by the ast/ directory.

    Handles AST nodes, lists, and primitive values recursively.
    """
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
    # Primitives: str, int, float, bool, None, bytes, Ellipsis
    return node


# ---------------------------------------------------------------------------
# Symbol extraction — convert JSON AST node to compact Symbol string
# ---------------------------------------------------------------------------

def _node_to_symbol(node: Any, expr_id: Optional[str] = None) -> str:
    """Convert a JSON AST node to a Symbol (no whitespace).

    If expr_id is provided for complex expressions, returns a reference ID
    instead of embedding operators in the symbol.
    """
    if node is None:
        return "None"
    if isinstance(node, (str, int, float, bool)):
        return repr(node).replace(" ", "")
    if not isinstance(node, dict):
        return str(node).replace(" ", "")
    t = node.get("_type", "")
    # Generate unique ID using line and column
    ln = node.get("_lineno", 0)
    col = node.get("_col_offset", 0)
    end_col = node.get("_end_col_offset", col + 1)
    uid = f"{ln}:{col}:{end_col}" if ln else "0:0:1"

    if t == "Name":
        return node["id"]
    if t == "Attribute":
        return f"{_node_to_symbol(node['value'])}.{node['attr']}"
    if t == "Constant":
        v = node["value"]
        if isinstance(v, str) and len(v) > 40:
            return repr(v[:37] + "...").replace(" ", "")
        return repr(v).replace(" ", "")
    if t == "Starred":
        return f"*{_node_to_symbol(node['value'])}"
    if t == "Subscript":
        # Return a reference to the subscript
        return f"subscript:{uid}"
    if t == "Tuple":
        elts = ",".join(_node_to_symbol(e) for e in node.get("elts", []))
        return f"({elts})"
    if t == "List":
        elts = ",".join(_node_to_symbol(e) for e in node.get("elts", []))
        return f"[{elts}]"
    if t == "Call":
        func = _node_to_symbol(node["func"])
        args_parts = [_node_to_symbol(a) for a in node.get("args", [])]
        for kw in node.get("keywords", []):
            if kw.get("arg"):
                args_parts.append(f"{kw['arg']}={_node_to_symbol(kw['value'])}")
            else:
                args_parts.append(f"**{_node_to_symbol(kw['value'])}")
        return f"{func}({','.join(args_parts)})"
    if t == "BinOp":
        # Return a reference to the binary operation
        return f"binop:{uid}"
    if t == "UnaryOp":
        # Return a reference to the unary operation
        return f"unary:{uid}"
    if t == "BoolOp":
        # Return a reference to the boolean operation
        return f"boolop:{uid}"
    if t == "Compare":
        # Return a reference to the comparison
        return f"compare:{uid}"
    if t == "Await":
        # Await expression - return the awaited value
        return f"await:{uid}"
    if t == "Slice":
        lo = _node_to_symbol(node["lower"]) if node.get("lower") else ""
        hi = _node_to_symbol(node["upper"]) if node.get("upper") else ""
        st = _node_to_symbol(node["step"]) if node.get("step") else ""
        return f"{lo}:{hi}:{st}" if st else f"{lo}:{hi}"
    if t == "JoinedStr":
        return "f-string"
    if t == "FormattedValue":
        return _node_to_symbol(node.get("value"))
    if t == "arg":
        return node["arg"]
    if t == "alias":
        return node["name"]
    if t == "IfExp":
        return f"ifexp:{uid}"
    if t == "Lambda":
        return f"lambda:{uid}"
    if t == "Dict":
        return f"dict:{uid}"
    if t == "Set":
        return f"set:{uid}"
    if t == "NamedExpr":
        return f"namedexpr:{uid}"
    if t in ("ListComp", "SetComp", "GeneratorExp", "DictComp"):
        return f"comp:{uid}"
    return f"<{t}>"


# ---------------------------------------------------------------------------
# Relation emitting
# ---------------------------------------------------------------------------

def _ln(node: Any) -> int:
    if isinstance(node, dict):
        return node.get("_lineno", 0)
    return 0


def _emit(rels: List[Dict], line: int, tag: str, lhs: str, rhs: str,
          rig: Optional[Dict] = None):
    rels.append({
        "line": line,
        "relation_in_group": rig,
        "relation_tag": tag,
        "lhs_tag": lhs,
        "rhs_tag": rhs,
    })


def _get_docstring(body: List[Dict]) -> Optional[Tuple[int, str]]:
    """Extract docstring from body if present, returns (line, docstring) or None."""
    if not body:
        return None
    first = body[0]
    if first.get("_type") != "Expr":
        return None
    val = first.get("value", {})
    if val.get("_type") != "Constant":
        return None
    doc = val.get("value")
    if not isinstance(doc, str):
        return None
    return _ln(first), doc


# ---------------------------------------------------------------------------
# Expression scanner — find calls, comprehensions, fstrings, walrus, etc.
# inside an expression subtree without emitting statement-level relations.
# ---------------------------------------------------------------------------

def _scan_expr(node: Any, scope: str, rels: List[Dict]):
    """Scan an expression tree for nested relational patterns."""
    if not isinstance(node, dict) or "_type" not in node:
        return
    t = node["_type"]
    ln = _ln(node)
    col = node.get("_col_offset", 0)
    end_col = node.get("_end_col_offset", col + 1)
    uid = f"{ln}:{col}:{end_col}" if ln else "0:0:1"

    if t == "Call":
        callee = _node_to_symbol(node["func"])
        _emit(rels, ln, "calls", scope, callee)
        # Emit positional arguments for reconstruction
        for i, arg in enumerate(node.get("args", [])):
            if isinstance(arg, dict) and arg.get("_type") == "Starred":
                _emit(rels, ln, "star_arg", callee,
                      _node_to_symbol(arg["value"]))
            else:
                _emit(rels, ln, "call_arg", callee, _node_to_symbol(arg))
        for kw in node.get("keywords", []):
            if kw.get("arg"):
                _emit(rels, ln, "keyword_arg", callee,
                      f"{kw['arg']}={_node_to_symbol(kw['value'])}")
            else:
                _emit(rels, ln, "double_star_arg", callee,
                      _node_to_symbol(kw["value"]))
        # recurse into func, args, keywords
        _scan_expr(node.get("func"), scope, rels)
        for a in node.get("args", []):
            _scan_expr(a, scope, rels)
        for kw in node.get("keywords", []):
            _scan_expr(kw.get("value"), scope, rels)
        return

    if t == "Attribute" and node.get("ctx", {}).get("_type") == "Load":
        _emit(rels, ln, "member_access", _node_to_symbol(node["value"]),
              node["attr"])

    if t == "BinOp":
        left = _node_to_symbol(node["left"])
        right = _node_to_symbol(node["right"])
        op = node["op"]["_type"]
        binop_id = f"binop:{uid}"
        # bin_op: op -> binop_id, then bin_op_left/right -> operands
        _emit(rels, ln, "bin_op", op, binop_id)
        _emit(rels, ln, "bin_op_left", binop_id, left)
        _emit(rels, ln, "bin_op_right", binop_id, right)
        _scan_expr(node.get("left"), scope, rels)
        _scan_expr(node.get("right"), scope, rels)
        return

    if t == "UnaryOp":
        operand = _node_to_symbol(node["operand"])
        op = node["op"]["_type"]
        unary_id = f"unary:{uid}"
        _emit(rels, ln, "unary_op", op, unary_id)
        _emit(rels, ln, "unary_op_operand", unary_id, operand)
        _scan_expr(node.get("operand"), scope, rels)
        return

    if t == "Compare":
        left = _node_to_symbol(node["left"])
        compare_id = f"compare:{uid}"
        _emit(rels, ln, "compare", left, "...")
        _emit(rels, ln, "compare_left", compare_id, left)
        _scan_expr(node.get("left"), scope, rels)  # Scan left for nested expressions
        for i, (op, comp) in enumerate(zip(node.get("ops", []), node.get("comparators", []))):
            _emit(rels, ln, "compare_op", compare_id, op["_type"])
            rhs = _node_to_symbol(comp)
            _emit(rels, ln, "compare_right", compare_id, rhs)
            _scan_expr(comp, scope, rels)
        return

    if t == "BoolOp":
        op = node["op"]["_type"]  # And / Or
        vals = [_node_to_symbol(v) for v in node.get("values", [])]
        boolop_id = f"boolop:{uid}"
        _emit(rels, ln, "bool_op", op, boolop_id)
        for i, v in enumerate(node.get("values", [])):
            _emit(rels, ln, "bool_op_operand", boolop_id, _node_to_symbol(v))
            _scan_expr(v, scope, rels)
        return

    if t == "Subscript":
        value = _node_to_symbol(node["value"])
        slice_node = node.get("slice")
        slice_sym = _node_to_symbol(slice_node)
        sub_id = f"subscript:{uid}"
        # Emit subscript with ID as lhs for easy reconstruction
        _emit(rels, ln, "subscript", sub_id, slice_sym)
        _emit(rels, ln, "subscript_value", sub_id, value)
        if isinstance(slice_node, dict) and slice_node.get("_type") == "Slice":
            _emit(rels, ln, "slice", sub_id, slice_sym)
            if slice_node.get("lower"):
                _emit(rels, ln, "slice_lower", sub_id, _node_to_symbol(slice_node["lower"]))
                _scan_expr(slice_node["lower"], scope, rels)
            if slice_node.get("upper"):
                _emit(rels, ln, "slice_upper", sub_id, _node_to_symbol(slice_node["upper"]))
                _scan_expr(slice_node["upper"], scope, rels)
            if slice_node.get("step"):
                _emit(rels, ln, "slice_step", sub_id, _node_to_symbol(slice_node["step"]))
                _scan_expr(slice_node["step"], scope, rels)
        else:
            # Non-slice subscript, just scan the index expression
            _scan_expr(slice_node, scope, rels)
        _scan_expr(node.get("value"), scope, rels)
        return

    if t == "Dict":
        keys = node.get("keys", [])
        vals = node.get("values", [])
        dict_id = f"dict:{uid}"
        _emit(rels, ln, "dict_literal", scope, dict_id)
        for i, (k, v) in enumerate(zip(keys, vals)):
            if k is None:
                _emit(rels, ln, "dict_value", dict_id, f"**{_node_to_symbol(v)}")
            else:
                _emit(rels, ln, "dict_key", dict_id, _node_to_symbol(k))
                _emit(rels, ln, "dict_value", dict_id, _node_to_symbol(v))
            _scan_expr(v, scope, rels)
        return

    if t == "Set":
        set_id = f"set:{uid}"
        _emit(rels, ln, "set_literal", scope, set_id)
        for e in node.get("elts", []):
            _scan_expr(e, scope, rels)
        return

    if t == "List":
        if node.get("ctx", {}).get("_type") == "Load":
            list_id = f"list:{uid}"
            _emit(rels, ln, "list_literal", scope, list_id)
        for e in node.get("elts", []):
            _scan_expr(e, scope, rels)
        return

    if t == "Tuple":
        if node.get("ctx", {}).get("_type") == "Load":
            tuple_id = f"tuple:{uid}"
            _emit(rels, ln, "tuple_literal", scope, tuple_id)
        for e in node.get("elts", []):
            _scan_expr(e, scope, rels)
        return

    if t == "NamedExpr":
        _emit(rels, ln, "named_expr", _node_to_symbol(node["target"]),
              _node_to_symbol(node["value"]))
        _emit(rels, ln, "walrus", _node_to_symbol(node["target"]),
              _node_to_symbol(node["value"]))
        _scan_expr(node.get("value"), scope, rels)
        return

    if t == "Await":
        await_id = f"await:{uid}"
        _emit(rels, ln, "await_expr", scope, await_id)
        _emit(rels, ln, "await_value", await_id, _node_to_symbol(node.get("value")))
        _scan_expr(node.get("value"), scope, rels)
        return

    if t == "IfExp":
        body = _node_to_symbol(node["body"])
        test = _node_to_symbol(node["test"])
        else_ = _node_to_symbol(node["orelse"])
        ifexp_id = f"ifexp:{uid}"
        _emit(rels, ln, "if_expr", body, test)
        _emit(rels, ln, "if_expr_body", ifexp_id, body)
        _emit(rels, ln, "if_expr_test", ifexp_id, test)
        _emit(rels, ln, "if_expr_else", ifexp_id, else_)
        _scan_expr(node.get("body"), scope, rels)
        _scan_expr(node.get("test"), scope, rels)
        _scan_expr(node.get("orelse"), scope, rels)
        return

    if t == "Lambda":
        lambda_id = f"lambda:{uid}"
        _emit(rels, ln, "lambda", scope, lambda_id)
        if node.get("body"):
            _emit(rels, ln, "lambda_body", lambda_id, _node_to_symbol(node["body"]))
            _scan_expr(node["body"], scope, rels)  # Scan body for nested expressions
        return

    if t == "JoinedStr":
        for val in node.get("values", []):
            if isinstance(val, dict) and val.get("_type") == "FormattedValue":
                _emit(rels, ln, "fstring_expr", scope,
                      _node_to_symbol(val["value"]))
        return

    if t in ("ListComp", "SetComp", "GeneratorExp", "DictComp"):
        comp_id = f"comp:{uid}"
        # Emit comprehension body (the output expression)
        if t == "ListComp":
            _emit(rels, ln, "comprehension_body", comp_id,
                  _node_to_symbol(node.get("elt")))
            _scan_expr(node.get("elt"), comp_id, rels)
        elif t == "SetComp":
            _emit(rels, ln, "comprehension_body", comp_id,
                  _node_to_symbol(node.get("elt")))
            _scan_expr(node.get("elt"), comp_id, rels)
        elif t == "GeneratorExp":
            _emit(rels, ln, "comprehension_body", comp_id,
                  _node_to_symbol(node.get("elt")))
            _scan_expr(node.get("elt"), comp_id, rels)
        elif t == "DictComp":
            _emit(rels, ln, "comprehension_body", comp_id,
                  f"{_node_to_symbol(node.get('key'))}:{_node_to_symbol(node.get('value'))}")
            _scan_expr(node.get("key"), comp_id, rels)
            _scan_expr(node.get("value"), comp_id, rels)
        for gen in node.get("generators", []):
            _emit(rels, ln, "comprehension_iter", comp_id,
                  _node_to_symbol(gen["iter"]))
            _emit(rels, ln, "comprehension_target", comp_id,
                  _node_to_symbol(gen["target"]))
            _emit(rels, ln, "for_target", comp_id,
                  _node_to_symbol(gen["target"]))
            _scan_expr(gen.get("iter"), comp_id, rels)
            for if_ in gen.get("ifs", []):
                _emit(rels, ln, "comprehension_if", comp_id,
                      _node_to_symbol(if_))
                _scan_expr(if_, comp_id, rels)
        return

    # Generic recurse into expression children
    for key in ("value", "func", "left", "right", "operand", "test",
                "body", "orelse", "slice"):
        child = node.get(key)
        if isinstance(child, dict) and "_type" in child:
            _scan_expr(child, scope, rels)
        elif isinstance(child, list):
            for item in child:
                if isinstance(item, dict) and "_type" in item:
                    _scan_expr(item, scope, rels)
    for key in ("elts", "args", "values", "comparators", "keys"):
        for item in node.get(key, []):
            if isinstance(item, dict) and "_type" in item:
                _scan_expr(item, scope, rels)
    for kw in node.get("keywords", []):
        _scan_expr(kw.get("value"), scope, rels)


# ---------------------------------------------------------------------------
# Statement-level visitors — walk JSON AST body lists
# ---------------------------------------------------------------------------

def _visit_stmt(node: Dict, scope: str, rels: List[Dict], stmt_seq: int = 0):
    """Visit a single statement node and emit its relations."""
    t = node.get("_type", "")
    ln = _ln(node)

    # Emit statement sequence for ordering
    _emit(rels, ln, "stmt_seq", scope, f"{stmt_seq}:{ln}")
    _emit(rels, ln, "contains", scope, f"stmt:{ln}")

    if t in ("FunctionDef", "AsyncFunctionDef"):
        _visit_function(node, scope, rels, is_async=(t == "AsyncFunctionDef"))

    elif t == "ClassDef":
        _visit_class(node, scope, rels)

    elif t == "Assign":
        rhs_sym = _node_to_symbol(node["value"])
        for target in node.get("targets", []):
            _emit(rels, ln, "assigns", _node_to_symbol(target), rhs_sym)
            # Scan target for subscript expressions
            _scan_expr(target, scope, rels)
        _scan_expr(node.get("value"), scope, rels)

    elif t == "AugAssign":
        target = _node_to_symbol(node["target"])
        value = _node_to_symbol(node["value"])
        op = node["op"]["_type"]
        _emit(rels, ln, "aug_assigns", target, value)
        _emit(rels, ln, "aug_op", target, op)
        _scan_expr(node.get("target"), scope, rels)  # Scan target for subscript expressions
        _scan_expr(node.get("value"), scope, rels)

    elif t == "AnnAssign":
        if node.get("target"):
            _emit(rels, ln, "annotation", _node_to_symbol(node["target"]),
                  _node_to_symbol(node["annotation"]))
            _scan_expr(node.get("annotation"), scope, rels)
            if node.get("value"):
                _emit(rels, ln, "assigns", _node_to_symbol(node["target"]),
                      _node_to_symbol(node["value"]))
                _scan_expr(node["value"], scope, rels)

    elif t == "Import":
        for alias in node.get("names", []):
            # For bare imports, the module is the name itself
            _emit(rels, ln, "imports", alias["name"], alias["name"])
            if alias.get("asname"):
                _emit(rels, ln, "aliases", alias["name"], alias["asname"])

    elif t == "ImportFrom":
        module = node.get("module") or ""
        for alias in node.get("names", []):
            _emit(rels, ln, "imports", module, alias["name"])
            if alias.get("asname"):
                _emit(rels, ln, "aliases", alias["name"], alias["asname"])

    elif t == "If":
        test_sym = _node_to_symbol(node["test"])
        _emit(rels, ln, "if_test", scope, test_sym)
        body_scope = f"if_body:{ln}"
        _emit(rels, ln, "if_body", scope, body_scope)
        _scan_expr(node["test"], scope, rels)
        for i, child in enumerate(node.get("body", [])):
            _visit_stmt(child, body_scope, rels, stmt_seq=i)
        orelse = node.get("orelse", [])
        if len(orelse) == 1 and isinstance(orelse[0], dict) \
                and orelse[0].get("_type") == "If":
            _emit(rels, _ln(orelse[0]), "elif_test", scope,
                  _node_to_symbol(orelse[0]["test"]))
        else_scope = f"else_body:{ln}"
        for i, child in enumerate(orelse):
            _visit_stmt(child, else_scope, rels, stmt_seq=i)

    elif t in ("For", "AsyncFor"):
        target_node = node.get("target")
        target_sym = _node_to_symbol(target_node)
        iter_sym = _node_to_symbol(node["iter"])
        _emit(rels, ln, "for_target", scope, target_sym)
        _emit(rels, ln, "for_iter", scope, iter_sym)
        body_scope = f"for_body:{ln}"
        _emit(rels, ln, "for_body", scope, body_scope)
        # Check for tuple target (unpacking)
        if isinstance(target_node, dict) and target_node.get("_type") == "Tuple":
            _emit(rels, ln, "for_tuple_target", scope, target_sym)
        _scan_expr(node["iter"], scope, rels)
        for i, child in enumerate(node.get("body", [])):
            _visit_stmt(child, body_scope, rels, stmt_seq=i)
        else_scope = f"for_else:{ln}"
        for i, child in enumerate(node.get("orelse", [])):
            _visit_stmt(child, else_scope, rels, stmt_seq=i)

    elif t == "While":
        _emit(rels, ln, "while_test", scope, _node_to_symbol(node["test"]))
        body_scope = f"while_body:{ln}"
        _emit(rels, ln, "while_body", scope, body_scope)
        _scan_expr(node["test"], scope, rels)
        for i, child in enumerate(node.get("body", [])):
            _visit_stmt(child, body_scope, rels, stmt_seq=i)
        else_scope = f"while_else:{ln}"
        for i, child in enumerate(node.get("orelse", [])):
            _visit_stmt(child, else_scope, rels, stmt_seq=i)

    elif t in ("With", "AsyncWith"):
        for item in node.get("items", []):
            ctx = item.get("context_expr")
            _emit(rels, ln, "with_context", scope, _node_to_symbol(ctx))
            if item.get("optional_vars"):
                _emit(rels, ln, "with_as", _node_to_symbol(ctx),
                      _node_to_symbol(item["optional_vars"]))
            _scan_expr(ctx, scope, rels)
        body_scope = f"with_body:{ln}"
        _emit(rels, ln, "with_body", scope, body_scope)
        for i, child in enumerate(node.get("body", [])):
            _visit_stmt(child, body_scope, rels, stmt_seq=i)

    elif t in ("Try", "TryStar"):
        _emit(rels, ln, "try_start", scope, f"try:{ln}")
        try_scope = f"try_body:{ln}"
        _emit(rels, ln, "try_body", scope, try_scope)
        for i, child in enumerate(node.get("body", [])):
            _visit_stmt(child, try_scope, rels, stmt_seq=i)
        for handler in node.get("handlers", []):
            if handler.get("type"):
                _emit(rels, _ln(handler), "handles", scope,
                      _node_to_symbol(handler["type"]))
                if handler.get("name"):
                    _emit(rels, _ln(handler), "except_as",
                          _node_to_symbol(handler["type"]), handler["name"])
            except_scope = f"except_body:{_ln(handler)}"
            _emit(rels, _ln(handler), "except_body", scope, except_scope)
            for i, child in enumerate(handler.get("body", [])):
                _visit_stmt(child, except_scope, rels, stmt_seq=i)
        if node.get("orelse"):
            _emit(rels, ln, "try_else", scope, f"else:{ln}")
            else_scope = f"else_body:{ln}"
            _emit(rels, ln, "else_body", scope, else_scope)
            for i, child in enumerate(node["orelse"]):
                _visit_stmt(child, else_scope, rels, stmt_seq=i)
        if node.get("finalbody"):
            _emit(rels, ln, "try_finally", scope, f"finally:{ln}")
            finally_scope = f"finally_body:{ln}"
            _emit(rels, ln, "finally_body", scope, finally_scope)
            for i, child in enumerate(node["finalbody"]):
                _visit_stmt(child, finally_scope, rels, stmt_seq=i)

    elif t == "Raise":
        if node.get("exc"):
            _emit(rels, ln, "raises", scope, _node_to_symbol(node["exc"]))
            if node.get("cause"):
                _emit(rels, ln, "raises_from", _node_to_symbol(node["exc"]),
                      _node_to_symbol(node["cause"]))
            _scan_expr(node["exc"], scope, rels)

    elif t == "Return":
        if node.get("value") is not None:
            _emit(rels, ln, "returns", scope, _node_to_symbol(node["value"]))
            _scan_expr(node["value"], scope, rels)
        else:
            # Bare 'return' statement
            _emit(rels, ln, "returns", scope, "None")

    elif t == "Expr":
        # Expression statement (bare call, docstring, etc.)
        val = node.get("value")
        if isinstance(val, dict):
            vt = val.get("_type", "")
            if vt == "Constant":
                pass  # docstring — skip
            elif vt == "Yield":
                yv = _node_to_symbol(val["value"]) if val.get("value") else "None"
                _emit(rels, ln, "yields", scope, yv)
            elif vt == "YieldFrom":
                _emit(rels, ln, "yields_from", scope,
                      _node_to_symbol(val["value"]))
            else:
                # Expression statement - emit expr_stmt for full expression
                expr_sym = _node_to_symbol(val)
                _emit(rels, ln, "expr_stmt", scope, expr_sym)
                _scan_expr(val, scope, rels)

    elif t == "Global":
        for name in node.get("names", []):
            _emit(rels, ln, "global_decl", scope, name)

    elif t == "Nonlocal":
        for name in node.get("names", []):
            _emit(rels, ln, "nonlocal_decl", scope, name)

    elif t == "Assert":
        _emit(rels, ln, "assert_test", scope, _node_to_symbol(node["test"]))
        if node.get("msg"):
            _emit(rels, ln, "assert_msg", scope, _node_to_symbol(node["msg"]))
        _scan_expr(node["test"], scope, rels)

    elif t == "Delete":
        for target in node.get("targets", []):
            _emit(rels, ln, "del_target", scope, _node_to_symbol(target))

    elif t == "Pass":
        _emit(rels, ln, "pass_stmt", scope, "pass")

    elif t == "Break":
        _emit(rels, ln, "break_stmt", scope, "break")

    elif t == "Continue":
        _emit(rels, ln, "continue_stmt", scope, "continue")

    else:
        # Fallback: scan for expressions
        _scan_expr(node, scope, rels)


def _visit_function(node: Dict, scope: str, rels: List[Dict],
                    is_async: bool = False):
    name = node["name"]
    ln = _ln(node)
    tag_name = f"async:{name}" if is_async else name
    _emit(rels, ln, "defines", scope, tag_name)

    if is_async:
        _emit(rels, ln, "async_def", name, "async")

    # Decorators
    for dec in node.get("decorator_list", []):
        dec_name = _node_to_symbol(dec)
        _emit(rels, _ln(dec), "decorates", dec_name, name)
        # Emit specific decorator types
        if dec_name == "staticmethod":
            _emit(rels, _ln(dec), "static_method", name, "staticmethod")
        elif dec_name == "classmethod":
            _emit(rels, _ln(dec), "class_method", name, "classmethod")
        elif dec_name == "property":
            _emit(rels, _ln(dec), "property_def", name, "property")

    # Parameters
    args = node.get("args", {})
    all_positional = args.get("posonlyargs", []) + args.get("args", [])
    defaults = args.get("defaults", [])
    defaults_offset = len(all_positional) - len(defaults)

    for i, arg in enumerate(all_positional):
        _emit(rels, ln, "param", name, arg["arg"])
        if arg.get("annotation"):
            _emit(rels, ln, "annotation", arg["arg"],
                  _node_to_symbol(arg["annotation"]))
            _scan_expr(arg["annotation"], name, rels)
        di = i - defaults_offset
        if 0 <= di < len(defaults) and defaults[di] is not None:
            _emit(rels, ln, "default_value", arg["arg"],
                  _node_to_symbol(defaults[di]))
            _scan_expr(defaults[di], name, rels)

    if args.get("vararg"):
        va = args["vararg"]
        _emit(rels, ln, "star_param", name, va["arg"])
        if va.get("annotation"):
            _emit(rels, ln, "annotation", va["arg"],
                  _node_to_symbol(va["annotation"]))
            _scan_expr(va["annotation"], name, rels)

    kw_defaults = args.get("kw_defaults", [])
    for i, arg in enumerate(args.get("kwonlyargs", [])):
        _emit(rels, ln, "param", name, arg["arg"])
        if arg.get("annotation"):
            _emit(rels, ln, "annotation", arg["arg"],
                  _node_to_symbol(arg["annotation"]))
            _scan_expr(arg["annotation"], name, rels)
        if i < len(kw_defaults) and kw_defaults[i] is not None:
            _emit(rels, ln, "default_value", arg["arg"],
                  _node_to_symbol(kw_defaults[i]))
            _scan_expr(kw_defaults[i], name, rels)

    if args.get("kwarg"):
        ka = args["kwarg"]
        _emit(rels, ln, "double_star_param", name, ka["arg"])
        if ka.get("annotation"):
            _emit(rels, ln, "annotation", ka["arg"],
                  _node_to_symbol(ka["annotation"]))
            _scan_expr(ka["annotation"], name, rels)

    # Return annotation
    if node.get("returns"):
        _emit(rels, ln, "returns", name, _node_to_symbol(node["returns"]))
        _scan_expr(node["returns"], name, rels)

    # Body — check for ellipsis body
    body = node.get("body", [])
    if len(body) == 1 and body[0].get("_type") == "Expr":
        val = body[0].get("value", {})
        if val.get("_type") == "Constant" and val.get("value") is ...:
            _emit(rels, _ln(body[0]), "ellipsis_body", name, "...")
            return

    # Check for docstring
    doc_info = _get_docstring(body)
    if doc_info:
        doc_ln, doc_str = doc_info
        # Truncate docstring for symbol constraint (no newlines)
        doc_sym = repr(doc_str)[:100].replace("\n", "\\n").replace(" ", "")
        _emit(rels, doc_ln, "docstring", name, doc_sym)

    for i, child in enumerate(body):
        _emit(rels, _ln(child), "contains", name, f"stmt:{_ln(child)}")
        _visit_stmt(child, name, rels, stmt_seq=i)


def _visit_class(node: Dict, scope: str, rels: List[Dict]):
    name = node["name"]
    ln = _ln(node)
    _emit(rels, ln, "defines", scope, name)

    for dec in node.get("decorator_list", []):
        _emit(rels, _ln(dec), "decorates", _node_to_symbol(dec), name)

    for base in node.get("bases", []):
        _emit(rels, ln, "bases", name, _node_to_symbol(base))

    for kw in node.get("keywords", []):
        if kw.get("arg") == "metaclass":
            _emit(rels, ln, "metaclass", name, _node_to_symbol(kw["value"]))
        elif kw.get("arg"):
            _emit(rels, ln, "keyword_arg", name,
                  f"{kw['arg']}={_node_to_symbol(kw['value'])}")

    body = node.get("body", [])
    # Check for docstring
    doc_info = _get_docstring(body)
    if doc_info:
        doc_ln, doc_str = doc_info
        doc_sym = repr(doc_str)[:100].replace("\n", "\\n").replace(" ", "")
        _emit(rels, doc_ln, "docstring", name, doc_sym)

    for i, child in enumerate(body):
        _emit(rels, _ln(child), "contains", name, f"stmt:{_ln(child)}")
        _visit_stmt(child, name, rels, stmt_seq=i)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_python_to_ast_tag_jsonl(ast_obj: Dict) -> str:
    """Convert a Python AST JSON dict to JSONL of AstTagRelation records.

    <- $ast_obj ast[Python] (JSON AST dict from ast/ directory)
    -> JsonLines[AstTagRelation[Python]]
    """
    rels: List[Dict] = []
    body = ast_obj.get("body", [])

    # Check for module-level docstring
    doc_info = _get_docstring(body)
    if doc_info:
        doc_ln, doc_str = doc_info
        doc_sym = repr(doc_str)[:100].replace("\n", "\\n").replace(" ", "")
        _emit(rels, doc_ln, "docstring", "<module>", doc_sym)

    for i, child in enumerate(body):
        _emit(rels, _ln(child), "contains", "<module>", f"stmt:{_ln(child)}")
        _visit_stmt(child, "<module>", rels, stmt_seq=i)
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in rels)


if __name__ == "__main__":
    CODEBASE_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "code_auto_encoder", "codebase",
    )

    # Collect all .py files
    py_files = []
    for root, _dirs, files in os.walk(CODEBASE_DIR):
        for f in sorted(files):
            if f.endswith(".py") and f != "__init__.py":
                py_files.append(os.path.join(root, f))
    py_files.sort()

    total_relations = 0
    for path in py_files:
        with open(path, "r", encoding="utf-8") as fh:
            source = fh.read()
        tree = ast.parse(source, filename=path)
        ast_obj = ast_to_dict(tree)
        jsonl = convert_python_to_ast_tag_jsonl(ast_obj)
        n = len(jsonl.strip().splitlines()) if jsonl.strip() else 0
        total_relations += n
        rel = os.path.relpath(path, CODEBASE_DIR)
        print(f"{rel}: {n} relations")

    print(f"\n--- total: {len(py_files)} files, {total_relations} relations ---")
