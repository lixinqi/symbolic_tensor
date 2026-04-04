"""Convert Python AST JSON to AstTagRelationGroup JSONL.

Generated from convert_python_to_ast_tag_jsonl.viba.

Viba DSL specification:
  convert_python_to_ast_tag_jsonl[ProgrammingLanguage] :=
    JsonLines[$ast_tag_rel_group AstTagRelationGroup[ProgrammingLanguage]]
    <- $ast_obj ast[ProgrammingLanguage]
    # inline
    <- Import[./ast_tag_relation_group.viba]
    <- { Assert that the $ast_tag_rel_group meets the AstTagRelationGroup schema }
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
        return f"unaryop:{uid}"
    if t == "Compare":
        return f"compare:{uid}"
    if t == "BoolOp":
        return f"boolop:{uid}"
    if t == "Await":
        return f"await:{uid}"
    if t == "Lambda":
        return f"lambda:{uid}"
    if t == "IfExp":
        return f"ifexp:{uid}"
    if t == "Yield":
        return f"yield:{uid}"
    if t == "YieldFrom":
        return f"yieldfrom:{uid}"
    if t == "Dict":
        return f"dict:{uid}"
    if t == "Set":
        return f"set:{uid}"
    if t == "Slice":
        return f"slice:{uid}"
    # For other complex nodes, use the uid reference
    if expr_id:
        return expr_id
    return f"{t.lower()}:{uid}"


def _get_op_symbol(node: Any) -> str:
    """Convert an operator node to its symbol string."""
    if isinstance(node, ast.AST):
        node = ast_to_dict(node)
    if isinstance(node, dict):
        t = node.get("_type", "")
        if t == "Add":
            return "+"
        if t == "Sub":
            return "-"
        if t == "Mult":
            return "*"
        if t == "Div":
            return "/"
        if t == "FloorDiv":
            return "//"
        if t == "Mod":
            return "%"
        if t == "Pow":
            return "**"
        if t == "LShift":
            return "<<"
        if t == "RShift":
            return ">>"
        if t == "BitOr":
            return "|"
        if t == "BitXor":
            return "^"
        if t == "BitAnd":
            return "&"
        if t == "MatMult":
            return "@"
        if t == "UAdd":
            return "+"
        if t == "USub":
            return "-"
        if t == "Not":
            return "not"
        if t == "Invert":
            return "~"
        if t == "Eq":
            return "=="
        if t == "NotEq":
            return "!="
        if t == "Lt":
            return "<"
        if t == "LtE":
            return "<="
        if t == "Gt":
            return ">"
        if t == "GtE":
            return ">="
        if t == "Is":
            return "is"
        if t == "IsNot":
            return "isnot"
        if t == "In":
            return "in"
        if t == "NotIn":
            return "notin"
        if t == "And":
            return "and"
        if t == "Or":
            return "or"
    return str(node)


# ---------------------------------------------------------------------------
# Relation extraction
# ---------------------------------------------------------------------------

def _extract_relations(node: Any, parent: Any = None, relations: Optional[List[Dict]] = None) -> List[Dict]:
    """Extract AstTagRelation records from a JSON AST node."""
    if relations is None:
        relations = []
    if not isinstance(node, dict):
        return relations

    t = node.get("_type", "")
    ln = node.get("_lineno", 0)
    col = node.get("_col_offset", 0)
    end_col = node.get("_end_col_offset", col + 1)
    uid = f"{ln}:{col}:{end_col}" if ln else "0:0:1"

    # Module → contains statements
    if t == "Module":
        for stmt in node.get("body", []):
            stmt_sym = _node_to_symbol(stmt)
            relations.append({
                "line": ln,
                "relation_tag": "contains",
                "lhs_tag": "module",
                "rhs_tag": stmt_sym
            })
            _extract_relations(stmt, node, relations)

    # FunctionDef / AsyncFunctionDef
    elif t in ("FunctionDef", "AsyncFunctionDef"):
        name = node.get("name", "")
        relations.append({
            "line": ln,
            "relation_tag": "defines",
            "lhs_tag": "module",
            "rhs_tag": name
        })
        # args → param relations
        args = node.get("args", {})
        for i, arg in enumerate(args.get("args", [])):
            arg_name = arg.get("arg", "")
            relations.append({
                "line": ln,
                "relation_tag": "param",
                "lhs_tag": name,
                "rhs_tag": arg_name
            })
            if arg.get("annotation"):
                ann_sym = _node_to_symbol(arg["annotation"])
                relations.append({
                    "line": ln,
                    "relation_tag": "annotation",
                    "lhs_tag": arg_name,
                    "rhs_tag": ann_sym
                })
        # returns
        if node.get("returns"):
            ret_sym = _node_to_symbol(node["returns"])
            relations.append({
                "line": ln,
                "relation_tag": "returns",
                "lhs_tag": name,
                "rhs_tag": ret_sym
            })
        # decorators
        for dec in node.get("decorator_list", []):
            dec_sym = _node_to_symbol(dec)
            relations.append({
                "line": ln,
                "relation_tag": "decorates",
                "lhs_tag": dec_sym,
                "rhs_tag": name
            })
        # body
        for stmt in node.get("body", []):
            stmt_sym = _node_to_symbol(stmt)
            relations.append({
                "line": ln,
                "relation_tag": "contains",
                "lhs_tag": name,
                "rhs_tag": stmt_sym
            })
            _extract_relations(stmt, node, relations)
        # async flag
        if t == "AsyncFunctionDef":
            relations.append({
                "line": ln,
                "relation_tag": "async_def",
                "lhs_tag": name,
                "rhs_tag": "async"
            })

    # ClassDef
    elif t == "ClassDef":
        name = node.get("name", "")
        relations.append({
            "line": ln,
            "relation_tag": "defines",
            "lhs_tag": "module",
            "rhs_tag": name
        })
        # bases
        for base in node.get("bases", []):
            base_sym = _node_to_symbol(base)
            relations.append({
                "line": ln,
                "relation_tag": "bases",
                "lhs_tag": name,
                "rhs_tag": base_sym
            })
        # keywords (metaclass, etc.)
        for kw in node.get("keywords", []):
            if kw.get("arg"):
                kw_sym = _node_to_symbol(kw.get("value"))
                relations.append({
                    "line": ln,
                    "relation_tag": "keyword_arg",
                    "lhs_tag": name,
                    "rhs_tag": kw_sym
                })
        # decorators
        for dec in node.get("decorator_list", []):
            dec_sym = _node_to_symbol(dec)
            relations.append({
                "line": ln,
                "relation_tag": "decorates",
                "lhs_tag": dec_sym,
                "rhs_tag": name
            })
        # body
        for stmt in node.get("body", []):
            stmt_sym = _node_to_symbol(stmt)
            relations.append({
                "line": ln,
                "relation_tag": "contains",
                "lhs_tag": name,
                "rhs_tag": stmt_sym
            })
            _extract_relations(stmt, node, relations)

    # Assign
    elif t == "Assign":
        targets = node.get("targets", [])
        value = node.get("value")
        value_sym = _node_to_symbol(value)
        for target in targets:
            target_sym = _node_to_symbol(target)
            relations.append({
                "line": ln,
                "relation_tag": "assigns",
                "lhs_tag": target_sym,
                "rhs_tag": value_sym
            })
        _extract_relations(value, node, relations)

    # AugAssign
    elif t == "AugAssign":
        target = node.get("target")
        op = node.get("op")
        value = node.get("value")
        target_sym = _node_to_symbol(target)
        op_sym = _get_op_symbol(op)
        value_sym = _node_to_symbol(value)
        relations.append({
            "line": ln,
            "relation_tag": "aug_op",
            "lhs_tag": target_sym,
            "rhs_tag": op_sym
        })
        relations.append({
            "line": ln,
            "relation_tag": "aug_assigns",
            "lhs_tag": target_sym,
            "rhs_tag": value_sym
        })
        _extract_relations(value, node, relations)

    # AnnAssign
    elif t == "AnnAssign":
        target = node.get("target")
        annotation = node.get("annotation")
        value = node.get("value")
        target_sym = _node_to_symbol(target)
        ann_sym = _node_to_symbol(annotation)
        relations.append({
            "line": ln,
            "relation_tag": "type_annotation",
            "lhs_tag": target_sym,
            "rhs_tag": ann_sym
        })
        if value:
            value_sym = _node_to_symbol(value)
            relations.append({
                "line": ln,
                "relation_tag": "assigns",
                "lhs_tag": target_sym,
                "rhs_tag": value_sym
            })
            _extract_relations(value, node, relations)

    # Expr (expression statement)
    elif t == "Expr":
        value = node.get("value")
        value_sym = _node_to_symbol(value)
        relations.append({
            "line": ln,
            "relation_tag": "expr_stmt",
            "lhs_tag": "expr",
            "rhs_tag": value_sym
        })
        _extract_relations(value, node, relations)

    # Return
    elif t == "Return":
        value = node.get("value")
        if value:
            value_sym = _node_to_symbol(value)
            relations.append({
                "line": ln,
                "relation_tag": "returns",
                "lhs_tag": "return",
                "rhs_tag": value_sym
            })
            _extract_relations(value, node, relations)

    # Call
    elif t == "Call":
        func = node.get("func")
        func_sym = _node_to_symbol(func)
        for i, arg in enumerate(node.get("args", [])):
            arg_sym = _node_to_symbol(arg)
            relations.append({
                "line": ln,
                "relation_tag": "call_arg",
                "lhs_tag": func_sym,
                "rhs_tag": arg_sym
            })
            _extract_relations(arg, node, relations)
        for kw in node.get("keywords", []):
            kw_val = kw.get("value")
            kw_sym = _node_to_symbol(kw_val)
            relations.append({
                "line": ln,
                "relation_tag": "call_arg_value",
                "lhs_tag": func_sym,
                "rhs_tag": kw_sym
            })
            _extract_relations(kw_val, node, relations)
        _extract_relations(func, node, relations)

    # BinOp
    elif t == "BinOp":
        left = node.get("left")
        op = node.get("op")
        right = node.get("right")
        left_sym = _node_to_symbol(left)
        op_sym = _get_op_symbol(op)
        right_sym = _node_to_symbol(right)
        binop_id = f"binop:{uid}"
        relations.append({
            "line": ln,
            "relation_tag": "bin_op",
            "lhs_tag": op_sym,
            "rhs_tag": binop_id
        })
        relations.append({
            "line": ln,
            "relation_tag": "bin_op_left",
            "lhs_tag": binop_id,
            "rhs_tag": left_sym
        })
        relations.append({
            "line": ln,
            "relation_tag": "bin_op_right",
            "lhs_tag": binop_id,
            "rhs_tag": right_sym
        })
        _extract_relations(left, node, relations)
        _extract_relations(right, node, relations)

    # UnaryOp
    elif t == "UnaryOp":
        op = node.get("op")
        operand = node.get("operand")
        op_sym = _get_op_symbol(op)
        operand_sym = _node_to_symbol(operand)
        unaryop_id = f"unaryop:{uid}"
        relations.append({
            "line": ln,
            "relation_tag": "unary_op",
            "lhs_tag": op_sym,
            "rhs_tag": unaryop_id
        })
        relations.append({
            "line": ln,
            "relation_tag": "unary_op_operand",
            "lhs_tag": unaryop_id,
            "rhs_tag": operand_sym
        })
        _extract_relations(operand, node, relations)

    # Compare
    elif t == "Compare":
        left = node.get("left")
        ops = node.get("ops", [])
        comparators = node.get("comparators", [])
        left_sym = _node_to_symbol(left)
        compare_id = f"compare:{uid}"
        relations.append({
            "line": ln,
            "relation_tag": "compare_left",
            "lhs_tag": compare_id,
            "rhs_tag": left_sym
        })
        for i, (op, comp) in enumerate(zip(ops, comparators)):
            op_sym = _get_op_symbol(op)
            comp_sym = _node_to_symbol(comp)
            relations.append({
                "line": ln,
                "relation_tag": "compare_op",
                "lhs_tag": compare_id,
                "rhs_tag": op_sym
            })
            relations.append({
                "line": ln,
                "relation_tag": "compare_right",
                "lhs_tag": compare_id,
                "rhs_tag": comp_sym
            })
            _extract_relations(comp, node, relations)
        _extract_relations(left, node, relations)

    # BoolOp
    elif t == "BoolOp":
        op = node.get("op")
        values = node.get("values", [])
        op_sym = _get_op_symbol(op)
        boolop_id = f"boolop:{uid}"
        relations.append({
            "line": ln,
            "relation_tag": "bool_op",
            "lhs_tag": op_sym,
            "rhs_tag": boolop_id
        })
        for val in values:
            val_sym = _node_to_symbol(val)
            relations.append({
                "line": ln,
                "relation_tag": "bool_op_operand",
                "lhs_tag": boolop_id,
                "rhs_tag": val_sym
            })
            _extract_relations(val, node, relations)

    # Subscript
    elif t == "Subscript":
        value = node.get("value")
        slice_val = node.get("slice")
        value_sym = _node_to_symbol(value)
        slice_sym = _node_to_symbol(slice_val)
        subscript_id = f"subscript:{uid}"
        relations.append({
            "line": ln,
            "relation_tag": "subscript_value",
            "lhs_tag": subscript_id,
            "rhs_tag": value_sym
        })
        relations.append({
            "line": ln,
            "relation_tag": "subscript",
            "lhs_tag": subscript_id,
            "rhs_tag": slice_sym
        })
        _extract_relations(value, node, relations)
        _extract_relations(slice_val, node, relations)

    # Slice
    elif t == "Slice":
        lower = node.get("lower")
        upper = node.get("upper")
        step = node.get("step")
        slice_id = f"slice:{uid}"
        if lower:
            lower_sym = _node_to_symbol(lower)
            relations.append({
                "line": ln,
                "relation_tag": "slice_lower",
                "lhs_tag": slice_id,
                "rhs_tag": lower_sym
            })
            _extract_relations(lower, node, relations)
        if upper:
            upper_sym = _node_to_symbol(upper)
            relations.append({
                "line": ln,
                "relation_tag": "slice_upper",
                "lhs_tag": slice_id,
                "rhs_tag": upper_sym
            })
            _extract_relations(upper, node, relations)
        if step:
            step_sym = _node_to_symbol(step)
            relations.append({
                "line": ln,
                "relation_tag": "slice_step",
                "lhs_tag": slice_id,
                "rhs_tag": step_sym
            })
            _extract_relations(step, node, relations)

    # If
    elif t == "If":
        test = node.get("test")
        body = node.get("body", [])
        orelse = node.get("orelse", [])
        test_sym = _node_to_symbol(test)
        relations.append({
            "line": ln,
            "relation_tag": "if_test",
            "lhs_tag": "if",
            "rhs_tag": test_sym
        })
        _extract_relations(test, node, relations)
        for stmt in body:
            stmt_sym = _node_to_symbol(stmt)
            relations.append({
                "line": ln,
                "relation_tag": "if_body",
                "lhs_tag": "if",
                "rhs_tag": stmt_sym
            })
            _extract_relations(stmt, node, relations)
        for stmt in orelse:
            stmt_sym = _node_to_symbol(stmt)
            relations.append({
                "line": ln,
                "relation_tag": "else_body",
                "lhs_tag": "if",
                "rhs_tag": stmt_sym
            })
            _extract_relations(stmt, node, relations)

    # For / AsyncFor
    elif t in ("For", "AsyncFor"):
        target = node.get("target")
        iter_val = node.get("iter")
        body = node.get("body", [])
        orelse = node.get("orelse", [])
        target_sym = _node_to_symbol(target)
        iter_sym = _node_to_symbol(iter_val)
        relations.append({
            "line": ln,
            "relation_tag": "for_target",
            "lhs_tag": "for",
            "rhs_tag": target_sym
        })
        relations.append({
            "line": ln,
            "relation_tag": "for_iter",
            "lhs_tag": "for",
            "rhs_tag": iter_sym
        })
        _extract_relations(iter_val, node, relations)
        for stmt in body:
            stmt_sym = _node_to_symbol(stmt)
            relations.append({
                "line": ln,
                "relation_tag": "for_body",
                "lhs_tag": "for",
                "rhs_tag": stmt_sym
            })
            _extract_relations(stmt, node, relations)

    # While
    elif t == "While":
        test = node.get("test")
        body = node.get("body", [])
        orelse = node.get("orelse", [])
        test_sym = _node_to_symbol(test)
        relations.append({
            "line": ln,
            "relation_tag": "while_test",
            "lhs_tag": "while",
            "rhs_tag": test_sym
        })
        _extract_relations(test, node, relations)
        for stmt in body:
            stmt_sym = _node_to_symbol(stmt)
            relations.append({
                "line": ln,
                "relation_tag": "while_body",
                "lhs_tag": "while",
                "rhs_tag": stmt_sym
            })
            _extract_relations(stmt, node, relations)

    # With / AsyncWith
    elif t in ("With", "AsyncWith"):
        items = node.get("items", [])
        body = node.get("body", [])
        for item in items:
            ctx = item.get("context_expr")
            opt_vars = item.get("optional_vars")
            ctx_sym = _node_to_symbol(ctx)
            relations.append({
                "line": ln,
                "relation_tag": "with_context",
                "lhs_tag": "with",
                "rhs_tag": ctx_sym
            })
            _extract_relations(ctx, node, relations)
            if opt_vars:
                vars_sym = _node_to_symbol(opt_vars)
                relations.append({
                    "line": ln,
                    "relation_tag": "with_as",
                    "lhs_tag": ctx_sym,
                    "rhs_tag": vars_sym
                })
        for stmt in body:
            stmt_sym = _node_to_symbol(stmt)
            relations.append({
                "line": ln,
                "relation_tag": "with_body",
                "lhs_tag": "with",
                "rhs_tag": stmt_sym
            })
            _extract_relations(stmt, node, relations)

    # Try
    elif t == "Try":
        body = node.get("body", [])
        handlers = node.get("handlers", [])
        orelse = node.get("orelse", [])
        finalbody = node.get("finalbody", [])
        for stmt in body:
            stmt_sym = _node_to_symbol(stmt)
            relations.append({
                "line": ln,
                "relation_tag": "try_body",
                "lhs_tag": "try",
                "rhs_tag": stmt_sym
            })
            _extract_relations(stmt, node, relations)
        for handler in handlers:
            exc_type = handler.get("type")
            name = handler.get("name")
            handler_body = handler.get("body", [])
            if exc_type:
                exc_sym = _node_to_symbol(exc_type)
                relations.append({
                    "line": ln,
                    "relation_tag": "handles",
                    "lhs_tag": "except",
                    "rhs_tag": exc_sym
                })
                _extract_relations(exc_type, node, relations)
            if name:
                relations.append({
                    "line": ln,
                    "relation_tag": "except_as",
                    "lhs_tag": "except",
                    "rhs_tag": name
                })
            for stmt in handler_body:
                stmt_sym = _node_to_symbol(stmt)
                relations.append({
                    "line": ln,
                    "relation_tag": "except_body",
                    "lhs_tag": "except",
                    "rhs_tag": stmt_sym
                })
                _extract_relations(stmt, node, relations)
        for stmt in orelse:
            stmt_sym = _node_to_symbol(stmt)
            relations.append({
                "line": ln,
                "relation_tag": "try_else",
                "lhs_tag": "try",
                "rhs_tag": stmt_sym
            })
            _extract_relations(stmt, node, relations)
        for stmt in finalbody:
            stmt_sym = _node_to_symbol(stmt)
            relations.append({
                "line": ln,
                "relation_tag": "finally_body",
                "lhs_tag": "try",
                "rhs_tag": stmt_sym
            })
            _extract_relations(stmt, node, relations)

    # Raise
    elif t == "Raise":
        exc = node.get("exc")
        cause = node.get("cause")
        if exc:
            exc_sym = _node_to_symbol(exc)
            relations.append({
                "line": ln,
                "relation_tag": "raises",
                "lhs_tag": "raise",
                "rhs_tag": exc_sym
            })
            _extract_relations(exc, node, relations)
        if cause:
            cause_sym = _node_to_symbol(cause)
            relations.append({
                "line": ln,
                "relation_tag": "raises_from",
                "lhs_tag": exc_sym if exc else "raise",
                "rhs_tag": cause_sym
            })
            _extract_relations(cause, node, relations)

    # Import
    elif t == "Import":
        for alias in node.get("names", []):
            name = alias.get("name")
            asname = alias.get("asname")
            relations.append({
                "line": ln,
                "relation_tag": "imports",
                "lhs_tag": "module",
                "rhs_tag": name
            })
            if asname:
                relations.append({
                    "line": ln,
                    "relation_tag": "aliases",
                    "lhs_tag": name,
                    "rhs_tag": asname
                })

    # ImportFrom
    elif t == "ImportFrom":
        module = node.get("module", "")
        for alias in node.get("names", []):
            name = alias.get("name")
            asname = alias.get("asname")
            full_name = f"{module}.{name}" if module else name
            relations.append({
                "line": ln,
                "relation_tag": "imports",
                "lhs_tag": "module",
                "rhs_tag": full_name
            })
            if asname:
                relations.append({
                    "line": ln,
                    "relation_tag": "aliases",
                    "lhs_tag": full_name,
                    "rhs_tag": asname
                })

    # Lambda
    elif t == "Lambda":
        args = node.get("args", {})
        body = node.get("body")
        lambda_id = f"lambda:{uid}"
        for arg in args.get("args", []):
            arg_name = arg.get("arg", "")
            relations.append({
                "line": ln,
                "relation_tag": "param",
                "lhs_tag": lambda_id,
                "rhs_tag": arg_name
            })
        if body:
            body_sym = _node_to_symbol(body)
            relations.append({
                "line": ln,
                "relation_tag": "lambda_body",
                "lhs_tag": lambda_id,
                "rhs_tag": body_sym
            })
            _extract_relations(body, node, relations)

    # IfExp (ternary)
    elif t == "IfExp":
        test = node.get("test")
        body = node.get("body")
        orelse = node.get("orelse")
        ifexp_id = f"ifexp:{uid}"
        test_sym = _node_to_symbol(test)
        body_sym = _node_to_symbol(body)
        else_sym = _node_to_symbol(orelse)
        relations.append({
            "line": ln,
            "relation_tag": "if_expr_test",
            "lhs_tag": ifexp_id,
            "rhs_tag": test_sym
        })
        relations.append({
            "line": ln,
            "relation_tag": "if_expr_body",
            "lhs_tag": ifexp_id,
            "rhs_tag": body_sym
        })
        relations.append({
            "line": ln,
            "relation_tag": "if_expr_else",
            "lhs_tag": ifexp_id,
            "rhs_tag": else_sym
        })
        _extract_relations(test, node, relations)
        _extract_relations(body, node, relations)
        _extract_relations(orelse, node, relations)

    # Await
    elif t == "Await":
        value = node.get("value")
        await_id = f"await:{uid}"
        value_sym = _node_to_symbol(value)
        relations.append({
            "line": ln,
            "relation_tag": "await_value",
            "lhs_tag": await_id,
            "rhs_tag": value_sym
        })
        _extract_relations(value, node, relations)

    # Yield
    elif t == "Yield":
        value = node.get("value")
        yield_id = f"yield:{uid}"
        if value:
            value_sym = _node_to_symbol(value)
            relations.append({
                "line": ln,
                "relation_tag": "yields",
                "lhs_tag": yield_id,
                "rhs_tag": value_sym
            })
            _extract_relations(value, node, relations)

    # YieldFrom
    elif t == "YieldFrom":
        value = node.get("value")
        yield_id = f"yieldfrom:{uid}"
        value_sym = _node_to_symbol(value)
        relations.append({
            "line": ln,
            "relation_tag": "yields_from",
            "lhs_tag": yield_id,
            "rhs_tag": value_sym
        })
        _extract_relations(value, node, relations)

    # Dict
    elif t == "Dict":
        keys = node.get("keys", [])
        values = node.get("values", [])
        dict_id = f"dict:{uid}"
        for k, v in zip(keys, values):
            if k:
                k_sym = _node_to_symbol(k)
                relations.append({
                    "line": ln,
                    "relation_tag": "dict_key",
                    "lhs_tag": dict_id,
                    "rhs_tag": k_sym
                })
                _extract_relations(k, node, relations)
            v_sym = _node_to_symbol(v)
            relations.append({
                "line": ln,
                "relation_tag": "dict_value",
                "lhs_tag": dict_id,
                "rhs_tag": v_sym
            })
            _extract_relations(v, node, relations)

    # ListComp / SetComp / DictComp / GeneratorExp
    elif t in ("ListComp", "SetComp", "DictComp", "GeneratorExp"):
        generators = node.get("generators", [])
        elt = node.get("elt")
        for gen in generators:
            target = gen.get("target")
            iter_val = gen.get("iter")
            ifs = gen.get("ifs", [])
            target_sym = _node_to_symbol(target)
            iter_sym = _node_to_symbol(iter_val)
            relations.append({
                "line": ln,
                "relation_tag": "comprehension_target",
                "lhs_tag": "comp",
                "rhs_tag": target_sym
            })
            relations.append({
                "line": ln,
                "relation_tag": "comprehension_iter",
                "lhs_tag": "comp",
                "rhs_tag": iter_sym
            })
            _extract_relations(iter_val, node, relations)
            for if_cond in ifs:
                if_sym = _node_to_symbol(if_cond)
                relations.append({
                    "line": ln,
                    "relation_tag": "comprehension_if",
                    "lhs_tag": "comp",
                    "rhs_tag": if_sym
                })
                _extract_relations(if_cond, node, relations)
        if elt:
            elt_sym = _node_to_symbol(elt)
            relations.append({
                "line": ln,
                "relation_tag": "comprehension_body",
                "lhs_tag": "comp",
                "rhs_tag": elt_sym
            })
            _extract_relations(elt, node, relations)

    # Pass / Break / Continue
    elif t == "Pass":
        relations.append({
            "line": ln,
            "relation_tag": "pass_stmt",
            "lhs_tag": "stmt",
            "rhs_tag": "pass"
        })
    elif t == "Break":
        relations.append({
            "line": ln,
            "relation_tag": "break_stmt",
            "lhs_tag": "stmt",
            "rhs_tag": "break"
        })
    elif t == "Continue":
        relations.append({
            "line": ln,
            "relation_tag": "continue_stmt",
            "lhs_tag": "stmt",
            "rhs_tag": "continue"
        })

    # Global / Nonlocal
    elif t == "Global":
        for name in node.get("names", []):
            relations.append({
                "line": ln,
                "relation_tag": "global_decl",
                "lhs_tag": "global",
                "rhs_tag": name
            })
    elif t == "Nonlocal":
        for name in node.get("names", []):
            relations.append({
                "line": ln,
                "relation_tag": "nonlocal_decl",
                "lhs_tag": "nonlocal",
                "rhs_tag": name
            })

    # Assert
    elif t == "Assert":
        test = node.get("test")
        msg = node.get("msg")
        test_sym = _node_to_symbol(test)
        relations.append({
            "line": ln,
            "relation_tag": "assert_test",
            "lhs_tag": "assert",
            "rhs_tag": test_sym
        })
        _extract_relations(test, node, relations)
        if msg:
            msg_sym = _node_to_symbol(msg)
            relations.append({
                "line": ln,
                "relation_tag": "assert_msg",
                "lhs_tag": "assert",
                "rhs_tag": msg_sym
            })
            _extract_relations(msg, node, relations)

    # Delete
    elif t == "Delete":
        for target in node.get("targets", []):
            target_sym = _node_to_symbol(target)
            relations.append({
                "line": ln,
                "relation_tag": "del_target",
                "lhs_tag": "del",
                "rhs_tag": target_sym
            })
            _extract_relations(target, node, relations)

    # NamedExpr (walrus operator)
    elif t == "NamedExpr":
        target = node.get("target")
        value = node.get("value")
        target_sym = _node_to_symbol(target)
        value_sym = _node_to_symbol(value)
        relations.append({
            "line": ln,
            "relation_tag": "walrus",
            "lhs_tag": target_sym,
            "rhs_tag": value_sym
        })
        _extract_relations(value, node, relations)

    # Constant
    elif t == "Constant":
        v = node.get("value")
        if v is None:
            relations.append({
                "line": ln,
                "relation_tag": "none_literal",
                "lhs_tag": "const",
                "rhs_tag": "None"
            })
        elif v is ...:
            relations.append({
                "line": ln,
                "relation_tag": "ellipsis_literal",
                "lhs_tag": "const",
                "rhs_tag": "..."
            })
        elif isinstance(v, bool):
            relations.append({
                "line": ln,
                "relation_tag": "const_value",
                "lhs_tag": "const",
                "rhs_tag": str(v)
            })
        elif isinstance(v, (int, float)):
            relations.append({
                "line": ln,
                "relation_tag": "const_value",
                "lhs_tag": "const",
                "rhs_tag": str(v)
            })
        elif isinstance(v, str):
            relations.append({
                "line": ln,
                "relation_tag": "string_literal",
                "lhs_tag": "const",
                "rhs_tag": _node_to_symbol(node)
            })
        elif isinstance(v, bytes):
            relations.append({
                "line": ln,
                "relation_tag": "bytes_literal",
                "lhs_tag": "const",
                "rhs_tag": repr(v)
            })

    # Attribute
    elif t == "Attribute":
        value = node.get("value")
        attr = node.get("attr")
        value_sym = _node_to_symbol(value)
        relations.append({
            "line": ln,
            "relation_tag": "member_access",
            "lhs_tag": value_sym,
            "rhs_tag": attr
        })
        _extract_relations(value, node, relations)

    # Recurse into children for unhandled nodes
    else:
        for key, val in node.items():
            if key.startswith("_"):
                continue
            if isinstance(val, dict):
                _extract_relations(val, node, relations)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        _extract_relations(item, node, relations)

    return relations


# ---------------------------------------------------------------------------
# Group relations into AstTagRelationGroup
# ---------------------------------------------------------------------------

def _group_relations(relations: List[Dict]) -> List[Dict]:
    """Group flat relations into AstTagRelationGroup records."""
    # Group by (line, relation_tag, owner)
    groups: Dict[Tuple[int, str, str], List[Dict]] = {}
    for rel in relations:
        line = rel.get("line", 0)
        tag = rel.get("relation_tag", "")
        lhs = rel.get("lhs_tag", "")
        rhs = rel.get("rhs_tag", "")
        # Use lhs as owner for grouping
        key = (line, tag, lhs)
        if key not in groups:
            groups[key] = []
        groups[key].append(rhs)

    result = []
    for (line, tag, owner), members in groups.items():
        result.append({
            "line": line,
            "relation_tag": tag,
            "owner_tag": owner,
            "member_tags": members
        })

    # Sort by line
    result.sort(key=lambda g: (g.get("line", 0), g.get("relation_tag", ""), g.get("owner_tag", "")))
    return result


# ---------------------------------------------------------------------------
# Main conversion function
# ---------------------------------------------------------------------------

def convert_python_to_jsonl(source_code: str) -> List[Dict]:
    """Convert Python source code to AstTagRelationGroup JSONL records."""
    tree = ast.parse(source_code)
    node_dict = ast_to_dict(tree)
    relations = _extract_relations(node_dict)
    return _group_relations(relations)


def convert_file_to_jsonl(filepath: str) -> List[Dict]:
    """Convert a Python file to AstTagRelationGroup JSONL records."""
    with open(filepath, "r", encoding="utf-8") as f:
        source_code = f.read()
    return convert_python_to_jsonl(source_code)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Find all JSON AST files in the ast/ directory
    base_dir = "ast"
    if os.path.exists(base_dir):
        json_files = []
        for root, dirs, files in os.walk(base_dir):
            for f in files:
                if f.endswith(".json") and f != "__init__.json":
                    json_files.append(os.path.join(root, f))

        total_relations = 0
        for json_file in sorted(json_files):
            with open(json_file, "r", encoding="utf-8") as f:
                node_dict = json.load(f)
            relations = _extract_relations(node_dict)
            groups = _group_relations(relations)
            print(f"{json_file}: {len(groups)} relations")
            total_relations += len(groups)

        print(f"\n--- total: {len(json_files)} files, {total_relations} relations ---")
    else:
        print("No ast/ directory found. Usage: python convert_python_to_ast_tag_jsonl.py")
        print("Or call convert_file_to_jsonl(filepath) directly.")
