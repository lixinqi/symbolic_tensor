"""Convert Python AST JSON to AstTagRelationGroup JSONL.

Generated from convert_python_to_ast_tag_jsonl.viba.

Viba DSL specification:
  convert_python_to_ast_tag_jsonl[ProgrammingLanguage] :=
    JsonLines[$ast_tag_rel_group AstTagRelationGroup[ProgrammingLanguage]]
    <- $ast_obj ast[ProgrammingLanguage]
    # inline
    <- Import[./ast_tag_relation_group.viba]
    <- { Assert that the $ast_tag_rel_group meets the AstTagRelationGroup schema }
    <- {
        all Symbol[ProgrammingLanguage]
        instances are used to generally connect semantics across lines and
        files. any line numbers or column numbers are not allowed to be encoded
        into Symbol[ProgrammingLanguage] instances. the underlying trouble is how to express ast tree structure.
        the solution is create a tmp_var symbol as owner_tag to flatten the tree structure.
    }
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
    """Convert an ast.AST tree to the JSON dict format."""
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
# Temp var generator for flattening tree structure
# ---------------------------------------------------------------------------

class _TempVarGen:
    """Generate unique temp var symbols for complex expressions."""

    def __init__(self):
        self._counters: Dict[str, int] = {}

    def gen(self, prefix: str) -> str:
        """Generate a unique temp var symbol like $subscript_0."""
        if prefix not in self._counters:
            self._counters[prefix] = 0
        idx = self._counters[prefix]
        self._counters[prefix] += 1
        return f"${prefix}_{idx}"


# ---------------------------------------------------------------------------
# Symbol extraction — convert JSON AST node to Symbol (no lineno/colno)
# ---------------------------------------------------------------------------

def _node_to_symbol(node: Any, tmp_gen: _TempVarGen) -> str:
    """Convert a JSON AST node to a Symbol (no line/column numbers).

    For simple nodes, returns the semantic value directly.
    For complex expressions, returns a temp var symbol.
    """
    if node is None:
        return "None"
    if isinstance(node, (str, int, float, bool)):
        return repr(node).replace(" ", "")
    if not isinstance(node, dict):
        return str(node).replace(" ", "")

    t = node.get("_type", "")

    # Simple leaf nodes - return semantic value
    if t == "Name":
        return node["id"]
    if t == "Constant":
        v = node["value"]
        if v is ...:
            return "..."
        return _encode_symbol(repr(v))
    if t == "Starred":
        return tmp_gen.gen("starred")

    # Collection literals - use temp vars to avoid double-counting elements
    if t == "Tuple":
        return tmp_gen.gen("tuple")
    if t == "List":
        return tmp_gen.gen("list")
    if t == "Set":
        return tmp_gen.gen("set")

    # Complex expressions - return temp var symbol
    if t == "Attribute":
        value = node.get("value", {})
        value_t = value.get("_type", "") if isinstance(value, dict) else ""
        # Only inline simple chains: Name.attr or Name.a.b.c
        if value_t == "Name":
            return f"{value['id']}.{node['attr']}"
        if value_t == "Attribute":
            inner = _node_to_symbol(value, tmp_gen)
            # If inner is still a simple dotted name (no $), inline it
            if not inner.startswith("$"):
                return f"{inner}.{node['attr']}"
        # Complex value: use temp var to avoid compound symbols
        return tmp_gen.gen("attr")
    if t == "Subscript":
        return tmp_gen.gen("subscript")
    if t == "Call":
        return tmp_gen.gen("call")
    if t == "BinOp":
        return tmp_gen.gen("binop")
    if t == "UnaryOp":
        return tmp_gen.gen("unaryop")
    if t == "Compare":
        return tmp_gen.gen("compare")
    if t == "BoolOp":
        return tmp_gen.gen("boolop")
    if t == "Await":
        return tmp_gen.gen("await")
    if t == "Lambda":
        return tmp_gen.gen("lambda")
    if t == "IfExp":
        return tmp_gen.gen("ifexp")
    if t == "Yield":
        return tmp_gen.gen("yield")
    if t == "YieldFrom":
        return tmp_gen.gen("yieldfrom")
    if t == "Dict":
        return tmp_gen.gen("dict")
    if t == "Slice":
        return tmp_gen.gen("slice")
    if t in ("ListComp", "SetComp", "DictComp", "GeneratorExp"):
        return tmp_gen.gen("comp")
    if t == "NamedExpr":
        return tmp_gen.gen("walrus")
    if t == "JoinedStr":
        return tmp_gen.gen("fstring")

    # Default: use type name as temp var
    return tmp_gen.gen(t.lower())


def _get_op_symbol(node: Any) -> str:
    """Convert an operator node to its symbol string."""
    if isinstance(node, dict):
        t = node.get("_type", "")
        op_map = {
            "Add": "+", "Sub": "-", "Mult": "*", "Div": "/", "FloorDiv": "//",
            "Mod": "%", "Pow": "**", "LShift": "<<", "RShift": ">>",
            "BitOr": "|", "BitXor": "^", "BitAnd": "&", "MatMult": "@",
            "UAdd": "+", "USub": "-", "Not": "not", "Invert": "~",
            "Eq": "==", "NotEq": "!=", "Lt": "<", "LtE": "<=",
            "Gt": ">", "GtE": ">=", "Is": "is", "IsNot": "is%20not",
            "In": "in", "NotIn": "not%20in", "And": "and", "Or": "or"
        }
        return op_map.get(t, t)
    return str(node)


def _encode_symbol(s: str) -> str:
    """Encode a string as a Symbol (no space/tab/newline per constraint)."""
    return (s.replace("%", "%25")
             .replace(" ", "%20")
             .replace("\t", "%09")
             .replace("\n", "%0A")
             .replace("\r", "%0D"))


# ---------------------------------------------------------------------------
# Relation extraction - flatten AST tree into relations
#
# KEY: self_sym parameter ensures member_tags from parent == owner_tags here.
# When a parent does:
#     child_sym = _node_to_symbol(child, tmp_gen)
#     add_rel("assigns", target, [child_sym])
#     _extract(child, ..., self_sym=child_sym)
# the child uses child_sym as its owner_tag, so the graph is connected.
# ---------------------------------------------------------------------------

def _extract(node: Any, rels: List[Dict], tmp_gen: _TempVarGen,
             self_sym: Optional[str] = None):
    """Extract AstTagRelation records from a JSON AST node.

    self_sym: if provided, this node should use self_sym as its owner_tag
              instead of generating a new temp var. This connects the graph.
    """
    if not isinstance(node, dict):
        return

    t = node.get("_type", "")
    ln = node.get("_lineno", 0)

    def add_rel(tag: str, owner: str, members: List[str]):
        relations = []
        for m in members:
            relations.append({
                "line": ln,
                "relation_tag": tag,
                "owner_tag": owner,
                "member_tags": [m]
            })
        rels.extend(relations)

    def child_sym_and_extract(child_node: Any) -> str:
        """Generate symbol for child AND recursively extract its relations."""
        sym = _node_to_symbol(child_node, tmp_gen)
        _extract(child_node, rels, tmp_gen, self_sym=sym)
        return sym

    # For complex expression types, determine the owner symbol:
    # use self_sym if provided (from parent), otherwise generate new one
    def get_self_sym(prefix: str) -> str:
        if self_sym is not None:
            return self_sym
        return tmp_gen.gen(prefix)

    # Module
    if t == "Module":
        for stmt in node.get("body", []):
            stmt_sym = child_sym_and_extract(stmt)
            add_rel("contains", "<module>", [stmt_sym])

    # FunctionDef / AsyncFunctionDef
    elif t in ("FunctionDef", "AsyncFunctionDef"):
        name = node.get("name", "")
        me = self_sym or name
        # Connect self_sym (from parent's contains) to the function name
        if me != name:
            add_rel("defines", me, [name])

        if t == "AsyncFunctionDef":
            add_rel("async_def", me, ["async"])

        # Decorators
        for dec in node.get("decorator_list", []):
            dec_sym = child_sym_and_extract(dec)
            add_rel("decorates", dec_sym, [name])
            # Emit specific decorator types
            if dec_sym == "staticmethod":
                add_rel("static_method", me, ["staticmethod"])
            elif dec_sym == "classmethod":
                add_rel("class_method", me, ["classmethod"])
            elif dec_sym == "property":
                add_rel("property_def", me, ["property"])

        # Parameters — all scoped to `me` for uniqueness
        args = node.get("args", {})
        all_positional = args.get("posonlyargs", []) + args.get("args", [])
        defaults = args.get("defaults", [])
        defaults_offset = len(all_positional) - len(defaults)

        for i, arg in enumerate(all_positional):
            arg_name = arg.get("arg", "")
            add_rel("param", me, [arg_name])
            if arg.get("annotation"):
                ann_sym = child_sym_and_extract(arg["annotation"])
                rels.append({"line": ln, "relation_tag": "param_annotation",
                             "owner_tag": me, "member_tags": [arg_name, ann_sym]})
            di = i - defaults_offset
            if 0 <= di < len(defaults) and defaults[di] is not None:
                def_sym = child_sym_and_extract(defaults[di])
                rels.append({"line": ln, "relation_tag": "param_default",
                             "owner_tag": me, "member_tags": [arg_name, def_sym]})

        if args.get("vararg"):
            va = args["vararg"]
            va_name = va.get("arg", "")
            add_rel("star_param", me, [va_name])
            if va.get("annotation"):
                ann_sym = child_sym_and_extract(va["annotation"])
                rels.append({"line": ln, "relation_tag": "param_annotation",
                             "owner_tag": me, "member_tags": [va_name, ann_sym]})

        kw_defaults = args.get("kw_defaults", [])
        for i, arg in enumerate(args.get("kwonlyargs", [])):
            arg_name = arg.get("arg", "")
            add_rel("param", me, [arg_name])
            if arg.get("annotation"):
                ann_sym = child_sym_and_extract(arg["annotation"])
                rels.append({"line": ln, "relation_tag": "param_annotation",
                             "owner_tag": me, "member_tags": [arg_name, ann_sym]})
            if i < len(kw_defaults) and kw_defaults[i] is not None:
                def_sym = child_sym_and_extract(kw_defaults[i])
                rels.append({"line": ln, "relation_tag": "param_default",
                             "owner_tag": me, "member_tags": [arg_name, def_sym]})

        if args.get("kwarg"):
            ka = args["kwarg"]
            ka_name = ka.get("arg", "")
            add_rel("double_star_param", me, [ka_name])
            if ka.get("annotation"):
                ann_sym = child_sym_and_extract(ka["annotation"])
                rels.append({"line": ln, "relation_tag": "param_annotation",
                             "owner_tag": me, "member_tags": [ka_name, ann_sym]})

        # Return annotation
        if node.get("returns"):
            ret_sym = child_sym_and_extract(node["returns"])
            add_rel("returns", me, [ret_sym])

        # Body — check for ellipsis body
        body = node.get("body", [])
        if (len(body) == 1 and body[0].get("_type") == "Expr"
                and body[0].get("value", {}).get("_type") == "Constant"
                and body[0].get("value", {}).get("value") is ...):
            add_rel("ellipsis_body", me, ["..."])
            return

        # Docstring
        if body and body[0].get("_type") == "Expr":
            val = body[0].get("value", {})
            if val.get("_type") == "Constant" and isinstance(val.get("value"), str):
                doc_str = val["value"]
                doc_sym = _encode_symbol(doc_str)
                add_rel("docstring", me, [doc_sym])

        # Body statements
        for i, stmt in enumerate(body):
            stmt_sym = child_sym_and_extract(stmt)
            add_rel("stmt_seq", me, [f"{i}:{stmt_sym}"])
            add_rel("contains", me, [stmt_sym])

    # ClassDef
    elif t == "ClassDef":
        name = node.get("name", "")
        me = self_sym or name
        if me != name:
            add_rel("defines", me, [name])

        for dec in node.get("decorator_list", []):
            dec_sym = child_sym_and_extract(dec)
            add_rel("decorates", dec_sym, [name])

        for base in node.get("bases", []):
            base_sym = child_sym_and_extract(base)
            add_rel("bases", me, [base_sym])

        for kw in node.get("keywords", []):
            if kw.get("arg") == "metaclass":
                mc_sym = child_sym_and_extract(kw["value"])
                add_rel("metaclass", me, [mc_sym])
            elif kw.get("arg"):
                kw_val_sym = child_sym_and_extract(kw["value"])
                add_rel("keyword_arg", me, [kw["arg"]])
                add_rel("keyword_arg", me, [kw_val_sym])

        # Docstring
        body = node.get("body", [])
        if body and body[0].get("_type") == "Expr":
            val = body[0].get("value", {})
            if val.get("_type") == "Constant" and isinstance(val.get("value"), str):
                doc_str = val["value"]
                doc_sym = _encode_symbol(doc_str)
                add_rel("docstring", me, [doc_sym])

        for i, stmt in enumerate(body):
            stmt_sym = child_sym_and_extract(stmt)
            add_rel("stmt_seq", me, [f"{i}:{stmt_sym}"])
            add_rel("contains", me, [stmt_sym])

    # Assign
    elif t == "Assign":
        me = self_sym or "assign"
        value_sym = child_sym_and_extract(node.get("value"))
        for tgt in node.get("targets", []):
            tgt_sym = child_sym_and_extract(tgt)
            add_rel("assigns", me, [tgt_sym])
            add_rel("assigns", me, [value_sym])

    # AugAssign
    elif t == "AugAssign":
        me = self_sym or "augassign"
        target_sym = child_sym_and_extract(node.get("target"))
        value_sym = child_sym_and_extract(node.get("value"))
        op_sym = _get_op_symbol(node.get("op"))
        add_rel("aug_assigns", me, [target_sym])
        add_rel("aug_assigns", me, [value_sym])
        add_rel("aug_op", me, [op_sym])

    # AnnAssign
    elif t == "AnnAssign":
        me = self_sym or "annassign"
        target_sym = child_sym_and_extract(node.get("target"))
        ann_sym = child_sym_and_extract(node.get("annotation"))
        add_rel("annotation", me, [target_sym])
        add_rel("annotation", me, [ann_sym])
        if node.get("value"):
            value_sym = child_sym_and_extract(node["value"])
            add_rel("assigns", me, [value_sym])

    # Import
    elif t == "Import":
        me = self_sym or "import"
        for alias in node.get("names", []):
            name = alias.get("name", "")
            add_rel("imports", me, [name])
            if alias.get("asname"):
                add_rel("aliases", me, [alias["asname"]])

    # ImportFrom
    elif t == "ImportFrom":
        me = self_sym or "importfrom"
        module = node.get("module") or ""
        add_rel("imports", me, [module])
        for alias in node.get("names", []):
            name = alias.get("name", "")
            add_rel("imports", me, [name])
            if alias.get("asname"):
                add_rel("aliases", me, [alias["asname"]])

    # Expr (expression statement)
    elif t == "Expr":
        me = self_sym or "expr"
        value = node.get("value", {})
        vt = value.get("_type", "") if isinstance(value, dict) else ""
        if vt == "Constant" and isinstance(value.get("value"), str):
            # docstring — mark as such so the graph stays connected
            doc_sym = _encode_symbol(str(value["value"]))
            add_rel("docstring", me, [doc_sym])
        elif vt == "Yield":
            yv_sym = child_sym_and_extract(value.get("value")) if value.get("value") else "None"
            add_rel("yields", me, [yv_sym])
        elif vt == "YieldFrom":
            yf_sym = child_sym_and_extract(value.get("value"))
            add_rel("yields_from", me, [yf_sym])
        else:
            expr_sym = child_sym_and_extract(value)
            add_rel("expr_stmt", me, [expr_sym])

    # Return
    elif t == "Return":
        me = self_sym or "return"
        if node.get("value") is not None:
            val_sym = child_sym_and_extract(node["value"])
            add_rel("returns", me, [val_sym])
        else:
            add_rel("bare_return", me, ["return"])

    # If
    elif t == "If":
        me = self_sym or tmp_gen.gen("if")
        test_sym = child_sym_and_extract(node.get("test"))
        add_rel("if_test", me, [test_sym])
        for stmt in node.get("body", []):
            stmt_sym = child_sym_and_extract(stmt)
            add_rel("if_body", me, [stmt_sym])
        for stmt in node.get("orelse", []):
            stmt_sym = child_sym_and_extract(stmt)
            add_rel("else_body", me, [stmt_sym])

    # For / AsyncFor
    elif t in ("For", "AsyncFor"):
        me = self_sym or tmp_gen.gen("for")
        if t == "AsyncFor":
            add_rel("async_for", me, ["async"])
        target_sym = child_sym_and_extract(node.get("target"))
        iter_sym = child_sym_and_extract(node.get("iter"))
        add_rel("for_target", me, [target_sym])
        add_rel("for_iter", me, [iter_sym])
        for stmt in node.get("body", []):
            stmt_sym = child_sym_and_extract(stmt)
            add_rel("for_body", me, [stmt_sym])
        for stmt in node.get("orelse", []):
            stmt_sym = child_sym_and_extract(stmt)
            add_rel("else_body", me, [stmt_sym])

    # While
    elif t == "While":
        me = self_sym or tmp_gen.gen("while")
        test_sym = child_sym_and_extract(node.get("test"))
        add_rel("while_test", me, [test_sym])
        for stmt in node.get("body", []):
            stmt_sym = child_sym_and_extract(stmt)
            add_rel("while_body", me, [stmt_sym])
        for stmt in node.get("orelse", []):
            stmt_sym = child_sym_and_extract(stmt)
            add_rel("else_body", me, [stmt_sym])

    # With / AsyncWith
    elif t in ("With", "AsyncWith"):
        me = self_sym or tmp_gen.gen("with")
        if t == "AsyncWith":
            add_rel("async_with", me, ["async"])
        for item in node.get("items", []):
            ctx = item.get("context_expr")
            ctx_sym = child_sym_and_extract(ctx)
            add_rel("with_context", me, [ctx_sym])
            if item.get("optional_vars"):
                vars_sym = child_sym_and_extract(item["optional_vars"])
                add_rel("with_as", ctx_sym, [vars_sym])
        for stmt in node.get("body", []):
            stmt_sym = child_sym_and_extract(stmt)
            add_rel("with_body", me, [stmt_sym])

    # Try / TryStar
    elif t in ("Try", "TryStar"):
        me = self_sym or tmp_gen.gen("try")
        add_rel("try_start", me, ["try"])
        for stmt in node.get("body", []):
            stmt_sym = child_sym_and_extract(stmt)
            add_rel("try_body", me, [stmt_sym])
        for handler in node.get("handlers", []):
            if handler.get("type"):
                exc_sym = child_sym_and_extract(handler["type"])
                add_rel("handles", me, [exc_sym])
                if handler.get("name"):
                    add_rel("except_as", exc_sym, [handler["name"]])
            for stmt in handler.get("body", []):
                stmt_sym = child_sym_and_extract(stmt)
                add_rel("except_body", me, [stmt_sym])
        for stmt in node.get("orelse", []):
            stmt_sym = child_sym_and_extract(stmt)
            add_rel("try_else", me, [stmt_sym])
        for stmt in node.get("finalbody", []):
            stmt_sym = child_sym_and_extract(stmt)
            add_rel("finally_body", me, [stmt_sym])

    # Raise
    elif t == "Raise":
        me = self_sym or "raise"
        if node.get("exc"):
            exc_sym = child_sym_and_extract(node["exc"])
            add_rel("raises", me, [exc_sym])
            if node.get("cause"):
                cause_sym = child_sym_and_extract(node["cause"])
                add_rel("raises_from", exc_sym, [cause_sym])

    # Call
    elif t == "Call":
        me = get_self_sym("call")
        func_sym = child_sym_and_extract(node.get("func"))
        add_rel("calls", me, [func_sym])
        for arg in node.get("args", []):
            if isinstance(arg, dict) and arg.get("_type") == "Starred":
                arg_sym = child_sym_and_extract(arg.get("value"))
                add_rel("call_pos_arg", me, ["*" + arg_sym])
            else:
                arg_sym = child_sym_and_extract(arg)
                add_rel("call_pos_arg", me, [arg_sym])
        for kw in node.get("keywords", []):
            val_sym = child_sym_and_extract(kw.get("value"))
            if kw.get("arg"):
                add_rel("keyword_arg", me, [kw["arg"]])
                add_rel("keyword_arg", me, [val_sym])
            else:
                add_rel("double_star_arg", me, [val_sym])

    # Subscript
    elif t == "Subscript":
        me = get_self_sym("subscript")
        value_sym = child_sym_and_extract(node.get("value"))
        slice_sym = child_sym_and_extract(node.get("slice"))
        add_rel("subscript_value", me, [value_sym])
        add_rel("subscript", me, [slice_sym])

    # Slice
    elif t == "Slice":
        me = get_self_sym("slice")
        has_parts = False
        if node.get("lower"):
            lo_sym = child_sym_and_extract(node["lower"])
            add_rel("slice_lower", me, [lo_sym])
            has_parts = True
        if node.get("upper"):
            up_sym = child_sym_and_extract(node["upper"])
            add_rel("slice_upper", me, [up_sym])
            has_parts = True
        if node.get("step"):
            st_sym = child_sym_and_extract(node["step"])
            add_rel("slice_step", me, [st_sym])
            has_parts = True
        if not has_parts:
            # Empty slice (:) — emit marker so graph stays connected
            add_rel("slice_marker", me, ["slice"])

    # BinOp
    elif t == "BinOp":
        me = get_self_sym("binop")
        op_sym = _get_op_symbol(node.get("op"))
        left_sym = child_sym_and_extract(node.get("left"))
        right_sym = child_sym_and_extract(node.get("right"))
        add_rel("bin_op", me, [op_sym])
        add_rel("bin_op_left", me, [left_sym])
        add_rel("bin_op_right", me, [right_sym])

    # UnaryOp
    elif t == "UnaryOp":
        me = get_self_sym("unaryop")
        op_sym = _get_op_symbol(node.get("op"))
        operand_sym = child_sym_and_extract(node.get("operand"))
        add_rel("unary_op", me, [op_sym])
        add_rel("unary_op_operand", me, [operand_sym])

    # Compare
    elif t == "Compare":
        me = get_self_sym("compare")
        left_sym = child_sym_and_extract(node.get("left"))
        add_rel("compare_left", me, [left_sym])
        for op, comp in zip(node.get("ops", []), node.get("comparators", [])):
            op_sym = _get_op_symbol(op)
            comp_sym = child_sym_and_extract(comp)
            add_rel("compare_op", me, [op_sym])
            add_rel("compare_right", me, [comp_sym])

    # BoolOp
    elif t == "BoolOp":
        me = get_self_sym("boolop")
        op_sym = _get_op_symbol(node.get("op"))
        add_rel("bool_op", me, [op_sym])
        for val in node.get("values", []):
            val_sym = child_sym_and_extract(val)
            add_rel("bool_op_operand", me, [val_sym])

    # IfExp (ternary)
    elif t == "IfExp":
        me = get_self_sym("ifexp")
        test_sym = child_sym_and_extract(node.get("test"))
        body_sym = child_sym_and_extract(node.get("body"))
        else_sym = child_sym_and_extract(node.get("orelse"))
        add_rel("if_expr_test", me, [test_sym])
        add_rel("if_expr_body", me, [body_sym])
        add_rel("if_expr_else", me, [else_sym])

    # Lambda
    elif t == "Lambda":
        me = get_self_sym("lambda")
        args = node.get("args", {})
        all_positional = args.get("posonlyargs", []) + args.get("args", [])
        defaults = args.get("defaults", [])
        defaults_offset = len(all_positional) - len(defaults)
        for i, arg in enumerate(all_positional):
            arg_name = arg.get("arg", "")
            add_rel("param", me, [arg_name])
            di = i - defaults_offset
            if 0 <= di < len(defaults) and defaults[di] is not None:
                def_sym = child_sym_and_extract(defaults[di])
                rels.append({"line": ln, "relation_tag": "param_default",
                             "owner_tag": me, "member_tags": [arg_name, def_sym]})
        if args.get("vararg"):
            va = args["vararg"]
            add_rel("star_param", me, [va.get("arg", "")])
        kw_defaults = args.get("kw_defaults", [])
        for i, arg in enumerate(args.get("kwonlyargs", [])):
            arg_name = arg.get("arg", "")
            add_rel("param", me, [arg_name])
            if i < len(kw_defaults) and kw_defaults[i] is not None:
                def_sym = child_sym_and_extract(kw_defaults[i])
                rels.append({"line": ln, "relation_tag": "param_default",
                             "owner_tag": me, "member_tags": [arg_name, def_sym]})
        if args.get("kwarg"):
            ka = args["kwarg"]
            add_rel("double_star_param", me, [ka.get("arg", "")])
        if node.get("body"):
            body_sym = child_sym_and_extract(node["body"])
            add_rel("lambda_body", me, [body_sym])

    # Await
    elif t == "Await":
        me = get_self_sym("await")
        val_sym = child_sym_and_extract(node.get("value"))
        add_rel("await_value", me, [val_sym])

    # Dict
    elif t == "Dict":
        me = get_self_sym("dict")
        add_rel("dict_literal", me, ["dict"])
        for k, v in zip(node.get("keys", []), node.get("values", [])):
            v_sym = child_sym_and_extract(v)
            if k is not None:
                k_sym = child_sym_and_extract(k)
                add_rel("dict_key", me, [k_sym])
                add_rel("dict_value", me, [v_sym])
            else:
                add_rel("double_star_arg", me, [v_sym])

    # JoinedStr (f-string) — preserve interleaving order via fstring_elem
    elif t == "JoinedStr":
        me = get_self_sym("fstring")
        for val in node.get("values", []):
            if isinstance(val, dict):
                vt = val.get("_type", "")
                if vt == "Constant":
                    # Literal part: tag with "lit:" prefix
                    add_rel("fstring_elem", me,
                            ["lit:" + _encode_symbol(str(val.get("value", "")))])
                elif vt == "FormattedValue":
                    inner_sym = child_sym_and_extract(val.get("value"))
                    conversion = val.get("conversion", -1)
                    conv_str = ""
                    if conversion == ord("s"):
                        conv_str = "!s"
                    elif conversion == ord("r"):
                        conv_str = "!r"
                    elif conversion == ord("a"):
                        conv_str = "!a"
                    fmt_str = ""
                    fmt_spec = val.get("format_spec")
                    if fmt_spec and isinstance(fmt_spec, dict):
                        for fv in fmt_spec.get("values", []):
                            if isinstance(fv, dict) and fv.get("_type") == "Constant":
                                fmt_str += str(fv.get("value", ""))
                    suffix = f"{conv_str}{':%s' % fmt_str if fmt_str else ''}"
                    # Formatted value: tag with "val:" prefix + optional suffix
                    add_rel("fstring_elem", me,
                            [f"val:{inner_sym}" + (f"|{suffix}" if suffix else "")])

    # Comprehensions (ListComp, SetComp, DictComp, GeneratorExp)
    elif t in ("ListComp", "SetComp", "DictComp", "GeneratorExp"):
        me = get_self_sym("comp")
        # Tag which kind of comprehension
        comp_type_map = {
            "ListComp": "list_literal",
            "SetComp": "set_literal",
            "DictComp": "dict_literal",
            "GeneratorExp": "tuple_literal",
        }
        add_rel(comp_type_map[t], me, [t])

        if t == "DictComp":
            key_sym = child_sym_and_extract(node.get("key"))
            val_sym = child_sym_and_extract(node.get("value"))
            add_rel("comprehension_body", me, [key_sym])
            add_rel("comprehension_body", me, [val_sym])
        else:
            elt_sym = child_sym_and_extract(node.get("elt"))
            add_rel("comprehension_body", me, [elt_sym])

        for gen in node.get("generators", []):
            tgt_sym = child_sym_and_extract(gen.get("target"))
            iter_sym = child_sym_and_extract(gen.get("iter"))
            add_rel("comprehension_target", me, [tgt_sym])
            add_rel("comprehension_iter", me, [iter_sym])
            for if_cond in gen.get("ifs", []):
                if_sym = child_sym_and_extract(if_cond)
                add_rel("comprehension_if", me, [if_sym])

    # NamedExpr (walrus)
    elif t == "NamedExpr":
        me = get_self_sym("walrus")
        target_sym = child_sym_and_extract(node.get("target"))
        value_sym = child_sym_and_extract(node.get("value"))
        add_rel("walrus", me, [target_sym])
        add_rel("walrus", me, [value_sym])

    # Pass / Break / Continue
    elif t == "Pass":
        me = self_sym or "pass"
        add_rel("pass_stmt", me, ["pass"])
    elif t == "Break":
        me = self_sym or "break"
        add_rel("break_stmt", me, ["break"])
    elif t == "Continue":
        me = self_sym or "continue"
        add_rel("continue_stmt", me, ["continue"])

    # Global / Nonlocal
    elif t == "Global":
        me = self_sym or "global"
        for name in node.get("names", []):
            add_rel("global_decl", me, [name])
    elif t == "Nonlocal":
        me = self_sym or "nonlocal"
        for name in node.get("names", []):
            add_rel("nonlocal_decl", me, [name])

    # Assert
    elif t == "Assert":
        me = self_sym or "assert"
        test_sym = child_sym_and_extract(node.get("test"))
        add_rel("assert_test", me, [test_sym])
        if node.get("msg"):
            msg_sym = child_sym_and_extract(node["msg"])
            add_rel("assert_msg", me, [msg_sym])

    # Delete
    elif t == "Delete":
        me = self_sym or "del"
        for tgt in node.get("targets", []):
            tgt_sym = child_sym_and_extract(tgt)
            add_rel("del_target", me, [tgt_sym])

    # Attribute — emit value and attr as separate relations (no compound symbols)
    elif t == "Attribute":
        me = self_sym or tmp_gen.gen("attr")
        value_sym = child_sym_and_extract(node.get("value"))
        attr_name = node.get("attr", "")
        add_rel("attr_value", me, [value_sym])
        add_rel("attr_name", me, [attr_name])

    # Constant — leaf node, no sub-relations needed
    elif t == "Constant":
        pass

    # Name — leaf node
    elif t == "Name":
        pass

    # Starred
    elif t == "Starred":
        me = get_self_sym("starred")
        val_sym = child_sym_and_extract(node.get("value"))
        add_rel("starred_value", me, [val_sym])

    # Tuple / List / Set — emit element relations
    elif t in ("Tuple", "List", "Set"):
        me = get_self_sym(t.lower())
        type_tag_map = {"Tuple": "tuple_literal", "List": "list_literal", "Set": "set_literal"}
        add_rel(type_tag_map[t], me, [t])
        for e in node.get("elts", []):
            elt_sym = child_sym_and_extract(e)
            add_rel("collection_element", me, [elt_sym])

    # Fallback: recurse into children
    else:
        for key, val in node.items():
            if key.startswith("_"):
                continue
            if isinstance(val, dict) and "_type" in val:
                child_sym_and_extract(val)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict) and "_type" in item:
                        child_sym_and_extract(item)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_python_to_ast_tag_jsonl(source_or_dict: Any) -> str:
    """Convert Python source or AST dict to JSONL of AstTagRelationGroup records.

    Accepts either a source code string or a pre-parsed JSON AST dict.
    """
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
    CODEBASE_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "code_auto_encoder", "codebase",
    )

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
        jsonl = convert_python_to_ast_tag_jsonl(source)
        n = len(jsonl.strip().splitlines()) if jsonl.strip() else 0
        total_relations += n
        rel = os.path.relpath(path, CODEBASE_DIR)
        print(f"{rel}: {n} relations")

    print(f"\n--- total: {len(py_files)} files, {total_relations} relations ---")

    # Verify graph connectivity: member_tags that are temp vars should appear as owner_tags
    print("\n=== Graph connectivity check ===")
    sample_path = py_files[0] if py_files else None
    if sample_path:
        with open(sample_path) as fh:
            source = fh.read()
        jsonl = convert_python_to_ast_tag_jsonl(source)
        all_owners = set()
        all_members = set()
        for line in jsonl.strip().splitlines():
            r = json.loads(line)
            all_owners.add(r["owner_tag"])
            for m in r["member_tags"]:
                all_members.add(m)
        tmp_members = {m for m in all_members if m.startswith("$")}
        tmp_owners = {o for o in all_owners if o.startswith("$")}
        disconnected = tmp_members - tmp_owners
        print(f"  {os.path.relpath(sample_path, CODEBASE_DIR)}:")
        print(f"    tmp_var member_tags: {len(tmp_members)}")
        print(f"    tmp_var owner_tags:  {len(tmp_owners)}")
        print(f"    disconnected (member but never owner): {len(disconnected)}")
        if disconnected:
            for d in sorted(disconnected)[:10]:
                print(f"      {d}")
