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
import urllib.parse
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict


# ---------------------------------------------------------------------------
# Relation indexing
# ---------------------------------------------------------------------------

class _RelationIndex:
    """Index AstTagRelationGroup records for fast lookup by owner_tag+relation_tag.

    Preserves insertion order so that the forward converter's emission order
    (which mirrors AST walk order) can be recovered.
    """

    def __init__(self, groups: List[Dict]):
        # owner_tag -> relation_tag -> [member_tags...]
        self._data: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # Reverse: member_tag -> relation_tag -> [owner_tags...]
        self._reverse: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # Grouped: owner_tag -> relation_tag -> [[member_tags], ...]
        self._grouped: Dict[str, Dict[str, List[List[str]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for g in groups:
            owner = g.get("owner_tag", "")
            rel = g.get("relation_tag", "")
            members = g.get("member_tags", [])
            self._grouped[owner][rel].append(members)
            for m in members:
                self._data[owner][rel].append(m)
                self._reverse[m][rel].append(owner)

    def get(self, owner: str, rel_tag: str) -> List[str]:
        return self._data.get(owner, {}).get(rel_tag, [])

    def get_groups(self, owner: str, rel_tag: str) -> List[List[str]]:
        """Get grouped member_tags, preserving multi-member records."""
        return self._grouped.get(owner, {}).get(rel_tag, [])

    def has(self, owner: str) -> bool:
        return owner in self._data

    def tags(self, owner: str) -> Set[str]:
        return set(self._data.get(owner, {}).keys())

    def rev_get(self, member: str, rel_tag: str) -> List[str]:
        """Reverse lookup: find all owner_tags that point to member via rel_tag."""
        return self._reverse.get(member, {}).get(rel_tag, [])


# ---------------------------------------------------------------------------
# Symbol decoding
# ---------------------------------------------------------------------------

def _decode_symbol(s: str) -> str:
    """Decode a percent-encoded symbol back to its original string."""
    return urllib.parse.unquote(s)


def _format_collection_symbol(sym: str, idx: _RelationIndex, _depth: int = 0) -> str:
    """Format a collection symbol like (a,b) to (a, b) with proper spacing.

    Also recursively expands any $tmp_var elements inside the collection.
    """
    if not sym:
        return sym
    if (sym.startswith("(") and sym.endswith(")")) or \
       (sym.startswith("[") and sym.endswith("]")) or \
       (sym.startswith("{") and sym.endswith("}")):
        open_br = sym[0]
        close_br = sym[-1]
        inner = sym[1:-1]
        parts = _split_collection(inner)
        # Recursively expand each part
        formatted = []
        for p in parts:
            p = p.strip()
            if p.startswith("$"):
                formatted.append(_expr(p, idx, _depth + 1))
            elif p.startswith("*"):
                formatted.append(f"*{_format_collection_symbol(p[1:], idx, _depth + 1)}")
            else:
                formatted.append(_format_collection_symbol(p, idx, _depth + 1))
        return f"{open_br}{', '.join(formatted)}{close_br}"
    return sym


def _strip_tuple_parens(sym: str) -> str:
    """Strip outer parentheses from a tuple symbol used as a target."""
    if sym.startswith("(") and sym.endswith(")"):
        return sym[1:-1]
    return sym


def _split_collection(s: str) -> List[str]:
    """Split a comma-separated string respecting nested brackets."""
    parts = []
    depth = 0
    current = []
    for ch in s:
        if ch in "([{":
            depth += 1
            current.append(ch)
        elif ch in ")]}":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts

# ---------------------------------------------------------------------------
# Operator precedence for correct parenthesization
# ---------------------------------------------------------------------------

_OP_PRECEDENCE = {
    "or": 4, "and": 5, "not": 6,
    "in": 7, "not in": 7, "is": 7, "is not": 7,
    "<": 7, "<=": 7, ">": 7, ">=": 7, "!=": 7, "==": 7,
    "|": 8, "^": 9, "&": 10,
    "<<": 11, ">>": 11,
    "+": 12, "-": 12,
    "*": 13, "@": 13, "/": 13, "//": 13, "%": 13,
    "~": 14, "**": 15,
}


def _sym_precedence(sym: str, idx: _RelationIndex) -> int:
    """Get the precedence of an expression symbol. Higher = binds tighter."""
    if not sym.startswith("$"):
        return 100  # atoms have highest precedence
    if sym.startswith("$binop_"):
        op = idx.get(sym, "bin_op")
        if op:
            return _OP_PRECEDENCE.get(_decode_symbol(op[0]), 12)
    if sym.startswith("$boolop_"):
        op = idx.get(sym, "bool_op")
        if op:
            return _OP_PRECEDENCE.get(_decode_symbol(op[0]), 4)
    if sym.startswith("$compare_"):
        return 7
    if sym.startswith("$unaryop_"):
        op = idx.get(sym, "unary_op")
        if op:
            return _OP_PRECEDENCE.get(_decode_symbol(op[0]), 14)
    if sym.startswith("$ifexp_"):
        return 3
    if sym.startswith("$lambda_"):
        return 2
    if sym.startswith("$walrus_"):
        return 1
    return 100  # calls, subscripts, attrs etc have high precedence


def _paren_if_needed(inner_sym: str, parent_prec: int, idx: _RelationIndex,
                      _depth: int, right_assoc: bool = False) -> str:
    """Wrap expression in parens if its precedence is lower than parent."""
    inner_prec = _sym_precedence(inner_sym, idx)
    inner_str = _expr(inner_sym, idx, _depth + 1)
    if inner_prec < parent_prec or (inner_prec == parent_prec and right_assoc):
        return f"({inner_str})"
    return inner_str


# ---------------------------------------------------------------------------
# Expression reconstruction
# ---------------------------------------------------------------------------

def _expr(sym: str, idx: _RelationIndex, _depth: int = 0) -> str:
    """Reconstruct an expression string from a symbol."""
    if _depth > 200:
        return sym

    if not sym.startswith("$"):
        # Decode percent-encoding and format collection symbols
        decoded = _decode_symbol(sym)
        return _format_collection_symbol(decoded, idx, _depth)

    # Attribute: $attr_N
    if sym.startswith("$attr_"):
        val = idx.get(sym, "attr_value")
        name = idx.get(sym, "attr_name")
        if val and name:
            # Need parens around binop/compare/etc used as attr value
            v = _paren_if_needed(val[0], 100, idx, _depth)
            return f"{v}.{name[0]}"
        return sym

    # Subscript: $subscript_N
    if sym.startswith("$subscript_"):
        val = idx.get(sym, "subscript_value")
        sl = idx.get(sym, "subscript")
        if val and sl:
            v = _paren_if_needed(val[0], 100, idx, _depth)
            s = _expr(sl[0], idx, _depth + 1)
            s = _strip_tuple_parens(s)
            return f"{v}[{s}]"
        return sym

    # Call: $call_N
    if sym.startswith("$call_"):
        func = idx.get(sym, "calls")
        pos_args = idx.get(sym, "call_pos_arg")
        kw_members = idx.get(sym, "keyword_arg")
        dstar_args = idx.get(sym, "double_star_arg")

        f = _paren_if_needed(func[0], 100, idx, _depth) if func else "?"
        parts = []
        for a in pos_args:
            if a.startswith("*"):
                parts.append(f"*{_expr(a[1:], idx, _depth + 1)}")
            else:
                parts.append(_expr(a, idx, _depth + 1))
        # keyword_arg members come in pairs: name, value, name, value, ...
        i = 0
        while i + 1 < len(kw_members):
            kw_name = kw_members[i]
            kw_val = _expr(kw_members[i + 1], idx, _depth + 1)
            parts.append(f"{kw_name}={kw_val}")
            i += 2
        for ds in dstar_args:
            parts.append(f"**{_expr(ds, idx, _depth + 1)}")
        # Generator as sole argument: sum(x for x in y) not sum((x for x in y))
        if (len(parts) == 1 and len(pos_args) == 1
                and not pos_args[0].startswith("*")
                and not kw_members and not dstar_args
                and pos_args[0].startswith("$comp_")
                and idx.get(pos_args[0], "tuple_literal")):
            inner = parts[0]
            if inner.startswith("(") and inner.endswith(")"):
                inner = inner[1:-1]
            return f"{f}({inner})"
        return f"{f}({', '.join(parts)})"

    # BinOp: $binop_N
    if sym.startswith("$binop_"):
        op = idx.get(sym, "bin_op")
        left = idx.get(sym, "bin_op_left")
        right = idx.get(sym, "bin_op_right")
        if op and left and right:
            o = _decode_symbol(op[0])
            my_prec = _OP_PRECEDENCE.get(o, 12)
            l = _paren_if_needed(left[0], my_prec, idx, _depth)
            # Right side: same precedence needs parens for left-assoc operators
            # (except ** which is right-assoc)
            right_assoc = (o == "**")
            r = _paren_if_needed(right[0], my_prec, idx, _depth,
                                  right_assoc=not right_assoc)
            return f"{l} {o} {r}"
        return sym

    # UnaryOp: $unaryop_N
    if sym.startswith("$unaryop_"):
        op = idx.get(sym, "unary_op")
        operand = idx.get(sym, "unary_op_operand")
        if op and operand:
            o = _decode_symbol(op[0])
            my_prec = _OP_PRECEDENCE.get(o, 14)
            v = _paren_if_needed(operand[0], my_prec, idx, _depth)
            if o == "not":
                return f"not {v}"
            return f"{o}{v}"
        return sym

    # Compare: $compare_N
    if sym.startswith("$compare_"):
        left = idx.get(sym, "compare_left")
        ops = idx.get(sym, "compare_op")
        rights = idx.get(sym, "compare_right")
        if left:
            my_prec = 7
            result = _paren_if_needed(left[0], my_prec, idx, _depth)
            for o, r in zip(ops, rights):
                r_str = _paren_if_needed(r, my_prec, idx, _depth, right_assoc=True)
                result = f"{result} {_decode_symbol(o)} {r_str}"
            return result
        return sym

    # BoolOp: $boolop_N
    if sym.startswith("$boolop_"):
        op = idx.get(sym, "bool_op")
        operands = idx.get(sym, "bool_op_operand")
        if op and operands:
            o = _decode_symbol(op[0])
            my_prec = _OP_PRECEDENCE.get(o, 4)
            parts = [_paren_if_needed(p, my_prec, idx, _depth) for p in operands]
            return f" {o} ".join(parts)
        return sym

    # Await: $await_N
    if sym.startswith("$await_"):
        val = idx.get(sym, "await_value")
        if val:
            return f"await {_expr(val[0], idx, _depth + 1)}"
        return sym

    # Lambda: $lambda_N
    if sym.startswith("$lambda_"):
        params = idx.get(sym, "param")
        star = idx.get(sym, "star_param")
        dstar = idx.get(sym, "double_star_param")
        body = idx.get(sym, "lambda_body")
        # Build default map
        lam_defaults: Dict[str, str] = {}
        for group in idx.get_groups(sym, "param_default"):
            if len(group) >= 2:
                lam_defaults[group[0]] = group[1]
        param_parts = []
        for p in params:
            part = p
            if p in lam_defaults:
                part += f"={_expr(lam_defaults[p], idx, _depth + 1)}"
            param_parts.append(part)
        if star:
            param_parts.append(f"*{star[0]}")
        if dstar:
            param_parts.append(f"**{dstar[0]}")
        p = ", ".join(param_parts)
        b = _expr(body[0], idx, _depth + 1) if body else "None"
        if p:
            return f"lambda {p}: {b}"
        return f"lambda: {b}"

    # IfExp: $ifexp_N
    if sym.startswith("$ifexp_"):
        test = idx.get(sym, "if_expr_test")
        body = idx.get(sym, "if_expr_body")
        orelse = idx.get(sym, "if_expr_else")
        if test and body and orelse:
            t = _expr(test[0], idx, _depth + 1)
            b = _expr(body[0], idx, _depth + 1)
            e = _expr(orelse[0], idx, _depth + 1)
            return f"{b} if {t} else {e}"
        return sym

    # Yield: $yield_N
    if sym.startswith("$yield_"):
        val = idx.get(sym, "yields")
        if val:
            return f"yield {_expr(val[0], idx, _depth + 1)}"
        return "yield"

    # YieldFrom: $yieldfrom_N
    if sym.startswith("$yieldfrom_"):
        val = idx.get(sym, "yields_from")
        if val:
            return f"yield from {_expr(val[0], idx, _depth + 1)}"
        return sym

    # Dict: $dict_N
    if sym.startswith("$dict_"):
        keys = idx.get(sym, "dict_key")
        vals = idx.get(sym, "dict_value")
        dstar = idx.get(sym, "double_star_arg")
        pairs = []
        for k, v in zip(keys, vals):
            pairs.append(f"{_expr(k, idx, _depth + 1)}: {_expr(v, idx, _depth + 1)}")
        for ds in dstar:
            pairs.append(f"**{_expr(ds, idx, _depth + 1)}")
        return "{" + ", ".join(pairs) + "}"

    # Slice: $slice_N
    if sym.startswith("$slice_"):
        lower = idx.get(sym, "slice_lower")
        upper = idx.get(sym, "slice_upper")
        step = idx.get(sym, "slice_step")
        l = _expr(lower[0], idx, _depth + 1) if lower else ""
        u = _expr(upper[0], idx, _depth + 1) if upper else ""
        s = _expr(step[0], idx, _depth + 1) if step else ""
        if s:
            return f"{l}:{u}:{s}"
        return f"{l}:{u}"

    # Starred: $starred_N
    if sym.startswith("$starred_"):
        val = idx.get(sym, "starred_value")
        if val:
            return f"*{_expr(val[0], idx, _depth + 1)}"
        return sym

    # Tuple: $tuple_N
    if sym.startswith("$tuple_"):
        elts = idx.get(sym, "collection_element")
        parts = [_expr(e, idx, _depth + 1) for e in elts]
        if len(parts) == 1:
            return f"({parts[0]},)"
        return f"({', '.join(parts)})"

    # List: $list_N
    if sym.startswith("$list_"):
        elts = idx.get(sym, "collection_element")
        parts = [_expr(e, idx, _depth + 1) for e in elts]
        return f"[{', '.join(parts)}]"

    # Set: $set_N
    if sym.startswith("$set_"):
        elts = idx.get(sym, "collection_element")
        parts = [_expr(e, idx, _depth + 1) for e in elts]
        return "{" + ", ".join(parts) + "}"

    # Comprehension: $comp_N
    if sym.startswith("$comp_"):
        # Determine comprehension type
        comp_type = None
        for ct in ("list_literal", "set_literal", "dict_literal", "tuple_literal"):
            if idx.get(sym, ct):
                comp_type = ct
                break
        if not comp_type:
            comp_type = "list_literal"

        body_parts = idx.get(sym, "comprehension_body")
        targets = idx.get(sym, "comprehension_target")
        iters = idx.get(sym, "comprehension_iter")
        ifs = idx.get(sym, "comprehension_if")

        gen_parts = []
        for tgt, it in zip(targets, iters):
            tgt_str = _strip_tuple_parens(_expr(tgt, idx, _depth + 1))
            clause = f"for {tgt_str} in {_expr(it, idx, _depth + 1)}"
            gen_parts.append(clause)
        for if_cond in ifs:
            gen_parts.append(f"if {_expr(if_cond, idx, _depth + 1)}")
        gen_str = " ".join(gen_parts)

        if comp_type == "dict_literal" and len(body_parts) >= 2:
            k = _expr(body_parts[0], idx, _depth + 1)
            v = _expr(body_parts[1], idx, _depth + 1)
            return "{" + f"{k}: {v} {gen_str}" + "}"
        elif comp_type == "set_literal":
            elt = _expr(body_parts[0], idx, _depth + 1) if body_parts else "?"
            return "{" + f"{elt} {gen_str}" + "}"
        elif comp_type == "tuple_literal":
            elt = _expr(body_parts[0], idx, _depth + 1) if body_parts else "?"
            return f"({elt} {gen_str})"
        else:
            elt = _expr(body_parts[0], idx, _depth + 1) if body_parts else "?"
            return f"[{elt} {gen_str}]"

    # NamedExpr (walrus): $walrus_N
    if sym.startswith("$walrus_"):
        members = idx.get(sym, "walrus")
        if len(members) >= 2:
            tgt = _expr(members[0], idx, _depth + 1)
            val = _expr(members[1], idx, _depth + 1)
            return f"({tgt} := {val})"
        return sym

    # f-string: $fstring_N
    if sym.startswith("$fstring_"):
        return _reconstruct_fstring(sym, idx, _depth)

    return sym


def _reconstruct_fstring(sym: str, idx: _RelationIndex, _depth: int) -> str:
    """Reconstruct an f-string from its ordered fstring_elem entries."""
    elems = idx.get(sym, "fstring_elem")

    result_parts = []
    for elem in elems:
        if elem.startswith("lit:"):
            # Literal part
            lit = _decode_symbol(elem[4:])
            # Escape braces in literals
            lit = lit.replace("{", "{{").replace("}", "}}")
            # Escape special characters for string literal
            lit = (lit.replace("\\", "\\\\")
                      .replace("\n", "\\n")
                      .replace("\r", "\\r")
                      .replace("\t", "\\t"))
            result_parts.append(lit)
        elif elem.startswith("val:"):
            # Formatted value — may have |suffix for conversion/format spec
            rest = elem[4:]
            suffix = ""
            pipe_pos = rest.find("|")
            if pipe_pos >= 0:
                suffix = rest[pipe_pos + 1:]
                rest = rest[:pipe_pos]
            v = _expr(rest, idx, _depth + 1)
            result_parts.append("{" + v + suffix + "}")

    content = "".join(result_parts)
    # Choose quote style that avoids conflicts
    if '"' not in content:
        return 'f"' + content + '"'
    if "'" not in content:
        return "f'" + content + "'"
    # Both quotes present — escape double quotes and use double-quote style
    content = content.replace('"', '\\"')
    return 'f"' + content + '"'


# ---------------------------------------------------------------------------
# Statement reconstruction
# ---------------------------------------------------------------------------

def _is_def_or_class(sym: str, idx: _RelationIndex) -> bool:
    """Check if a symbol represents a function/class definition."""
    tags = idx.tags(sym)
    if "defines" in tags:
        return True
    if "param" in tags or "star_param" in tags or "double_star_param" in tags:
        return True
    if "ellipsis_body" in tags:
        return True
    if "bases" in tags:
        return True
    # Detect by prefix (forward converter uses $functiondef_N / $classdef_N)
    if sym.startswith("$functiondef_") or sym.startswith("$classdef_"):
        return True
    return False


def _stmt(sym: str, idx: _RelationIndex, indent: str = "") -> str:
    """Reconstruct a statement from a symbol."""
    tags = idx.tags(sym)

    # defines — wrapper symbol $functiondef_N or $classdef_N
    if "defines" in tags and "assigns" not in tags:
        defined = idx.get(sym, "defines")
        if defined:
            name = defined[0]
            if sym.startswith("$classdef_"):
                return _reconstruct_classdef(name, idx, indent, scope=sym)
            else:
                return _reconstruct_funcdef(name, idx, indent, scope=sym)
        return ""

    # FunctionDef (direct name as owner — shouldn't normally happen at top level)
    if ("param" in tags or "star_param" in tags or "double_star_param" in tags or
            "ellipsis_body" in tags) and "assigns" not in tags:
        return _reconstruct_funcdef(sym, idx, indent, scope=sym)

    # ClassDef (direct name as owner)
    if "bases" in tags:
        return _reconstruct_classdef(sym, idx, indent, scope=sym)

    # Assign: $assign_N
    if "assigns" in tags and "aug_op" not in tags and "annotation" not in tags:
        members = idx.get(sym, "assigns")
        if len(members) >= 2:
            value = members[-1]
            targets = members[:-1]
            # Strip outer parens from tuple targets (ast.unparse convention)
            tgt_strs = [_strip_tuple_parens(_expr(t, idx)) for t in targets]
            val_str = _expr(value, idx)
            return f"{indent}{' = '.join(tgt_strs)} = {val_str}\n"
        return ""

    # AugAssign
    if "aug_assigns" in tags:
        aug = idx.get(sym, "aug_assigns")
        op = idx.get(sym, "aug_op")
        if len(aug) >= 2 and op:
            tgt = _expr(aug[0], idx)
            val = _expr(aug[1], idx)
            return f"{indent}{tgt} {op[0]}= {val}\n"
        return ""

    # AnnAssign
    if "annotation" in tags and "param" not in tags and "contains" not in tags:
        ann_members = idx.get(sym, "annotation")
        assign_members = idx.get(sym, "assigns")
        if len(ann_members) >= 2:
            tgt = _expr(ann_members[0], idx)
            ann = _expr(ann_members[1], idx)
            if assign_members:
                val = _expr(assign_members[0], idx)
                return f"{indent}{tgt}: {ann} = {val}\n"
            return f"{indent}{tgt}: {ann}\n"
        return ""

    # Import: $import_N
    if sym.startswith("$import_"):
        imp = idx.get(sym, "imports")
        aliases = idx.get(sym, "aliases")
        if imp:
            if aliases:
                return f"{indent}import {imp[0]} as {aliases[0]}\n"
            return f"{indent}import {imp[0]}\n"
        return ""

    # ImportFrom: $importfrom_N
    if sym.startswith("$importfrom_"):
        imp = idx.get(sym, "imports")
        aliases = idx.get(sym, "aliases")
        if imp:
            module = imp[0]
            names = imp[1:]
            if names:
                # Match aliases to names (aliases come in order)
                name_strs = []
                ai = 0
                for n in names:
                    if ai < len(aliases):
                        name_strs.append(f"{n} as {aliases[ai]}")
                        ai += 1
                    else:
                        name_strs.append(n)
                return f"{indent}from {module} import {', '.join(name_strs)}\n"
        return ""

    # Return
    # Return
    if "bare_return" in tags:
        return f"{indent}return\n"
    if "returns" in tags and "param" not in tags and "contains" not in tags:
        val = idx.get(sym, "returns")
        if val:
            v = _expr(val[0], idx)
            # Strip outer parens from tuple returns: return a, b not return (a, b)
            if val[0].startswith("$tuple_"):
                v = _strip_tuple_parens(v)
            return f"{indent}return {v}\n"
        return f"{indent}return\n"

    # Expr statement
    if "expr_stmt" in tags:
        val = idx.get(sym, "expr_stmt")
        if val:
            return f"{indent}{_expr(val[0], idx)}\n"
        return ""

    # Docstring (standalone Expr containing string)
    if "docstring" in tags and "contains" not in tags and "param" not in tags:
        doc = idx.get(sym, "docstring")
        if doc:
            text = _decode_symbol(doc[0])
            return f'{indent}"""{text}"""\n'
        return ""

    # Yield
    if "yields" in tags:
        val = idx.get(sym, "yields")
        if val and val[0] != "None":
            return f"{indent}yield {_expr(val[0], idx)}\n"
        return f"{indent}yield\n"

    # YieldFrom
    if "yields_from" in tags:
        val = idx.get(sym, "yields_from")
        if val:
            return f"{indent}yield from {_expr(val[0], idx)}\n"
        return ""

    # If
    if "if_test" in tags:
        return _reconstruct_if(sym, idx, indent)

    # For
    if "for_target" in tags:
        return _reconstruct_for(sym, idx, indent)

    # While
    if "while_test" in tags:
        return _reconstruct_while(sym, idx, indent)

    # With
    if "with_context" in tags:
        return _reconstruct_with(sym, idx, indent)

    # Try
    if "try_start" in tags:
        return _reconstruct_try(sym, idx, indent)

    # Raise
    if "raises" in tags:
        exc = idx.get(sym, "raises")
        if exc:
            cause = idx.get(exc[0], "raises_from")
            exc_str = _expr(exc[0], idx)
            if cause:
                return f"{indent}raise {exc_str} from {_expr(cause[0], idx)}\n"
            return f"{indent}raise {exc_str}\n"
        return f"{indent}raise\n"

    # Pass / Break / Continue
    if "pass_stmt" in tags:
        return f"{indent}pass\n"
    if "break_stmt" in tags:
        return f"{indent}break\n"
    if "continue_stmt" in tags:
        return f"{indent}continue\n"

    # Global / Nonlocal
    if "global_decl" in tags:
        names = idx.get(sym, "global_decl")
        return f"{indent}global {', '.join(names)}\n"
    if "nonlocal_decl" in tags:
        names = idx.get(sym, "nonlocal_decl")
        return f"{indent}nonlocal {', '.join(names)}\n"

    # Assert
    if "assert_test" in tags:
        test = idx.get(sym, "assert_test")
        msg = idx.get(sym, "assert_msg")
        test_str = _expr(test[0], idx) if test else "True"
        if msg:
            return f"{indent}assert {test_str}, {_expr(msg[0], idx)}\n"
        return f"{indent}assert {test_str}\n"

    # Delete
    if "del_target" in tags:
        targets = idx.get(sym, "del_target")
        tgt_strs = [_expr(t, idx) for t in targets]
        return f"{indent}del {', '.join(tgt_strs)}\n"

    # Fallback: try as expression
    expr_str = _expr(sym, idx)
    if expr_str != sym:
        return f"{indent}{expr_str}\n"

    return ""


# ---------------------------------------------------------------------------
# Compound statement helpers
# ---------------------------------------------------------------------------

def _get_body_order(name: str, idx: _RelationIndex) -> List[str]:
    """Get ordered body statements using stmt_seq relations."""
    seq = idx.get(name, "stmt_seq")
    if seq:
        ordered = []
        for s in seq:
            colon_pos = s.find(":")
            if colon_pos >= 0:
                try:
                    n = int(s[:colon_pos])
                    ordered.append((n, s[colon_pos + 1:]))
                    continue
                except ValueError:
                    pass
            ordered.append((0, s))
        ordered.sort(key=lambda x: x[0])
        return [sym for _, sym in ordered]
    return idx.get(name, "contains")


def _reconstruct_body(syms: List[str], idx: _RelationIndex, indent: str) -> str:
    """Reconstruct a body block with proper blank lines before defs."""
    lines = []
    for i, s in enumerate(syms):
        # ast.unparse adds blank line before def/class
        if i > 0 and _is_def_or_class(s, idx):
            lines.append(f"{indent}\n")
        lines.append(_stmt(s, idx, indent))
    return "".join(lines)


def _reconstruct_funcdef(name: str, idx: _RelationIndex, indent: str = "",
                         scope: Optional[str] = None) -> str:
    """Reconstruct a function definition.

    scope: the unique owner symbol ($functiondef_N) for looking up params/body/etc.
    """
    s = scope or name
    is_async = bool(idx.get(s, "async_def"))
    prefix = "async def" if is_async else "def"

    lines = []

    # Decorators: find all symbols that have "decorates" relation with this name as member
    decorator_syms = idx.rev_get(name, "decorates")
    for dec_sym in decorator_syms:
        lines.append(f"{indent}@{_expr(dec_sym, idx)}\n")

    # Build param annotation and default maps from grouped member_tags
    param_anns: Dict[str, str] = {}
    for group in idx.get_groups(s, "param_annotation"):
        if len(group) >= 2:
            param_anns[group[0]] = group[1]
    param_defaults: Dict[str, str] = {}
    for group in idx.get_groups(s, "param_default"):
        if len(group) >= 2:
            param_defaults[group[0]] = group[1]

    # Parameters
    params = idx.get(s, "param")
    star = idx.get(s, "star_param")
    dstar = idx.get(s, "double_star_param")

    param_parts = []
    for p in params:
        part = p
        if p in param_anns:
            part = f"{p}: {_expr(param_anns[p], idx)}"
        if p in param_defaults:
            part += f"={_expr(param_defaults[p], idx)}"
        param_parts.append(part)

    if star:
        s_name = star[0]
        if s_name in param_anns:
            param_parts.append(f"*{s_name}: {_expr(param_anns[s_name], idx)}")
        else:
            param_parts.append(f"*{s_name}")

    if dstar:
        d_name = dstar[0]
        if d_name in param_anns:
            param_parts.append(f"**{d_name}: {_expr(param_anns[d_name], idx)}")
        else:
            param_parts.append(f"**{d_name}")

    sig = f"{prefix} {name}({', '.join(param_parts)})"

    ret = idx.get(s, "returns")
    if ret:
        sig += f" -> {_expr(ret[0], idx)}"
    sig += ":"

    # Check for ellipsis body
    if idx.get(s, "ellipsis_body"):
        lines.append(f"{indent}{sig}\n")
        lines.append(f"{indent}    ...\n")
        return "".join(lines)

    lines.append(f"{indent}{sig}\n")

    # Docstring
    docs = idx.get(s, "docstring")

    # Body statements (ordered by stmt_seq)
    body = _get_body_order(s, idx)

    # Skip docstring Expr if already handled by docstring relation
    skip_first = False
    if body and docs:
        first_sym = body[0]
        if idx.get(first_sym, "docstring"):
            skip_first = True

    body_stmts = body[1:] if skip_first else body

    if docs:
        doc_text = _decode_symbol(docs[0])
        lines.append(f'{indent}    """{doc_text}"""\n')

    lines.append(_reconstruct_body(body_stmts, idx, indent + "    "))

    if not body and not docs:
        lines.append(f"{indent}    pass\n")

    return "".join(lines)


def _reconstruct_classdef(name: str, idx: _RelationIndex, indent: str = "",
                          scope: Optional[str] = None) -> str:
    """Reconstruct a class definition.

    scope: the unique owner symbol ($classdef_N) for looking up bases/body/etc.
    """
    s = scope or name
    lines = []

    # Decorators
    decorator_syms = idx.rev_get(name, "decorates")
    for dec_sym in decorator_syms:
        lines.append(f"{indent}@{_expr(dec_sym, idx)}\n")

    bases = idx.get(s, "bases")
    kw_members = idx.get(s, "keyword_arg")
    metaclass = idx.get(s, "metaclass")

    parts = [_expr(b, idx) for b in bases]
    i = 0
    while i + 1 < len(kw_members):
        kw_name = kw_members[i]
        kw_val = _expr(kw_members[i + 1], idx)
        parts.append(f"{kw_name}={kw_val}")
        i += 2
    if metaclass:
        parts.append(f"metaclass={_expr(metaclass[0], idx)}")

    header = f"class {name}"
    if parts:
        header += f"({', '.join(parts)})"
    header += ":"

    lines.append(f"{indent}{header}\n")

    docs = idx.get(s, "docstring")
    body = _get_body_order(s, idx)

    skip_first = False
    if body and docs:
        first_sym = body[0]
        if idx.get(first_sym, "docstring"):
            skip_first = True

    body_stmts = body[1:] if skip_first else body

    if docs:
        doc_text = _decode_symbol(docs[0])
        lines.append(f'{indent}    """{doc_text}"""\n')

    lines.append(_reconstruct_body(body_stmts, idx, indent + "    "))

    if not body and not docs:
        lines.append(f"{indent}    pass\n")

    return "".join(lines)


def _reconstruct_if(sym: str, idx: _RelationIndex, indent: str) -> str:
    """Reconstruct an if/elif/else statement."""
    test = idx.get(sym, "if_test")
    body_syms = idx.get(sym, "if_body")
    else_syms = idx.get(sym, "else_body")

    lines = [f"{indent}if {_expr(test[0], idx) if test else 'True'}:\n"]
    lines.append(_reconstruct_body(body_syms, idx, indent + "    "))

    if else_syms:
        if len(else_syms) == 1 and "if_test" in idx.tags(else_syms[0]):
            # elif chain
            elif_code = _reconstruct_if(else_syms[0], idx, indent)
            elif_code = elif_code.replace(f"{indent}if ", f"{indent}elif ", 1)
            lines.append(elif_code)
        else:
            lines.append(f"{indent}else:\n")
            lines.append(_reconstruct_body(else_syms, idx, indent + "    "))

    return "".join(lines)


def _reconstruct_for(sym: str, idx: _RelationIndex, indent: str) -> str:
    """Reconstruct a for loop."""
    target = idx.get(sym, "for_target")
    iter_sym = idx.get(sym, "for_iter")
    body_syms = idx.get(sym, "for_body")
    else_syms = idx.get(sym, "else_body")

    tgt = _strip_tuple_parens(_expr(target[0], idx)) if target else "_"
    it = _expr(iter_sym[0], idx) if iter_sym else "[]"

    is_async = bool(idx.get(sym, "async_for"))
    prefix = "async for" if is_async else "for"
    lines = [f"{indent}{prefix} {tgt} in {it}:\n"]
    lines.append(_reconstruct_body(body_syms, idx, indent + "    "))

    if else_syms:
        lines.append(f"{indent}else:\n")
        lines.append(_reconstruct_body(else_syms, idx, indent + "    "))

    return "".join(lines)


def _reconstruct_while(sym: str, idx: _RelationIndex, indent: str) -> str:
    """Reconstruct a while loop."""
    test = idx.get(sym, "while_test")
    body_syms = idx.get(sym, "while_body")
    else_syms = idx.get(sym, "else_body")

    lines = [f"{indent}while {_expr(test[0], idx) if test else 'True'}:\n"]
    lines.append(_reconstruct_body(body_syms, idx, indent + "    "))

    if else_syms:
        lines.append(f"{indent}else:\n")
        lines.append(_reconstruct_body(else_syms, idx, indent + "    "))

    return "".join(lines)


def _reconstruct_with(sym: str, idx: _RelationIndex, indent: str) -> str:
    """Reconstruct a with statement."""
    ctx_syms = idx.get(sym, "with_context")
    body_syms = idx.get(sym, "with_body")

    items = []
    for ctx_s in ctx_syms:
        as_syms = idx.get(ctx_s, "with_as")
        ctx_str = _expr(ctx_s, idx)
        if as_syms:
            ctx_str += f" as {_expr(as_syms[0], idx)}"
        items.append(ctx_str)

    is_async = bool(idx.get(sym, "async_with"))
    prefix = "async with" if is_async else "with"
    lines = [f"{indent}{prefix} {', '.join(items)}:\n"]
    lines.append(_reconstruct_body(body_syms, idx, indent + "    "))

    return "".join(lines)


def _reconstruct_try(sym: str, idx: _RelationIndex, indent: str) -> str:
    """Reconstruct a try statement."""
    body_syms = idx.get(sym, "try_body")
    handlers = idx.get(sym, "handles")
    except_body = idx.get(sym, "except_body")
    else_syms = idx.get(sym, "try_else")
    finally_syms = idx.get(sym, "finally_body")

    lines = [f"{indent}try:\n"]
    lines.append(_reconstruct_body(body_syms, idx, indent + "    "))

    if handlers:
        for h in handlers:
            as_names = idx.get(h, "except_as")
            h_str = _expr(h, idx)
            if as_names:
                lines.append(f"{indent}except {h_str} as {as_names[0]}:\n")
            else:
                lines.append(f"{indent}except {h_str}:\n")
        lines.append(_reconstruct_body(except_body, idx, indent + "    "))

    if else_syms:
        lines.append(f"{indent}else:\n")
        lines.append(_reconstruct_body(else_syms, idx, indent + "    "))

    if finally_syms:
        lines.append(f"{indent}finally:\n")
        lines.append(_reconstruct_body(finally_syms, idx, indent + "    "))

    return "".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_jsonl_to_python(groups: List[Dict]) -> str:
    """Convert AstTagRelationGroup records to Python source code."""
    idx = _RelationIndex(groups)
    top_level = idx.get("<module>", "contains")

    parts = []
    for i, sym in enumerate(top_level):
        # ast.unparse adds blank line before def/class at module level
        if i > 0 and _is_def_or_class(sym, idx):
            parts.append("\n")
        code = _stmt(sym, idx, "")
        parts.append(code)

    result = "".join(parts)
    return result.rstrip("\n")


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

if __name__ == "__main__":
    import convert_python_to_ast_tag_jsonl as fwd
    import black

    def _normalize(code: str) -> str:
        """Normalize Python code through ast.unparse then black for comparison.

        Using ast.unparse first ensures both sides lose the same information
        (escape sequences, comments) that the AST doesn't preserve.
        """
        try:
            tree = ast_mod.parse(code)
            code = ast_mod.unparse(tree)
            return black.format_str(code, mode=black.Mode())
        except Exception:
            return code

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

        jsonl = fwd.convert_python_to_ast_tag_jsonl(source)
        groups = [json.loads(line) for line in jsonl.strip().splitlines()]
        reconstructed = convert_jsonl_to_python(groups)

        rel = os.path.relpath(path, CODEBASE_DIR)

        # Check if reconstructed code is valid Python
        try:
            ast_mod.parse(reconstructed)
        except SyntaxError as e:
            failed += 1
            parse_errors += 1
            errors.append(rel)
            print(f"  PARSE_ERROR {rel}: {e}")
            continue

        # Normalize both through black for comparison
        expected_norm = _normalize(source)
        reconstructed_norm = _normalize(reconstructed)

        if expected_norm == reconstructed_norm:
            passed += 1
            print(f"  PASS {rel}")
        else:
            failed += 1
            errors.append(rel)
            print(f"  FAIL {rel}")
            exp_lines = expected_norm.strip().splitlines()
            rec_lines = reconstructed_norm.strip().splitlines()
            for li, (e, r) in enumerate(zip(exp_lines, rec_lines)):
                if e != r:
                    print(f"    line {li+1}:")
                    print(f"      expected:      {e!r}")
                    print(f"      reconstructed: {r!r}")
                    break
            else:
                if len(exp_lines) != len(rec_lines):
                    print(f"    line count: expected={len(exp_lines)} reconstructed={len(rec_lines)}")

    print(f"\n--- {passed} passed, {failed} failed ({parse_errors} parse errors) out of {passed + failed} ---")
    print(f"all(source.py == reconstructed.py) {failed == 0}")
    if errors:
        print(f"\nFailed files:")
        for e in errors:
            print(f"  {e}")
