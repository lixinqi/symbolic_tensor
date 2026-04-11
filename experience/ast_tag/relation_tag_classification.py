"""
Lexical vs Dynamic relation_tag classification.

Lexical (compile-time): static containment structure in source text.
Dynamic (run-time): runtime execution, data flow, invocation.

relation_tag format: Type__field_name (aligned with Python ast module)
"""

LEXICAL_RELATION_TAGS: frozenset[str] = frozenset({
    # block bodies — the AST tree backbone
    "Module__body",
    "FunctionDef__body",
    "AsyncFunctionDef__body",
    "ClassDef__body",
    "If__body", "If__orelse",
    "For__body", "For__orelse",
    "AsyncFor__body", "AsyncFor__orelse",
    "While__body", "While__orelse",
    "With__body", "AsyncWith__body",
    "Try__body", "Try__orelse", "Try__finalbody",
    "TryStar__body", "TryStar__orelse", "TryStar__finalbody",
    "ExceptHandler__body",
    "match_case__body",
    # definition structure
    "FunctionDef__name", "AsyncFunctionDef__name", "ClassDef__name",
    "FunctionDef__decorator_list", "AsyncFunctionDef__decorator_list", "ClassDef__decorator_list",
    "ClassDef__bases", "ClassDef__keywords",
    "FunctionDef__returns", "AsyncFunctionDef__returns",
    # parameters
    "FunctionDef__args", "AsyncFunctionDef__args", "Lambda__args",
    "arguments__posonlyargs", "arguments__args", "arguments__kwonlyargs",
    "arguments__vararg", "arguments__kwarg",
    "arguments__kw_defaults", "arguments__defaults",
    "arg__arg", "arg__annotation",
    "keyword__arg",
    # import structure
    "Import__names", "ImportFrom__names",
    "ImportFrom__module", "ImportFrom__level",
    "alias__name", "alias__asname",
    "Global__names", "Nonlocal__names",
    # exception structure
    "Try__handlers", "TryStar__handlers",
    "ExceptHandler__type", "ExceptHandler__name",
    # with items
    "With__items", "AsyncWith__items",
    "withitem__context_expr", "withitem__optional_vars",
    # comprehension structure
    "ListComp__generators", "SetComp__generators",
    "GeneratorExp__generators", "DictComp__generators",
    "comprehension__ifs", "comprehension__is_async",
    # match structure
    "Match__cases",
    "match_case__pattern", "match_case__guard",
    "MatchSequence__patterns", "MatchMapping__keys", "MatchMapping__patterns",
    "MatchMapping__rest", "MatchClass__cls", "MatchClass__patterns",
    "MatchClass__kwd_attrs", "MatchClass__kwd_patterns",
    "MatchStar__name", "MatchAs__pattern", "MatchAs__name",
    "MatchOr__patterns",
    # annotation
    "AnnAssign__annotation", "AnnAssign__simple",
})

DYNAMIC_RELATION_TAGS: frozenset[str] = frozenset({
    # call — runtime invocation
    "Call__func", "Call__args", "Call__keywords",
    "keyword__value",
    # assignment — runtime binding
    "Assign__targets", "Assign__value",
    "AugAssign__target", "AugAssign__op", "AugAssign__value",
    "AnnAssign__target", "AnnAssign__value",
    "Delete__targets",
    "NamedExpr__target", "NamedExpr__value",
    # return/yield/await
    "Return__value",
    "Yield__value", "YieldFrom__value", "Await__value",
    # expression wrapper
    "Expr__value",
    # operators — runtime evaluation
    "BinOp__left", "BinOp__op", "BinOp__right",
    "UnaryOp__op", "UnaryOp__operand",
    "BoolOp__op", "BoolOp__values",
    "Compare__left", "Compare__ops", "Compare__comparators",
    # control flow expressions
    "If__test", "While__test",
    "For__target", "For__iter",
    "AsyncFor__target", "AsyncFor__iter",
    "IfExp__test", "IfExp__body", "IfExp__orelse",
    # comprehension expressions
    "ListComp__elt", "SetComp__elt", "GeneratorExp__elt",
    "DictComp__key", "DictComp__value",
    "comprehension__target", "comprehension__iter",
    # attribute/subscript — runtime lookup
    "Attribute__value", "Attribute__attr",
    "Subscript__value", "Subscript__slice",
    "Starred__value",
    # collection literals
    "Dict__keys", "Dict__values",
    "Set__elts", "List__elts", "Tuple__elts",
    # string
    "JoinedStr__values",
    "FormattedValue__value", "FormattedValue__conversion", "FormattedValue__format_spec",
    # exception
    "Raise__exc", "Raise__cause",
    "Assert__test", "Assert__msg",
    # lambda
    "Lambda__body",
    # slice
    "Slice__lower", "Slice__upper", "Slice__step",
    # match values
    "Match__subject",
    "MatchValue__value", "MatchSingleton__value",
})


if __name__ == "__main__":
    print(f"LEXICAL_RELATION_TAGS: {len(LEXICAL_RELATION_TAGS)} tags")
    print(f"DYNAMIC_RELATION_TAGS: {len(DYNAMIC_RELATION_TAGS)} tags")
    overlap = LEXICAL_RELATION_TAGS & DYNAMIC_RELATION_TAGS
    assert len(overlap) == 0, f"overlap: {overlap}"
    print(f"Total: {len(LEXICAL_RELATION_TAGS) + len(DYNAMIC_RELATION_TAGS)} tags, no overlap")
    print("relation_tag_classification: OK")
