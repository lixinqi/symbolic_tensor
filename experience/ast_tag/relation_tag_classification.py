"""
Lexical vs Dynamic relation_tag classification.

Lexical (compile-time): static containment structure in source text.
Dynamic (run-time): runtime execution, data flow, invocation.

Classification derived from 55,019 records across 62 JSONL files.
"""

LEXICAL_RELATION_TAGS: frozenset[str] = frozenset({
    # containment — the AST tree backbone
    "contains",
    "stmt_seq",
    "expr_stmt",
    # definition structure
    "defines",
    "param",
    "star_param",
    "double_star_param",
    "param_annotation",
    "param_default",
    "annotation",
    "decorates",
    "bases",
    "metaclass",
    "static_method",
    "class_method",
    "property_def",
    "async_def",
    # block bodies
    "if_body",
    "else_body",
    "for_body",
    "while_body",
    "with_body",
    "with_context",
    "try_body",
    "try_start",
    "except_body",
    "finally_body",
    "try_else",
    "lambda_body",
    "comprehension_body",
    "if_expr_body",
    "ellipsis_body",
    "async_for",
    "async_with",
    # string content
    "docstring",
    "text_content",
    # keyword statements
    "pass_stmt",
    "break_stmt",
    "continue_stmt",
    "bare_return",
    # assert/delete
    "assert_test",
    "assert_msg",
    "del_target",
    # scope declarations
    "global_decl",
    "nonlocal_decl",
    # exception structure
    "handles",
    "except_as",
    "raises",
    "raises_from",
})

DYNAMIC_RELATION_TAGS: frozenset[str] = frozenset({
    # calls and arguments — runtime invocation
    "calls",
    "call_pos_arg",
    "keyword_arg",
    "double_star_arg",
    # assignment — runtime binding
    "assigns",
    "aug_assigns",
    "aug_op",
    "walrus",
    # return/yield — runtime value flow
    "returns",
    "yields",
    "yields_from",
    "await_value",
    # import — runtime module loading
    "imports",
    "aliases",
    # attribute/subscript — runtime lookup
    "attr_value",
    "attr_name",
    "subscript",
    "subscript_value",
    # operators — runtime evaluation
    "bin_op",
    "bin_op_left",
    "bin_op_right",
    "unary_op",
    "unary_op_operand",
    "compare_left",
    "compare_op",
    "compare_right",
    "bool_op",
    "bool_op_operand",
    # control flow expressions — runtime condition evaluation
    "if_test",
    "while_test",
    "for_target",
    "for_iter",
    "if_expr_test",
    "if_expr_else",
    # comprehension expressions
    "comprehension_iter",
    "comprehension_if",
    "comprehension_target",
    # collection literals — runtime construction
    "dict_literal",
    "dict_key",
    "dict_value",
    "set_literal",
    "list_literal",
    "tuple_literal",
    "collection_element",
    "fstring_elem",
    # slice — runtime access
    "slice_lower",
    "slice_upper",
    "slice_step",
    "slice_marker",
    # with binding — runtime context manager result
    "with_as",
    # starred — runtime unpacking
    "starred_value",
})


if __name__ == "__main__":
    print(f"LEXICAL_RELATION_TAGS: {len(LEXICAL_RELATION_TAGS)} tags")
    print(f"DYNAMIC_RELATION_TAGS: {len(DYNAMIC_RELATION_TAGS)} tags")
    overlap = LEXICAL_RELATION_TAGS & DYNAMIC_RELATION_TAGS
    assert len(overlap) == 0, f"overlap: {overlap}"
    print(f"Total: {len(LEXICAL_RELATION_TAGS) + len(DYNAMIC_RELATION_TAGS)} tags, no overlap")
    print("relation_tag_classification: OK")
