"""
test_lexical_scope_expand_children: 100 concrete test cases.
Each case: (file_id, owner_tag) -> expected children with exact member_tags, relation_tags, order.
Round-trip: go_to_parent on each $-prefixed child returns the original owner.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ast_tag_db import load_jsonl_dataset_into_ast_tag_db, AstTagDB
from relation_tag_classification import LEXICAL_RELATION_TAGS
from tag_actions.lexical_scope_expand_children import lexical_scope_expand_children
from tag_actions.lexical_scope_go_to_parent import lexical_scope_go_to_parent


def _db() -> AstTagDB:
    if not hasattr(_db, "_instance"):
        dataset_dir = os.path.join(os.path.dirname(__file__), "..", "test_dataset")
        _db._instance = load_jsonl_dataset_into_ast_tag_db(dataset_dir)
    return _db._instance


class TestLexicalScopeExpandChildren(unittest.TestCase):

    def test_000_symbolic_tensor_tensor_util_make_none_tensor__expr_7(self):
        """expand_children('symbolic_tensor/tensor_util/make_none_tensor.jsonl', '$expr_7'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', '$expr_7')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_12'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_7')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_7')

    def test_001_test_test_transform_method_time_comparison__with_1(self):
        """expand_children('test/test_transform_method_time_comparison.jsonl', '$with_1'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'test/test_transform_method_time_comparison.jsonl', '$with_1')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$assign_10', '$call_13'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['with_body', 'with_context'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$with_1')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'test/test_transform_method_time_comparison.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$with_1')

    def test_002_symbolic_tensor_function_slice_view__expr_45(self):
        """expand_children('symbolic_tensor/function/slice_view.jsonl', '$expr_45'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_view.jsonl', '$expr_45')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_134'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_45')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_view.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_45')

    def test_003_symbolic_tensor_tensor_util_pack_tensor__expr_0(self):
        """expand_children('symbolic_tensor/tensor_util/pack_tensor.jsonl', '$expr_0'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/pack_tensor.jsonl', '$expr_0')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$docstring_1'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['docstring'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/pack_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_0')

    def test_004_symbolic_tensor_tensor_util_assign_tensor__with_6(self):
        """expand_children('symbolic_tensor/tensor_util/assign_tensor.jsonl', '$with_6'): 6 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/assign_tensor.jsonl', '$with_6')
        self.assertEqual(len(children), 6)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$assign_21', '$call_84', '$assign_22', '$expr_32', '$expr_33', '$expr_34'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['with_body', 'with_context', 'with_body', 'with_body', 'with_body', 'with_body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$with_6')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/assign_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$with_6')

    def test_005_symbolic_tensor_tensor_util_st_patched__expr_12(self):
        """expand_children('symbolic_tensor/tensor_util/st_patched.jsonl', '$expr_12'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/st_patched.jsonl', '$expr_12')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_33'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_12')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/st_patched.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_12')

    def test_006_symbolic_tensor_function_merge_forward__expr_52(self):
        """expand_children('symbolic_tensor/function/merge_forward.jsonl', '$expr_52'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/merge_forward.jsonl', '$expr_52')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_144'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_52')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/merge_forward.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_52')

    def test_007_symbolic_tensor_tensor_util_make_tensor__expr_17(self):
        """expand_children('symbolic_tensor/tensor_util/make_tensor.jsonl', '$expr_17'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/make_tensor.jsonl', '$expr_17')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_49'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_17')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/make_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_17')

    def test_008_llm_client_coding_agent_task_handler__if_1(self):
        """expand_children('llm_client/coding_agent_task_handler.jsonl', '$if_1'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'llm_client/coding_agent_task_handler.jsonl', '$if_1')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$assign_5'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['if_body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$if_1')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'llm_client/coding_agent_task_handler.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$if_1')

    def test_009_symbolic_tensor_tensor_util_empty_tensor_like__if_1(self):
        """expand_children('symbolic_tensor/tensor_util/empty_tensor_like.jsonl', '$if_1'): 12 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', '$if_1')
        self.assertEqual(len(children), 12)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$import_1', '$import_2', '$importfrom_2', '$expr_2', '$functiondef_2', '$expr_7', '$with_0', '$expr_11', '$with_2', '$expr_14', '$with_4', '$expr_17'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['if_body', 'if_body', 'if_body', 'if_body', 'if_body', 'if_body', 'if_body', 'if_body', 'if_body', 'if_body', 'if_body', 'if_body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$if_1')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$if_1')

    def test_010_fs_util_text_merger__expr_0(self):
        """expand_children('fs_util/text_merger.jsonl', '$expr_0'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'fs_util/text_merger.jsonl', '$expr_0')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$docstring_0'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['docstring'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'fs_util/text_merger.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_0')

    def test_011_sparse_util_convert_nested_list_coordinates_to_pairs_coordinates__docstring_2(self):
        """expand_children('sparse_util/convert_nested_list_coordinates_to_pairs_coordinates.jsonl', '$docstring_2'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'sparse_util/convert_nested_list_coordinates_to_pairs_coordinates.jsonl', '$docstring_2')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['Recursively collect (coordinate_key, leaf_value) pairs.'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['text_content'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$docstring_2')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'sparse_util/convert_nested_list_coordinates_to_pairs_coordinates.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$docstring_2')

    def test_012_symbolic_tensor_data_loader_sole_file_batch_data_loader__expr_23(self):
        """expand_children('symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', '$expr_23'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', '$expr_23')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_57'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_23')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_23')

    def test_013_symbolic_tensor_tensor_util_load_tensor__for_1(self):
        """expand_children('symbolic_tensor/tensor_util/load_tensor.jsonl', '$for_1'): 3 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/load_tensor.jsonl', '$for_1')
        self.assertEqual(len(children), 3)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$assign_6', '$assign_7', '$expr_5'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['for_body', 'for_body', 'for_body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$for_1')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/load_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$for_1')

    def test_014_symbolic_tensor_function_st_stack__if_8(self):
        """expand_children('symbolic_tensor/function/st_stack.jsonl', '$if_8'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_stack.jsonl', '$if_8')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$assign_48'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['if_body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$if_8')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_stack.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$if_8')

    def test_015_symbolic_tensor_tensor_util_pack_tensor__with_2(self):
        """expand_children('symbolic_tensor/tensor_util/pack_tensor.jsonl', '$with_2'): 5 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/pack_tensor.jsonl', '$with_2')
        self.assertEqual(len(children), 5)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$assign_5', '$call_22', '$assign_6', '$expr_15', '$expr_16'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['with_body', 'with_context', 'with_body', 'with_body', 'with_body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$with_2')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/pack_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$with_2')

    def test_016_symbolic_tensor_tensor_util_register_tensor_ops__functiondef_10(self):
        """expand_children('symbolic_tensor/tensor_util/register_tensor_ops.jsonl', '$functiondef_10'): 7 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl', '$functiondef_10')
        self.assertEqual(len(children), 7)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$importfrom_8', '_st_value_slicer', 'self', 'property', '0:$importfrom_8', '$return_9', '1:$return_9'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['contains', 'defines', 'param', 'property_def', 'stmt_seq', 'contains', 'stmt_seq'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$functiondef_10')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$functiondef_10')

    def test_017_llm_client_coding_agent_query__if_0(self):
        """expand_children('llm_client/coding_agent_query.jsonl', '$if_0'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'llm_client/coding_agent_query.jsonl', '$if_0')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$assign_0'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['if_body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$if_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'llm_client/coding_agent_query.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$if_0')

    def test_018_symbolic_tensor_function_symbolic_grad_registry__docstring_1(self):
        """expand_children('symbolic_tensor/function/symbolic_grad_registry.jsonl', '$docstring_1'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/symbolic_grad_registry.jsonl', '$docstring_1')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['Associate a symbolic gradient with a key (string uid or int id).'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['text_content'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$docstring_1')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/symbolic_grad_registry.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$docstring_1')

    def test_019_symbolic_tensor_function_slice_attention__expr_43(self):
        """expand_children('symbolic_tensor/function/slice_attention.jsonl', '$expr_43'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_attention.jsonl', '$expr_43')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_120'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_43')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_attention.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_43')

    def test_020_symbolic_tensor_tensor_util_dense_to_sparse__with_1(self):
        """expand_children('symbolic_tensor/tensor_util/dense_to_sparse.jsonl', '$with_1'): 8 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', '$with_1')
        self.assertEqual(len(children), 8)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$assign_11', '$call_28', '$assign_12', '$expr_9', '$expr_10', '$expr_11', '$expr_12', '$expr_13'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['with_body', 'with_context', 'with_body', 'with_body', 'with_body', 'with_body', 'with_body', 'with_body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$with_1')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$with_1')

    def test_021_symbolic_tensor_tensor_util_sparse_to_dense__expr_44(self):
        """expand_children('symbolic_tensor/tensor_util/sparse_to_dense.jsonl', '$expr_44'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', '$expr_44')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_139'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_44')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_44')

    def test_022_symbolic_tensor_data_loader_sole_file_batch_data_loader__expr_43(self):
        """expand_children('symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', '$expr_43'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', '$expr_43')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_113'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_43')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_43')

    def test_023_symbolic_tensor_tensor_util_patch_tensor__expr_14(self):
        """expand_children('symbolic_tensor/tensor_util/patch_tensor.jsonl', '$expr_14'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/patch_tensor.jsonl', '$expr_14')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_61'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_14')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/patch_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_14')

    def test_024_symbolic_tensor_function_fork_tensor__expr_46(self):
        """expand_children('symbolic_tensor/function/fork_tensor.jsonl', '$expr_46'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/fork_tensor.jsonl', '$expr_46')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_134'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_46')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/fork_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_46')

    def test_025_llm_client_task_handler__if_1(self):
        """expand_children('llm_client/task_handler.jsonl', '$if_1'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'llm_client/task_handler.jsonl', '$if_1')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$raise_0', '$expr_1'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['else_body', 'if_body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$if_1')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'llm_client/task_handler.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$if_1')

    def test_026_symbolic_tensor_function_merge_forward__with_9(self):
        """expand_children('symbolic_tensor/function/merge_forward.jsonl', '$with_9'): 8 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/merge_forward.jsonl', '$with_9')
        self.assertEqual(len(children), 8)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$assign_55', '$call_122', '$assign_56', '$expr_46', '$assign_57', '$assign_58', '$expr_47', '$expr_48'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['with_body', 'with_context', 'with_body', 'with_body', 'with_body', 'with_body', 'with_body', 'with_body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$with_9')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/merge_forward.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$with_9')

    def test_027_fs_util_pack_dir__try_0(self):
        """expand_children('fs_util/pack_dir.jsonl', '$try_0'): 4 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'fs_util/pack_dir.jsonl', '$try_0')
        self.assertEqual(len(children), 4)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$continue_0', '$tuple_1', '$with_0', 'try'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['except_body', 'handles', 'try_body', 'try_start'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$try_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'fs_util/pack_dir.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$try_0')

    def test_028_symbolic_tensor_tensor_util_pack_tensor__expr_6(self):
        """expand_children('symbolic_tensor/tensor_util/pack_tensor.jsonl', '$expr_6'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/pack_tensor.jsonl', '$expr_6')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_7'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_6')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/pack_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_6')

    def test_029_symbolic_tensor_tensor_util_register_tensor_ops__expr_2(self):
        """expand_children('symbolic_tensor/tensor_util/register_tensor_ops.jsonl', '$expr_2'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl', '$expr_2')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_12'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_2')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_2')

    def test_030_symbolic_tensor_tensor_util_empty_tensor_like__module(self):
        """expand_children('symbolic_tensor/tensor_util/empty_tensor_like.jsonl', '<module>'): 7 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', '<module>')
        self.assertEqual(len(children), 7)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$import_0', '$importfrom_0', '$importfrom_1', '$assign_0', '$functiondef_0', '$functiondef_1', '$if_1'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['contains', 'contains', 'contains', 'contains', 'contains', 'contains', 'contains'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '<module>')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '<module>')

    def test_031_symbolic_tensor_tensor_util_empty_tensor_like__functiondef_0(self):
        """expand_children('symbolic_tensor/tensor_util/empty_tensor_like.jsonl', '$functiondef_0'): 14 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', '$functiondef_0')
        self.assertEqual(len(children), 14)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$expr_0', '_expand', '$docstring_0', 'size', 'size', '0:$expr_0', '$if_0', 'fill', 'torch.Size', '1:$if_0', '$return_1', 'fill', '2:$return_1', 'str'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['contains', 'defines', 'docstring', 'param', 'param_annotation', 'stmt_seq', 'contains', 'param', 'param_annotation', 'stmt_seq', 'contains', 'param_annotation', 'stmt_seq', 'param_annotation'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$functiondef_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$functiondef_0')

    def test_032_symbolic_tensor_tensor_util_pack_tensor__expr_11(self):
        """expand_children('symbolic_tensor/tensor_util/pack_tensor.jsonl', '$expr_11'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/pack_tensor.jsonl', '$expr_11')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_18'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_11')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/pack_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_11')

    def test_033_symbolic_tensor_function_slice_attention_forward__expr_58(self):
        """expand_children('symbolic_tensor/function/slice_attention_forward.jsonl', '$expr_58'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_attention_forward.jsonl', '$expr_58')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_171'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_58')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_attention_forward.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_58')

    def test_034_symbolic_tensor_tensor_util_load_tensor__with_6(self):
        """expand_children('symbolic_tensor/tensor_util/load_tensor.jsonl', '$with_6'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/load_tensor.jsonl', '$with_6')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$expr_25', '$call_90'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['with_body', 'with_context'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$with_6')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/load_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$with_6')

    def test_035_symbolic_tensor_module_with_dense_view__if_2(self):
        """expand_children('symbolic_tensor/module/with_dense_view.jsonl', '$if_2'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/module/with_dense_view.jsonl', '$if_2')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$expr_5', '$expr_6'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['if_body', 'if_body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$if_2')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/module/with_dense_view.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$if_2')

    def test_036_symbolic_tensor_tensor_util_assign_view__docstring_1(self):
        """expand_children('symbolic_tensor/tensor_util/assign_view.jsonl', '$docstring_1'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/assign_view.jsonl', '$docstring_1')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['Get storage file path WITHOUT resolving symlinks.'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['text_content'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$docstring_1')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/assign_view.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$docstring_1')

    def test_037_symbolic_tensor_function_st_copy__expr_2(self):
        """expand_children('symbolic_tensor/function/st_copy.jsonl', '$expr_2'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_copy.jsonl', '$expr_2')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_3'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_2')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_copy.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_2')

    def test_038_symbolic_tensor_function_st_attention__with_0(self):
        """expand_children('symbolic_tensor/function/st_attention.jsonl', '$with_0'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_attention.jsonl', '$with_0')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$return_2', '$call_13'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['with_body', 'with_context'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$with_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_attention.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$with_0')

    def test_039_llm_client_raw_llm_task_handler__docstring_3(self):
        """expand_children('llm_client/raw_llm_task_handler.jsonl', '$docstring_3'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'llm_client/raw_llm_task_handler.jsonl', '$docstring_3')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['Find all files under root_dir whose content contains the hint string.'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['text_content'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$docstring_3')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'llm_client/raw_llm_task_handler.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$docstring_3')

    def test_040_symbolic_tensor_function_get_causal_attention_mask__functiondef_1(self):
        """expand_children('symbolic_tensor/function/get_causal_attention_mask.jsonl', '$functiondef_1'): 15 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/get_causal_attention_mask.jsonl', '$functiondef_1')
        self.assertEqual(len(children), 15)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$if_1', 'run_test', 'name', 'name', 'expected', '0:$if_1', 'condition', 'str', 'None', 'expected', 'condition', 'actual', 'actual', 'bool', 'None'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['contains', 'defines', 'param', 'param_annotation', 'param_default', 'stmt_seq', 'param', 'param_annotation', 'param_default', 'param', 'param_annotation', 'param_default', 'param', 'param_annotation', 'param_default'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$functiondef_1')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/get_causal_attention_mask.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$functiondef_1')

    def test_041_test_test_st_attention_followed_by_st_moe__expr_4(self):
        """expand_children('test/test_st_attention_followed_by_st_moe.jsonl', '$expr_4'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'test/test_st_attention_followed_by_st_moe.jsonl', '$expr_4')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$docstring_1'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['docstring'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_4')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'test/test_st_attention_followed_by_st_moe.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_4')

    def test_042_symbolic_tensor_function_get_query_tensor__if_3(self):
        """expand_children('symbolic_tensor/function/get_query_tensor.jsonl', '$if_3'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/get_query_tensor.jsonl', '$if_3')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$assign_16', '$assign_17'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['if_body', 'if_body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$if_3')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/get_query_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$if_3')

    def test_043_symbolic_tensor_tensor_util_dense_to_sparse__if_3(self):
        """expand_children('symbolic_tensor/tensor_util/dense_to_sparse.jsonl', '$if_3'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', '$if_3')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$expr_6', '$expr_7'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['if_body', 'if_body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$if_3')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$if_3')

    def test_044_symbolic_tensor_tensor_util_st_patched__expr_9(self):
        """expand_children('symbolic_tensor/tensor_util/st_patched.jsonl', '$expr_9'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/st_patched.jsonl', '$expr_9')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_26'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_9')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/st_patched.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_9')

    def test_045_symbolic_tensor_tensor_util_none_tensor_like__expr_10(self):
        """expand_children('symbolic_tensor/tensor_util/none_tensor_like.jsonl', '$expr_10'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', '$expr_10')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_17'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_10')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_10')

    def test_046_symbolic_tensor_function_symbolic_grad_registry__expr_3(self):
        """expand_children('symbolic_tensor/function/symbolic_grad_registry.jsonl', '$expr_3'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/symbolic_grad_registry.jsonl', '$expr_3')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$docstring_6'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['docstring'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_3')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/symbolic_grad_registry.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_3')

    def test_047_symbolic_tensor_tensor_util_empty_tensor_like__expr_12(self):
        """expand_children('symbolic_tensor/tensor_util/empty_tensor_like.jsonl', '$expr_12'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', '$expr_12')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_31'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_12')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_12')

    def test_048_symbolic_tensor_function_with_dense_view__expr_24(self):
        """expand_children('symbolic_tensor/function/with_dense_view.jsonl', '$expr_24'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/with_dense_view.jsonl', '$expr_24')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_74'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_24')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/with_dense_view.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_24')

    def test_049_symbolic_tensor_tensor_util_todo_tensor_like__expr_22(self):
        """expand_children('symbolic_tensor/tensor_util/todo_tensor_like.jsonl', '$expr_22'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl', '$expr_22')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_69'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_22')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_22')

    def test_050_symbolic_tensor_function_get_query_tensor__if_1(self):
        """expand_children('symbolic_tensor/function/get_query_tensor.jsonl', '$if_1'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/get_query_tensor.jsonl', '$if_1')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$with_0', '$with_1'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['if_body', 'if_body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$if_1')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/get_query_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$if_1')

    def test_051_symbolic_tensor_function_merge_backward__for_1(self):
        """expand_children('symbolic_tensor/function/merge_backward.jsonl', '$for_1'): 7 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/merge_backward.jsonl', '$for_1')
        self.assertEqual(len(children), 7)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$assign_23', '$assign_24', '$assign_25', '$assign_26', '$assign_27', '$assign_28', '$expr_4'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['for_body', 'for_body', 'for_body', 'for_body', 'for_body', 'for_body', 'for_body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$for_1')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/merge_backward.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$for_1')

    def test_052_symbolic_tensor_function_st_stack__with_10(self):
        """expand_children('symbolic_tensor/function/st_stack.jsonl', '$with_10'): 4 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_stack.jsonl', '$with_10')
        self.assertEqual(len(children), 4)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$assign_87', '$call_216', '$assign_88', '$try_0'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['with_body', 'with_context', 'with_body', 'with_body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$with_10')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_stack.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$with_10')

    def test_053_symbolic_tensor_tensor_util_patch_tensor__expr_2(self):
        """expand_children('symbolic_tensor/tensor_util/patch_tensor.jsonl', '$expr_2'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/patch_tensor.jsonl', '$expr_2')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_27'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_2')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/patch_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_2')

    def test_054_fs_util_get_nested_list_file_pathes__comp_1(self):
        """expand_children('fs_util/get_nested_list_file_pathes.jsonl', '$comp_1'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'fs_util/get_nested_list_file_pathes.jsonl', '$comp_1')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_7'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['comprehension_body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$comp_1')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'fs_util/get_nested_list_file_pathes.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$comp_1')

    def test_055_symbolic_tensor_tensor_util_load_tensor__docstring_0(self):
        """expand_children('symbolic_tensor/tensor_util/load_tensor.jsonl', '$docstring_0'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/load_tensor.jsonl', '$docstring_0')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['Convert a string representation of an integer into a list of digit strings.'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['text_content'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$docstring_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/load_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$docstring_0')

    def test_056_fs_util_get_nested_list_file_pathes__docstring_3(self):
        """expand_children('fs_util/get_nested_list_file_pathes.jsonl', '$docstring_3'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'fs_util/get_nested_list_file_pathes.jsonl', '$docstring_3')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['Reshape a flat list of paths into a nested list matching the tensor shape.'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['text_content'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$docstring_3')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'fs_util/get_nested_list_file_pathes.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$docstring_3')

    def test_057_symbolic_tensor_function_merge__expr_25(self):
        """expand_children('symbolic_tensor/function/merge.jsonl', '$expr_25'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/merge.jsonl', '$expr_25')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_61'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_25')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/merge.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_25')

    def test_058_symbolic_tensor_tensor_util_dump_tensor__expr_18(self):
        """expand_children('symbolic_tensor/tensor_util/dump_tensor.jsonl', '$expr_18'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/dump_tensor.jsonl', '$expr_18')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_62'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_18')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/dump_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_18')

    def test_059_symbolic_tensor_function_select_qkv_indexes__expr_30(self):
        """expand_children('symbolic_tensor/function/select_qkv_indexes.jsonl', '$expr_30'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/select_qkv_indexes.jsonl', '$expr_30')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_99'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_30')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/select_qkv_indexes.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_30')

    def test_060_llm_client_raw_llm_task_handler__continue_0(self):
        """expand_children('llm_client/raw_llm_task_handler.jsonl', '$continue_0'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'llm_client/raw_llm_task_handler.jsonl', '$continue_0')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['continue'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['continue_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$continue_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'llm_client/raw_llm_task_handler.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$continue_0')

    def test_061_test_test_gain_st_sgd__with_0(self):
        """expand_children('test/test_gain_st_sgd.jsonl', '$with_0'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'test/test_gain_st_sgd.jsonl', '$with_0')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$return_0', '$call_4'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['with_body', 'with_context'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$with_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'test/test_gain_st_sgd.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$with_0')

    def test_062_symbolic_tensor_function_slice_attention_backward__expr_28(self):
        """expand_children('symbolic_tensor/function/slice_attention_backward.jsonl', '$expr_28'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_attention_backward.jsonl', '$expr_28')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_105'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_28')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_attention_backward.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_28')

    def test_063_symbolic_tensor_function_get_query_tensor__expr_12(self):
        """expand_children('symbolic_tensor/function/get_query_tensor.jsonl', '$expr_12'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/get_query_tensor.jsonl', '$expr_12')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_60'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_12')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/get_query_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_12')

    def test_064_symbolic_tensor_function_merge_backward__docstring_0(self):
        """expand_children('symbolic_tensor/function/merge_backward.jsonl', '$docstring_0'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/merge_backward.jsonl', '$docstring_0')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['Get storage file path for a flat index.'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['text_content'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$docstring_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/merge_backward.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$docstring_0')

    def test_065_llm_client_agent_task__module(self):
        """expand_children('llm_client/agent_task.jsonl', '<module>'): 3 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'llm_client/agent_task.jsonl', '<module>')
        self.assertEqual(len(children), 3)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$importfrom_0', '$importfrom_1', '$classdef_0'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['contains', 'contains', 'contains'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '<module>')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'llm_client/agent_task.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '<module>')

    def test_066_llm_client_coding_agent_query__for_0(self):
        """expand_children('llm_client/coding_agent_query.jsonl', '$for_0'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'llm_client/coding_agent_query.jsonl', '$for_0')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$assign_2', '$assign_3'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['for_body', 'for_body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$for_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'llm_client/coding_agent_query.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$for_0')

    def test_067_symbolic_tensor_function_slice_view__if_4(self):
        """expand_children('symbolic_tensor/function/slice_view.jsonl', '$if_4'): 3 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_view.jsonl', '$if_4')
        self.assertEqual(len(children), 3)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$expr_7', '$expr_6', '$augassign_0'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['else_body', 'if_body', 'else_body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$if_4')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_view.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$if_4')

    def test_068_symbolic_tensor_function_slice_attention_forward__expr_0(self):
        """expand_children('symbolic_tensor/function/slice_attention_forward.jsonl', '$expr_0'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_attention_forward.jsonl', '$expr_0')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$docstring_1'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['docstring'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_attention_forward.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_0')

    def test_069_symbolic_tensor_tensor_util_make_none_tensor__docstring_0(self):
        """expand_children('symbolic_tensor/tensor_util/make_none_tensor.jsonl', '$docstring_0'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', '$docstring_0')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['\n    Create a zero-filled symbolic tensor with metadata attributes.\n    Args:\n        shape: The shape of the tensor to create.\n        relative_to: The root directory path to associate with this tensor.\n        dtype: The tensor dtype (default: torch.bfloat16).\n    Returns:\n        A zero-filled torch.Tensor with st_relative_to and st_tensor_uid attributes.\n    '])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['text_content'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$docstring_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$docstring_0')

    def test_070_llm_client_agent_task__classdef_0(self):
        """expand_children('llm_client/agent_task.jsonl', '$classdef_0'): 9 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'llm_client/agent_task.jsonl', '$classdef_0')
        self.assertEqual(len(children), 9)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$annassign_0', 'AgentTask', '0:$annassign_0', '$annassign_1', '1:$annassign_1', '$annassign_2', '2:$annassign_2', '$annassign_3', '3:$annassign_3'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['contains', 'defines', 'stmt_seq', 'contains', 'stmt_seq', 'contains', 'stmt_seq', 'contains', 'stmt_seq'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$classdef_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'llm_client/agent_task.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$classdef_0')

    def test_071_symbolic_tensor_tensor_util_make_tensor__assert_0(self):
        """expand_children('symbolic_tensor/tensor_util/make_tensor.jsonl', '$assert_0'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/make_tensor.jsonl', '$assert_0')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$fstring_0', '$call_5'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['assert_msg', 'assert_test'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$assert_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/make_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$assert_0')

    def test_072_symbolic_tensor_function_coding_agent__expr_36(self):
        """expand_children('symbolic_tensor/function/coding_agent.jsonl', '$expr_36'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/coding_agent.jsonl', '$expr_36')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_109'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_36')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/coding_agent.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_36')

    def test_073_symbolic_tensor_function_st_moe_forward__docstring_3(self):
        """expand_children('symbolic_tensor/function/st_moe_forward.jsonl', '$docstring_3'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe_forward.jsonl', '$docstring_3')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['Replace the last index tensor with a full slice to keep all q/k/v.'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['text_content'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$docstring_3')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe_forward.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$docstring_3')

    def test_074_symbolic_tensor_tensor_util_load_tensor__functiondef_0(self):
        """expand_children('symbolic_tensor/tensor_util/load_tensor.jsonl', '$functiondef_0'): 9 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/load_tensor.jsonl', '$functiondef_0')
        self.assertEqual(len(children), 9)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$expr_0', '_str_to_digit_list', '$docstring_0', 's', 's', '0:$expr_0', '$return_0', 'str', '1:$return_0'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['contains', 'defines', 'docstring', 'param', 'param_annotation', 'stmt_seq', 'contains', 'param_annotation', 'stmt_seq'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$functiondef_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/load_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$functiondef_0')

    def test_075_fs_util_pack_dir__functiondef_0(self):
        """expand_children('fs_util/pack_dir.jsonl', '$functiondef_0'): 14 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'fs_util/pack_dir.jsonl', '$functiondef_0')
        self.assertEqual(len(children), 14)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$assign_0', 'pack_dir', 'root_dir', 'root_dir', '0:$assign_0', '$assign_1', 'str', '1:$assign_1', '$for_0', '2:$for_0', '$assign_6', '3:$assign_6', '$return_0', '4:$return_0'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['contains', 'defines', 'param', 'param_annotation', 'stmt_seq', 'contains', 'param_annotation', 'stmt_seq', 'contains', 'stmt_seq', 'contains', 'stmt_seq', 'contains', 'stmt_seq'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$functiondef_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'fs_util/pack_dir.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$functiondef_0')

    def test_076_symbolic_tensor_function_st_attention__expr_4(self):
        """expand_children('symbolic_tensor/function/st_attention.jsonl', '$expr_4'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_attention.jsonl', '$expr_4')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_5'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_4')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_attention.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_4')

    def test_077_symbolic_tensor_tensor_util_patch_tensor__expr_1(self):
        """expand_children('symbolic_tensor/tensor_util/patch_tensor.jsonl', '$expr_1'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/patch_tensor.jsonl', '$expr_1')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$docstring_3'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['docstring'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_1')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/patch_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_1')

    def test_078_symbolic_tensor_tensor_util_patch_tensor__expr_28(self):
        """expand_children('symbolic_tensor/tensor_util/patch_tensor.jsonl', '$expr_28'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/patch_tensor.jsonl', '$expr_28')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_99'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_28')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/patch_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_28')

    def test_079_symbolic_tensor_tensor_util_st_patched__expr_9(self):
        """expand_children('symbolic_tensor/tensor_util/st_patched.jsonl', '$expr_9'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/st_patched.jsonl', '$expr_9')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_26'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_9')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/st_patched.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_9')

    def test_080_symbolic_tensor_tensor_util_assign_tensor__with_8(self):
        """expand_children('symbolic_tensor/tensor_util/assign_tensor.jsonl', '$with_8'): 8 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/assign_tensor.jsonl', '$with_8')
        self.assertEqual(len(children), 8)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$importfrom_3', '$call_104', '$assign_27', '$assign_28', '$importfrom_4', '$assign_29', '$expr_40', '$expr_41'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['with_body', 'with_context', 'with_body', 'with_body', 'with_body', 'with_body', 'with_body', 'with_body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$with_8')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/assign_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$with_8')

    def test_081_symbolic_tensor_function_merge__comp_0(self):
        """expand_children('symbolic_tensor/function/merge.jsonl', '$comp_0'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/merge.jsonl', '$comp_0')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$fstring_4'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['comprehension_body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$comp_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/merge.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$comp_0')

    def test_082_symbolic_tensor_function_with_dense_view__module(self):
        """expand_children('symbolic_tensor/function/with_dense_view.jsonl', '<module>'): 9 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/with_dense_view.jsonl', '<module>')
        self.assertEqual(len(children), 9)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$import_0', '$importfrom_0', '$importfrom_1', '$importfrom_2', '$importfrom_3', '$importfrom_4', '$classdef_0', '$functiondef_2', '$if_10'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['contains', 'contains', 'contains', 'contains', 'contains', 'contains', 'contains', 'contains', 'contains'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '<module>')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/with_dense_view.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '<module>')

    def test_083_symbolic_tensor_function_st_moe_backward__docstring_9(self):
        """expand_children('symbolic_tensor/function/st_moe_backward.jsonl', '$docstring_9'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe_backward.jsonl', '$docstring_9')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['Write text content to a flat index.'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['text_content'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$docstring_9')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe_backward.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$docstring_9')

    def test_084_symbolic_tensor_module_st_moe__functiondef_0(self):
        """expand_children('symbolic_tensor/module/st_moe.jsonl', '$functiondef_0'): 76 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/module/st_moe.jsonl', '$functiondef_0')
        self.assertEqual(len(children), 76)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$expr_1', '__init__', 'self', 'experience_shape', 'output_prompt', '0:$expr_1', '$assign_0', 'experience_shape', '$subscript_0', 'None', '1:$assign_0', '$assign_1', 'output_prompt', 'output_prompt', 'query_prompt', '2:$assign_1', '$assign_2', 'query_prompt', '$subscript_1', 'None', '3:$assign_2', '$assign_3', 'grad_input_prompt', 'query_prompt', 'grad_input_prompt', '4:$assign_3', '$assign_4', 'grad_exp_key_prompt', '$subscript_3', 'None', '5:$assign_4', '$assign_5', 'grad_exp_value_prompt', 'grad_input_prompt', 'grad_exp_key_prompt', '6:$assign_5', '$assign_6', 'task_prompt', '$subscript_5', 'None', '7:$assign_6', '$assign_7', 'topk', 'grad_exp_key_prompt', 'grad_exp_value_prompt', '8:$assign_7', '$assign_8', 'retrieval_method', '$subscript_7', 'None', '9:$assign_8', '$assign_9', 'llm_env', 'grad_exp_value_prompt', 'task_prompt', '10:$assign_9', '$assign_10', '$subscript_9', "''", '11:$assign_10', '$expr_2', 'task_prompt', 'topk', '12:$expr_2', 'str', '16', 'topk', 'retrieval_method', 'int', 'None', 'retrieval_method', 'llm_env', '$subscript_11', 'None', 'llm_env', '$subscript_12'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['contains', 'defines', 'param', 'param_annotation', 'param_default', 'stmt_seq', 'contains', 'param', 'param_annotation', 'param_default', 'stmt_seq', 'contains', 'param', 'param_annotation', 'param_default', 'stmt_seq', 'contains', 'param', 'param_annotation', 'param_default', 'stmt_seq', 'contains', 'param', 'param_annotation', 'param_default', 'stmt_seq', 'contains', 'param', 'param_annotation', 'param_default', 'stmt_seq', 'contains', 'param', 'param_annotation', 'param_default', 'stmt_seq', 'contains', 'param', 'param_annotation', 'param_default', 'stmt_seq', 'contains', 'param', 'param_annotation', 'param_default', 'stmt_seq', 'contains', 'param', 'param_annotation', 'param_default', 'stmt_seq', 'contains', 'param', 'param_annotation', 'param_default', 'stmt_seq', 'contains', 'param_annotation', 'param_default', 'stmt_seq', 'contains', 'param_annotation', 'param_default', 'stmt_seq', 'param_annotation', 'param_default', 'param_annotation', 'param_default', 'param_annotation', 'param_default', 'param_annotation', 'param_default', 'param_annotation', 'param_default', 'param_annotation', 'param_annotation'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$functiondef_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/module/st_moe.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$functiondef_0')

    def test_085_sparse_util_group_random_select__docstring_0(self):
        """expand_children('sparse_util/group_random_select.jsonl', '$docstring_0'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'sparse_util/group_random_select.jsonl', '$docstring_0')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['\n    Randomly select one element uniformly from each group, returning original indices.\n    group_ids: (N,) integer tensor\n    '])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['text_content'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$docstring_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'sparse_util/group_random_select.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$docstring_0')

    def test_086_symbolic_tensor_function_st_moe_backward__expr_96(self):
        """expand_children('symbolic_tensor/function/st_moe_backward.jsonl', '$expr_96'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe_backward.jsonl', '$expr_96')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_279'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_96')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe_backward.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_96')

    def test_087_symbolic_tensor_function_slice_attention_backward__expr_24(self):
        """expand_children('symbolic_tensor/function/slice_attention_backward.jsonl', '$expr_24'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_attention_backward.jsonl', '$expr_24')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_92'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_24')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_attention_backward.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_24')

    def test_088_sparse_util_transpose_pairs_coordinates__for_1(self):
        """expand_children('sparse_util/transpose_pairs_coordinates.jsonl', '$for_1'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'sparse_util/transpose_pairs_coordinates.jsonl', '$for_1')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$if_1'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['for_body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$for_1')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'sparse_util/transpose_pairs_coordinates.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$for_1')

    def test_089_symbolic_tensor_function_get_query_tensor__assert_1(self):
        """expand_children('symbolic_tensor/function/get_query_tensor.jsonl', '$assert_1'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/get_query_tensor.jsonl', '$assert_1')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$fstring_4', '$compare_3'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['assert_msg', 'assert_test'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$assert_1')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/get_query_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$assert_1')

    def test_090_symbolic_tensor_function_st_moe__expr_9(self):
        """expand_children('symbolic_tensor/function/st_moe.jsonl', '$expr_9'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe.jsonl', '$expr_9')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_20'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_9')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_9')

    def test_091_symbolic_tensor_function_st_stack__expr_45(self):
        """expand_children('symbolic_tensor/function/st_stack.jsonl', '$expr_45'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_stack.jsonl', '$expr_45')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_154'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_45')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_stack.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_45')

    def test_092_symbolic_tensor_tensor_util_slice_tensor__expr_12(self):
        """expand_children('symbolic_tensor/tensor_util/slice_tensor.jsonl', '$expr_12'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/slice_tensor.jsonl', '$expr_12')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_54'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_12')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/slice_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_12')

    def test_093_symbolic_tensor_function_st_copy__expr_9(self):
        """expand_children('symbolic_tensor/function/st_copy.jsonl', '$expr_9'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_copy.jsonl', '$expr_9')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_22'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_9')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_copy.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_9')

    def test_094_symbolic_tensor_function_slice_attention__if_7(self):
        """expand_children('symbolic_tensor/function/slice_attention.jsonl', '$if_7'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_attention.jsonl', '$if_7')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$return_2'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['if_body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$if_7')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_attention.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$if_7')

    def test_095_llm_client_coding_agent_task_handler__if_1(self):
        """expand_children('llm_client/coding_agent_task_handler.jsonl', '$if_1'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'llm_client/coding_agent_task_handler.jsonl', '$if_1')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$assign_5'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['if_body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$if_1')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'llm_client/coding_agent_task_handler.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$if_1')

    def test_096_symbolic_tensor_tensor_util_dense_to_sparse__expr_17(self):
        """expand_children('symbolic_tensor/tensor_util/dense_to_sparse.jsonl', '$expr_17'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', '$expr_17')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$call_47'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_17')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_17')

    def test_097_llm_client_raw_llm_task_handler__continue_0(self):
        """expand_children('llm_client/raw_llm_task_handler.jsonl', '$continue_0'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'llm_client/raw_llm_task_handler.jsonl', '$continue_0')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['continue'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['continue_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$continue_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'llm_client/raw_llm_task_handler.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$continue_0')

    def test_098_symbolic_tensor_module_st_moe__functiondef_2(self):
        """expand_children('symbolic_tensor/module/st_moe.jsonl', '$functiondef_2'): 7 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/module/st_moe.jsonl', '$functiondef_2')
        self.assertEqual(len(children), 7)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$return_0', 'forward', 'self', 'input', '0:$return_0', 'input', 'torch.Tensor'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['contains', 'defines', 'param', 'param_annotation', 'stmt_seq', 'param', 'param_annotation'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$functiondef_2')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/module/st_moe.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$functiondef_2')

    def test_099_llm_client_raw_llm_query__expr_0(self):
        """expand_children('llm_client/raw_llm_query.jsonl', '$expr_0'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'llm_client/raw_llm_query.jsonl', '$expr_0')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$await_1'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['expr_stmt'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$expr_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'llm_client/raw_llm_query.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$expr_0')


if __name__ == "__main__":
    unittest.main()
