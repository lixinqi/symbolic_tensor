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

    def test_000_symbolic_tensor_function_coding_agent__alias_16(self):
        """expand_children('symbolic_tensor/function/coding_agent.jsonl', '$alias_16'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/coding_agent.jsonl', '$alias_16')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['subprocess'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['alias__name'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$alias_16')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))

    def test_001_symbolic_tensor_function_select_qkv_indexes__With_2(self):
        """expand_children('symbolic_tensor/function/select_qkv_indexes.jsonl', '$With_2'): 8 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/select_qkv_indexes.jsonl', '$With_2')
        self.assertEqual(len(children), 8)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Assign_36', '$withitem_2', '$Assign_37', '$Assign_38', '$Expr_22', '$Expr_23', '$Expr_24', '$Expr_25'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['With__body', 'With__items', 'With__body', 'With__body', 'With__body', 'With__body', 'With__body', 'With__body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$With_2')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/select_qkv_indexes.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$With_2')

    def test_002_symbolic_tensor_tensor_util_slice_tensor__FunctionDef_3(self):
        """expand_children('symbolic_tensor/tensor_util/slice_tensor.jsonl', '$FunctionDef_3'): 19 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/slice_tensor.jsonl', '$FunctionDef_3')
        self.assertEqual(len(children), 19)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$arguments_3', '$Expr_4', 'slice_tensor', '$Attribute_31', '$Assert_0', '$Assign_7', '$Assign_8', '$Assign_9', '$For_2', '$If_7', '$Assign_15', '$AnnAssign_0', '$For_3', '$If_8', '$Assign_18', '$Assign_19', '$Assign_20', '$Assert_1', '$Return_9'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['FunctionDef__args', 'FunctionDef__body', 'FunctionDef__name', 'FunctionDef__returns', 'FunctionDef__body', 'FunctionDef__body', 'FunctionDef__body', 'FunctionDef__body', 'FunctionDef__body', 'FunctionDef__body', 'FunctionDef__body', 'FunctionDef__body', 'FunctionDef__body', 'FunctionDef__body', 'FunctionDef__body', 'FunctionDef__body', 'FunctionDef__body', 'FunctionDef__body', 'FunctionDef__body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$FunctionDef_3')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/slice_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$FunctionDef_3')

    def test_003_symbolic_tensor_tensor_util_assign_tensor__FunctionDef_1(self):
        """expand_children('symbolic_tensor/tensor_util/assign_tensor.jsonl', '$FunctionDef_1'): 7 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/assign_tensor.jsonl', '$FunctionDef_1')
        self.assertEqual(len(children), 7)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$arguments_1', '$Expr_1', 'assign_tensor', 'None', '$Assert_0', '$Assign_4', '$For_0'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['FunctionDef__args', 'FunctionDef__body', 'FunctionDef__name', 'FunctionDef__returns', 'FunctionDef__body', 'FunctionDef__body', 'FunctionDef__body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$FunctionDef_1')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/assign_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$FunctionDef_1')

    def test_004_symbolic_tensor_tensor_util_assign_view__With_7(self):
        """expand_children('symbolic_tensor/tensor_util/assign_view.jsonl', '$With_7'): 9 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/assign_view.jsonl', '$With_7')
        self.assertEqual(len(children), 9)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Assign_25', '$withitem_7', '$Assign_26', '$Expr_38', '$Assign_27', '$Expr_39', '$Assign_28', '$Expr_40', '$Expr_41'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['With__body', 'With__items', 'With__body', 'With__body', 'With__body', 'With__body', 'With__body', 'With__body', 'With__body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$With_7')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/assign_view.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$With_7')

    def test_005_symbolic_tensor_function_fork_tensor__arg_30(self):
        """expand_children('symbolic_tensor/function/fork_tensor.jsonl', '$arg_30'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/fork_tensor.jsonl', '$arg_30')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['flat_index'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['arg__arg'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$arg_30')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))

    def test_006_symbolic_tensor_function_merge_backward__ExceptHandler_0(self):
        """expand_children('symbolic_tensor/function/merge_backward.jsonl', '$ExceptHandler_0'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/merge_backward.jsonl', '$ExceptHandler_0')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Expr_40', 'AssertionError'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['ExceptHandler__body', 'ExceptHandler__type'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$ExceptHandler_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/merge_backward.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$ExceptHandler_0')

    def test_007_symbolic_tensor_function_fork_tensor__ListComp_0(self):
        """expand_children('symbolic_tensor/function/fork_tensor.jsonl', '$ListComp_0'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/fork_tensor.jsonl', '$ListComp_0')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$comprehension_0'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['ListComp__generators'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$ListComp_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/fork_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$ListComp_0')

    def test_008_symbolic_tensor_tensor_util_register_tensor_ops__FunctionDef_4(self):
        """expand_children('symbolic_tensor/tensor_util/register_tensor_ops.jsonl', '$FunctionDef_4'): 4 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl', '$FunctionDef_4')
        self.assertEqual(len(children), 4)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$arguments_4', '$ImportFrom_4', 'st_patch', '$Return_4'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['FunctionDef__args', 'FunctionDef__body', 'FunctionDef__name', 'FunctionDef__body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$FunctionDef_4')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$FunctionDef_4')

    def test_009_symbolic_tensor_function_slice_and_concat_attention_forward__ImportFrom_3(self):
        """expand_children('symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', '$ImportFrom_3'): 3 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', '$ImportFrom_3')
        self.assertEqual(len(children), 3)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['0', 'experience.symbolic_tensor.tensor_util.make_tensor', '$alias_6'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['ImportFrom__level', 'ImportFrom__module', 'ImportFrom__names'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$ImportFrom_3')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$ImportFrom_3')

    def test_010_symbolic_tensor_tensor_util_load_tensor__arg_11(self):
        """expand_children('symbolic_tensor/tensor_util/load_tensor.jsonl', '$arg_11'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/load_tensor.jsonl', '$arg_11')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['actual'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['arg__arg'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$arg_11')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))

    def test_011_symbolic_tensor_module_st_moe__alias_8(self):
        """expand_children('symbolic_tensor/module/st_moe.jsonl', '$alias_8'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/module/st_moe.jsonl', '$alias_8')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['List'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['alias__name'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$alias_8')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))

    def test_012_symbolic_tensor_function_with_dense_view__arguments_5(self):
        """expand_children('symbolic_tensor/function/with_dense_view.jsonl', '$arguments_5'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/with_dense_view.jsonl', '$arguments_5')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$arg_12'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['arguments__args'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$arguments_5')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/with_dense_view.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$arguments_5')

    def test_013_symbolic_tensor_tensor_util_make_tensor__withitem_4(self):
        """expand_children('symbolic_tensor/tensor_util/make_tensor.jsonl', '$withitem_4'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/make_tensor.jsonl', '$withitem_4')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Call_80', 'f'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['withitem__context_expr', 'withitem__optional_vars'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$withitem_4')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/make_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$withitem_4')

    def test_014_symbolic_tensor_data_loader_sole_file_batch_data_loader__arguments_4(self):
        """expand_children('symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', '$arguments_4'): 6 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', '$arguments_4')
        self.assertEqual(len(children), 6)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$arg_8', 'None', '$arg_9', 'None', '$arg_10', '$arg_11'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['arguments__args', 'arguments__defaults', 'arguments__args', 'arguments__defaults', 'arguments__args', 'arguments__args'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$arguments_4')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$arguments_4')

    def test_015_symbolic_tensor_function_slice_attention_backward__If_1(self):
        """expand_children('symbolic_tensor/function/slice_attention_backward.jsonl', '$If_1'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_attention_backward.jsonl', '$If_1')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Return_2'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['If__body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$If_1')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_attention_backward.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$If_1')

    def test_016_symbolic_tensor_function_get_edit_distance_ratio__ImportFrom_1(self):
        """expand_children('symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$ImportFrom_1'): 3 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$ImportFrom_1')
        self.assertEqual(len(children), 3)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['0', 'experience.symbolic_tensor.tensor_util.make_tensor', '$alias_6'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['ImportFrom__level', 'ImportFrom__module', 'ImportFrom__names'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$ImportFrom_1')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$ImportFrom_1')

    def test_017_symbolic_tensor_function_slice_view__With_1(self):
        """expand_children('symbolic_tensor/function/slice_view.jsonl', '$With_1'): 6 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_view.jsonl', '$With_1')
        self.assertEqual(len(children), 6)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Assign_34', '$withitem_1', '$Assign_35', '$Expr_19', '$Expr_20', '$Expr_21'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['With__body', 'With__items', 'With__body', 'With__body', 'With__body', 'With__body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$With_1')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_view.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$With_1')

    def test_018_symbolic_tensor_function_slice_tensor__FunctionDef_0(self):
        """expand_children('symbolic_tensor/function/slice_tensor.jsonl', '$FunctionDef_0'): 9 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_tensor.jsonl', '$FunctionDef_0')
        self.assertEqual(len(children), 9)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$arguments_0', '$Assign_0', 'staticmethod', 'forward', '$Assign_1', '$Expr_0', '$Expr_1', '$Assign_2', '$Return_0'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['FunctionDef__args', 'FunctionDef__body', 'FunctionDef__decorator_list', 'FunctionDef__name', 'FunctionDef__body', 'FunctionDef__body', 'FunctionDef__body', 'FunctionDef__body', 'FunctionDef__body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$FunctionDef_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$FunctionDef_0')

    def test_019_symbolic_tensor_function_st_stack__arguments_6(self):
        """expand_children('symbolic_tensor/function/st_stack.jsonl', '$arguments_6'): 3 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_stack.jsonl', '$arguments_6')
        self.assertEqual(len(children), 3)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$arg_15', '$arg_17', '$arg_16'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['arguments__args', 'arguments__vararg', 'arguments__args'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$arguments_6')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_stack.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$arguments_6')

    def test_020_symbolic_tensor_function_merge_forward__arg_12(self):
        """expand_children('symbolic_tensor/function/merge_forward.jsonl', '$arg_12'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/merge_forward.jsonl', '$arg_12')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['expected'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['arg__arg'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$arg_12')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))

    def test_021_symbolic_tensor_function_merge__For_4(self):
        """expand_children('symbolic_tensor/function/merge.jsonl', '$For_4'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/merge.jsonl', '$For_4')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Assign_48', '$Expr_46'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['For__body', 'For__body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$For_4')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/merge.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$For_4')

    def test_022_symbolic_tensor_function_st_moe_backward__arg_26(self):
        """expand_children('symbolic_tensor/function/st_moe_backward.jsonl', '$arg_26'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe_backward.jsonl', '$arg_26')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['int', 'topk'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['arg__annotation', 'arg__arg'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$arg_26')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))

    def test_023_symbolic_tensor_data_loader_sole_file_batch_data_loader__keyword_1(self):
        """expand_children('symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', '$keyword_1'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', '$keyword_1')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['symlink'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['keyword__arg'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$keyword_1')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))

    def test_024_symbolic_tensor_function_st_moe_backward__comprehension_11(self):
        """expand_children('symbolic_tensor/function/st_moe_backward.jsonl', '$comprehension_11'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe_backward.jsonl', '$comprehension_11')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['0'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['comprehension__is_async'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$comprehension_11')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))

    def test_025_symbolic_tensor_function_fork_tensor__arg_23(self):
        """expand_children('symbolic_tensor/function/fork_tensor.jsonl', '$arg_23'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/fork_tensor.jsonl', '$arg_23')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['ctx'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['arg__arg'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$arg_23')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))

    def test_026_fs_util_get_nested_list_file_pathes__comprehension_3(self):
        """expand_children('fs_util/get_nested_list_file_pathes.jsonl', '$comprehension_3'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'fs_util/get_nested_list_file_pathes.jsonl', '$comprehension_3')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['0'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['comprehension__is_async'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$comprehension_3')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))

    def test_027_symbolic_tensor_function_st_moe__With_1(self):
        """expand_children('symbolic_tensor/function/st_moe.jsonl', '$With_1'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe.jsonl', '$With_1')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Assign_30', '$withitem_1'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['With__body', 'With__items'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$With_1')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$With_1')

    def test_028_symbolic_tensor_module_st_moe__arg_13(self):
        """expand_children('symbolic_tensor/module/st_moe.jsonl', '$arg_13'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/module/st_moe.jsonl', '$arg_13')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['self'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['arg__arg'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$arg_13')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))

    def test_029_symbolic_tensor_tensor_util_assign_tensor__ImportFrom_1(self):
        """expand_children('symbolic_tensor/tensor_util/assign_tensor.jsonl', '$ImportFrom_1'): 3 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/assign_tensor.jsonl', '$ImportFrom_1')
        self.assertEqual(len(children), 3)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['0', 'experience.symbolic_tensor.tensor_util.make_tensor', '$alias_6'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['ImportFrom__level', 'ImportFrom__module', 'ImportFrom__names'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$ImportFrom_1')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/assign_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$ImportFrom_1')

    def test_030_symbolic_tensor_function_get_edit_distance_ratio__For_4(self):
        """expand_children('symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$For_4'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$For_4')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$If_2'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['For__body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$For_4')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$For_4')

    def test_031_llm_client_coding_agent_task_handler__ImportFrom_1(self):
        """expand_children('llm_client/coding_agent_task_handler.jsonl', '$ImportFrom_1'): 3 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'llm_client/coding_agent_task_handler.jsonl', '$ImportFrom_1')
        self.assertEqual(len(children), 3)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['0', 'experience.llm_client.agent_task', '$alias_5'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['ImportFrom__level', 'ImportFrom__module', 'ImportFrom__names'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$ImportFrom_1')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'llm_client/coding_agent_task_handler.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$ImportFrom_1')

    def test_032_symbolic_tensor_function_coding_agent__If_8(self):
        """expand_children('symbolic_tensor/function/coding_agent.jsonl', '$If_8'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/coding_agent.jsonl', '$If_8')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Return_6'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['If__body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$If_8')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/coding_agent.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$If_8')

    def test_033_test_test_attention_vs_traditional__Import_2(self):
        """expand_children('test/test_attention_vs_traditional.jsonl', '$Import_2'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'test/test_attention_vs_traditional.jsonl', '$Import_2')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$alias_2'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['Import__names'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$Import_2')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'test/test_attention_vs_traditional.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$Import_2')

    def test_034_llm_client_raw_llm_task_handler__module_(self):
        """expand_children('llm_client/raw_llm_task_handler.jsonl', '<module>'): 8 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'llm_client/raw_llm_task_handler.jsonl', '<module>')
        self.assertEqual(len(children), 8)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Import_0', '$Import_1', '$ImportFrom_0', '$ImportFrom_1', '$ImportFrom_2', '$FunctionDef_0', '$FunctionDef_1', '$ClassDef_0'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['Module__body', 'Module__body', 'Module__body', 'Module__body', 'Module__body', 'Module__body', 'Module__body', 'Module__body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '<module>')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'llm_client/raw_llm_task_handler.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '<module>')

    def test_035_symbolic_tensor_function_slice_attention_backward__alias_4(self):
        """expand_children('symbolic_tensor/function/slice_attention_backward.jsonl', '$alias_4'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_attention_backward.jsonl', '$alias_4')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['Callable'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['alias__name'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$alias_4')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))

    def test_036_symbolic_tensor_tensor_util_slice_view__withitem_4(self):
        """expand_children('symbolic_tensor/tensor_util/slice_view.jsonl', '$withitem_4'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/slice_view.jsonl', '$withitem_4')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Call_85', 'f'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['withitem__context_expr', 'withitem__optional_vars'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$withitem_4')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/slice_view.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$withitem_4')

    def test_037_symbolic_tensor_data_loader_sole_file_batch_data_loader__With_5(self):
        """expand_children('symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', '$With_5'): 6 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', '$With_5')
        self.assertEqual(len(children), 6)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$For_4', '$withitem_5', '$Assign_17', '$Assign_18', '$Expr_26', '$Expr_27'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['With__body', 'With__items', 'With__body', 'With__body', 'With__body', 'With__body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$With_5')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$With_5')

    def test_038_symbolic_tensor_tensor_util_make_tensor__If_2(self):
        """expand_children('symbolic_tensor/tensor_util/make_tensor.jsonl', '$If_2'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/make_tensor.jsonl', '$If_2')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Assert_0', '$Return_4'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['If__body', 'If__body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$If_2')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/make_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$If_2')

    def test_039_symbolic_tensor_function_slice_attention__arg_15(self):
        """expand_children('symbolic_tensor/function/slice_attention.jsonl', '$arg_15'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_attention.jsonl', '$arg_15')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['flat_index'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['arg__arg'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$arg_15')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))

    def test_040_symbolic_tensor_function_select_qkv_indexes__If_10(self):
        """expand_children('symbolic_tensor/function/select_qkv_indexes.jsonl', '$If_10'): 3 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/select_qkv_indexes.jsonl', '$If_10')
        self.assertEqual(len(children), 3)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Expr_12', '$Expr_13', '$If_11'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['If__body', 'If__orelse', 'If__orelse'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$If_10')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/select_qkv_indexes.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$If_10')

    def test_041_symbolic_tensor_module_st_moe__With_0(self):
        """expand_children('symbolic_tensor/module/st_moe.jsonl', '$With_0'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/module/st_moe.jsonl', '$With_0')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Return_1', '$withitem_0'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['With__body', 'With__items'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$With_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/module/st_moe.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$With_0')

    def test_042_symbolic_tensor_tensor_util_dense_to_sparse__withitem_5(self):
        """expand_children('symbolic_tensor/tensor_util/dense_to_sparse.jsonl', '$withitem_5'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', '$withitem_5')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Call_76', 'tmpdir'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['withitem__context_expr', 'withitem__optional_vars'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$withitem_5')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$withitem_5')

    def test_043_symbolic_tensor_module_st_moe__keyword_9(self):
        """expand_children('symbolic_tensor/module/st_moe.jsonl', '$keyword_9'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/module/st_moe.jsonl', '$keyword_9')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['topk'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['keyword__arg'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$keyword_9')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))

    def test_044_symbolic_tensor_tensor_util_slice_view__arguments_3(self):
        """expand_children('symbolic_tensor/tensor_util/slice_view.jsonl', '$arguments_3'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/slice_view.jsonl', '$arguments_3')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$arg_6', '$arg_7'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['arguments__args', 'arguments__args'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$arguments_3')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/slice_view.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$arguments_3')

    def test_045_symbolic_tensor_tensor_util_dump_view__module_(self):
        """expand_children('symbolic_tensor/tensor_util/dump_view.jsonl', '<module>'): 9 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/dump_view.jsonl', '<module>')
        self.assertEqual(len(children), 9)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Import_0', '$Import_1', '$Import_2', '$ImportFrom_0', '$FunctionDef_0', '$FunctionDef_1', '$FunctionDef_2', '$FunctionDef_3', '$If_1'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['Module__body', 'Module__body', 'Module__body', 'Module__body', 'Module__body', 'Module__body', 'Module__body', 'Module__body', 'Module__body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '<module>')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/dump_view.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '<module>')

    def test_046_symbolic_tensor_data_loader_sole_file_batch_data_loader__withitem_9(self):
        """expand_children('symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', '$withitem_9'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', '$withitem_9')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Call_89', 'tmpdir'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['withitem__context_expr', 'withitem__optional_vars'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$withitem_9')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$withitem_9')

    def test_047_symbolic_tensor_function_st_moe__ImportFrom_2(self):
        """expand_children('symbolic_tensor/function/st_moe.jsonl', '$ImportFrom_2'): 6 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe.jsonl', '$ImportFrom_2')
        self.assertEqual(len(children), 6)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['0', 'experience.symbolic_tensor.function.st_moe_backward', '$alias_11', '$alias_12', '$alias_13', '$alias_14'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['ImportFrom__level', 'ImportFrom__module', 'ImportFrom__names', 'ImportFrom__names', 'ImportFrom__names', 'ImportFrom__names'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$ImportFrom_2')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$ImportFrom_2')

    def test_048_symbolic_tensor_function_slice_attention_backward__arguments_4(self):
        """expand_children('symbolic_tensor/function/slice_attention_backward.jsonl', '$arguments_4'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_attention_backward.jsonl', '$arguments_4')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$arg_26', '$arg_27'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['arguments__args', 'arguments__args'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$arguments_4')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_attention_backward.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$arguments_4')

    def test_049_test_test_attention_vs_traditional__FunctionDef_1(self):
        """expand_children('test/test_attention_vs_traditional.jsonl', '$FunctionDef_1'): 6 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'test/test_attention_vs_traditional.jsonl', '$FunctionDef_1')
        self.assertEqual(len(children), 6)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$arguments_1', '$Expr_1', 'get_frames', '$Assign_3', '$If_1', '$Return_3'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['FunctionDef__args', 'FunctionDef__body', 'FunctionDef__name', 'FunctionDef__body', 'FunctionDef__body', 'FunctionDef__body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$FunctionDef_1')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'test/test_attention_vs_traditional.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$FunctionDef_1')

    def test_050_symbolic_tensor_function_st_moe_backward__withitem_3(self):
        """expand_children('symbolic_tensor/function/st_moe_backward.jsonl', '$withitem_3'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe_backward.jsonl', '$withitem_3')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Call_197', 'tmpdir'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['withitem__context_expr', 'withitem__optional_vars'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$withitem_3')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe_backward.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$withitem_3')

    def test_051_symbolic_tensor_function_merge_backward__keyword_7(self):
        """expand_children('symbolic_tensor/function/merge_backward.jsonl', '$keyword_7'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/merge_backward.jsonl', '$keyword_7')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['axis'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['keyword__arg'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$keyword_7')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))

    def test_052_symbolic_tensor_tensor_util_assign_tensor__alias_7(self):
        """expand_children('symbolic_tensor/tensor_util/assign_tensor.jsonl', '$alias_7'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/assign_tensor.jsonl', '$alias_7')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['todo_tensor_like'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['alias__name'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$alias_7')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))

    def test_053_symbolic_tensor_module_st_moe__keyword_1(self):
        """expand_children('symbolic_tensor/module/st_moe.jsonl', '$keyword_1'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/module/st_moe.jsonl', '$keyword_1')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['text'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['keyword__arg'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$keyword_1')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))

    def test_054_symbolic_tensor_function_st_attention__With_4(self):
        """expand_children('symbolic_tensor/function/st_attention.jsonl', '$With_4'): 6 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_attention.jsonl', '$With_4')
        self.assertEqual(len(children), 6)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Assign_29', '$withitem_4', '$Assign_30', '$Assign_31', '$Expr_33', '$Expr_34'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['With__body', 'With__items', 'With__body', 'With__body', 'With__body', 'With__body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$With_4')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_attention.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$With_4')

    def test_055_symbolic_tensor_function_slice_view__arguments_6(self):
        """expand_children('symbolic_tensor/function/slice_view.jsonl', '$arguments_6'): 3 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_view.jsonl', '$arguments_6')
        self.assertEqual(len(children), 3)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$arg_14', '$arg_15', '$arg_16'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['arguments__args', 'arguments__args', 'arguments__args'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$arguments_6')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_view.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$arguments_6')

    def test_056_test_test_transform_method_time_comparison__With_1(self):
        """expand_children('test/test_transform_method_time_comparison.jsonl', '$With_1'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'test/test_transform_method_time_comparison.jsonl', '$With_1')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Assign_10', '$withitem_1'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['With__body', 'With__items'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$With_1')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'test/test_transform_method_time_comparison.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$With_1')

    def test_057_test_test_st_attention_followed_by_st_moe__keyword_0(self):
        """expand_children('test/test_st_attention_followed_by_st_moe.jsonl', '$keyword_0'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'test/test_st_attention_followed_by_st_moe.jsonl', '$keyword_0')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['experience_shape'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['keyword__arg'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$keyword_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))

    def test_058_symbolic_tensor_function_get_edit_distance_ratio__arg_12(self):
        """expand_children('symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$arg_12'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$arg_12')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['actual'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['arg__arg'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$arg_12')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))

    def test_059_symbolic_tensor_function_slice_attention_backward__alias_6(self):
        """expand_children('symbolic_tensor/function/slice_attention_backward.jsonl', '$alias_6'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_attention_backward.jsonl', '$alias_6')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['List'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['alias__name'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$alias_6')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))

    def test_060_symbolic_tensor_tensor_util_make_none_tensor__If_2(self):
        """expand_children('symbolic_tensor/tensor_util/make_none_tensor.jsonl', '$If_2'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', '$If_2')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Expr_4', '$Expr_5'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['If__body', 'If__body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$If_2')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$If_2')

    def test_061_symbolic_tensor_function_slice_attention_backward__keyword_12(self):
        """expand_children('symbolic_tensor/function/slice_attention_backward.jsonl', '$keyword_12'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_attention_backward.jsonl', '$keyword_12')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['dim'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['keyword__arg'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$keyword_12')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))

    def test_062_symbolic_tensor_function_slice_attention_backward__With_5(self):
        """expand_children('symbolic_tensor/function/slice_attention_backward.jsonl', '$With_5'): 13 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_attention_backward.jsonl', '$With_5')
        self.assertEqual(len(children), 13)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Assign_51', '$withitem_5', '$Expr_30', '$Assign_52', '$Assign_53', '$Assign_54', '$Assign_55', '$Assign_56', '$Assign_57', '$Assign_58', '$Assign_59', '$Expr_31', '$For_5'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['With__body', 'With__items', 'With__body', 'With__body', 'With__body', 'With__body', 'With__body', 'With__body', 'With__body', 'With__body', 'With__body', 'With__body', 'With__body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$With_5')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_attention_backward.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$With_5')

    def test_063_symbolic_tensor_tensor_util_dump_tensor__Import_3(self):
        """expand_children('symbolic_tensor/tensor_util/dump_tensor.jsonl', '$Import_3'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/dump_tensor.jsonl', '$Import_3')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$alias_3'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['Import__names'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$Import_3')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/dump_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$Import_3')

    def test_064_symbolic_tensor_function_slice_attention_backward__FunctionDef_3(self):
        """expand_children('symbolic_tensor/function/slice_attention_backward.jsonl', '$FunctionDef_3'): 3 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_attention_backward.jsonl', '$FunctionDef_3')
        self.assertEqual(len(children), 3)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$arguments_3', '$If_7', 'run_test'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['FunctionDef__args', 'FunctionDef__body', 'FunctionDef__name'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$FunctionDef_3')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_attention_backward.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$FunctionDef_3')

    def test_065_symbolic_tensor_function_st_copy__arg_10(self):
        """expand_children('symbolic_tensor/function/st_copy.jsonl', '$arg_10'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_copy.jsonl', '$arg_10')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['actual'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['arg__arg'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$arg_10')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))

    def test_066_symbolic_tensor_tensor_util_todo_tensor_like__With_6(self):
        """expand_children('symbolic_tensor/tensor_util/todo_tensor_like.jsonl', '$With_6'): 5 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl', '$With_6')
        self.assertEqual(len(children), 5)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Assign_17', '$withitem_6', '$Assign_18', '$Expr_21', '$Expr_22'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['With__body', 'With__items', 'With__body', 'With__body', 'With__body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$With_6')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$With_6')

    def test_067_symbolic_tensor_tensor_util_make_tensor__alias_3(self):
        """expand_children('symbolic_tensor/tensor_util/make_tensor.jsonl', '$alias_3'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/make_tensor.jsonl', '$alias_3')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['torch'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['alias__name'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$alias_3')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))

    def test_068_symbolic_tensor_tensor_util_dense_to_sparse__alias_3(self):
        """expand_children('symbolic_tensor/tensor_util/dense_to_sparse.jsonl', '$alias_3'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', '$alias_3')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['List'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['alias__name'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$alias_3')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))

    def test_069_symbolic_tensor_function_slice_attention_forward__alias_7(self):
        """expand_children('symbolic_tensor/function/slice_attention_forward.jsonl', '$alias_7'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_attention_forward.jsonl', '$alias_7')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['tempfile'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['alias__name'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$alias_7')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))

    def test_070_symbolic_tensor_function_st_moe_forward__Import_2(self):
        """expand_children('symbolic_tensor/function/st_moe_forward.jsonl', '$Import_2'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe_forward.jsonl', '$Import_2')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$alias_2'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['Import__names'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$Import_2')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe_forward.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$Import_2')

    def test_071_symbolic_tensor_function_st_moe_backward__keyword_47(self):
        """expand_children('symbolic_tensor/function/st_moe_backward.jsonl', '$keyword_47'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe_backward.jsonl', '$keyword_47')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['dtype'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['keyword__arg'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$keyword_47')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))

    def test_072_symbolic_tensor_function_st_copy__withitem_0(self):
        """expand_children('symbolic_tensor/function/st_copy.jsonl', '$withitem_0'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_copy.jsonl', '$withitem_0')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Call_0', 'tmp_dump_dir'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['withitem__context_expr', 'withitem__optional_vars'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$withitem_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_copy.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$withitem_0')

    def test_073_symbolic_tensor_function_merge_forward__FunctionDef_2(self):
        """expand_children('symbolic_tensor/function/merge_forward.jsonl', '$FunctionDef_2'): 7 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/merge_forward.jsonl', '$FunctionDef_2')
        self.assertEqual(len(children), 7)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$arguments_2', '$Expr_2', '_write_storage', 'None', '$Assign_2', '$Expr_3', '$With_1'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['FunctionDef__args', 'FunctionDef__body', 'FunctionDef__name', 'FunctionDef__returns', 'FunctionDef__body', 'FunctionDef__body', 'FunctionDef__body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$FunctionDef_2')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/merge_forward.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$FunctionDef_2')

    def test_074_symbolic_tensor_data_loader_sole_file_batch_data_loader__arguments_0(self):
        """expand_children('symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', '$arguments_0'): 3 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', '$arguments_0')
        self.assertEqual(len(children), 3)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$arg_0', 'None', '$arg_1'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['arguments__args', 'arguments__defaults', 'arguments__args'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$arguments_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$arguments_0')

    def test_075_symbolic_tensor_tensor_util_load_tensor__With_2(self):
        """expand_children('symbolic_tensor/tensor_util/load_tensor.jsonl', '$With_2'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/load_tensor.jsonl', '$With_2')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Assign_9', '$withitem_2'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['With__body', 'With__items'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$With_2')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/load_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$With_2')

    def test_076_test_test_attention_vs_traditional__withitem_0(self):
        """expand_children('test/test_attention_vs_traditional.jsonl', '$withitem_0'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'test/test_attention_vs_traditional.jsonl', '$withitem_0')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Call_6', 'f'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['withitem__context_expr', 'withitem__optional_vars'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$withitem_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'test/test_attention_vs_traditional.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$withitem_0')

    def test_077_symbolic_tensor_function_coding_agent__comprehension_4(self):
        """expand_children('symbolic_tensor/function/coding_agent.jsonl', '$comprehension_4'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/coding_agent.jsonl', '$comprehension_4')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['0'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['comprehension__is_async'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$comprehension_4')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))

    def test_078_symbolic_tensor_function_st_stack__arguments_5(self):
        """expand_children('symbolic_tensor/function/st_stack.jsonl', '$arguments_5'): 5 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_stack.jsonl', '$arguments_5')
        self.assertEqual(len(children), 5)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$arg_11', '0', '$arg_12', '$arg_13', '$arg_14'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['arguments__args', 'arguments__defaults', 'arguments__args', 'arguments__args', 'arguments__args'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$arguments_5')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_stack.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$arguments_5')

    def test_079_symbolic_tensor_tensor_util_dump_view__With_2(self):
        """expand_children('symbolic_tensor/tensor_util/dump_view.jsonl', '$With_2'): 7 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/dump_view.jsonl', '$With_2')
        self.assertEqual(len(children), 7)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Assign_12', '$withitem_3', '$Assign_13', '$withitem_4', '$Expr_17', '$Assign_14', '$For_2'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['With__body', 'With__items', 'With__body', 'With__items', 'With__body', 'With__body', 'With__body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$With_2')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/dump_view.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$With_2')

    def test_080_symbolic_tensor_tensor_util_get_diff_tensor__Import_2(self):
        """expand_children('symbolic_tensor/tensor_util/get_diff_tensor.jsonl', '$Import_2'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', '$Import_2')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$alias_2'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['Import__names'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$Import_2')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$Import_2')

    def test_081_fs_util_get_nested_list_file_pathes__comprehension_0(self):
        """expand_children('fs_util/get_nested_list_file_pathes.jsonl', '$comprehension_0'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'fs_util/get_nested_list_file_pathes.jsonl', '$comprehension_0')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['0'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['comprehension__is_async'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$comprehension_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))

    def test_082_symbolic_tensor_tensor_util_pack_tensor__withitem_3(self):
        """expand_children('symbolic_tensor/tensor_util/pack_tensor.jsonl', '$withitem_3'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/pack_tensor.jsonl', '$withitem_3')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Call_28', 'tmpdir'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['withitem__context_expr', 'withitem__optional_vars'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$withitem_3')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/pack_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$withitem_3')

    def test_083_symbolic_tensor_function_st_stack__alias_9(self):
        """expand_children('symbolic_tensor/function/st_stack.jsonl', '$alias_9'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_stack.jsonl', '$alias_9')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['assign_tensor'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['alias__name'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$alias_9')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))

    def test_084_symbolic_tensor_function_get_edit_distance_ratio__With_2(self):
        """expand_children('symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$With_2'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$With_2')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Return_9', '$withitem_3'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['With__body', 'With__items'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$With_2')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$With_2')

    def test_085_symbolic_tensor_tensor_util_make_none_tensor__FunctionDef_1(self):
        """expand_children('symbolic_tensor/tensor_util/make_none_tensor.jsonl', '$FunctionDef_1'): 3 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', '$FunctionDef_1')
        self.assertEqual(len(children), 3)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$arguments_1', '$If_1', 'run_test'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['FunctionDef__args', 'FunctionDef__body', 'FunctionDef__name'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$FunctionDef_1')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$FunctionDef_1')

    def test_086_symbolic_tensor_tensor_util_assign_tensor__arg_1(self):
        """expand_children('symbolic_tensor/tensor_util/assign_tensor.jsonl', '$arg_1'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/assign_tensor.jsonl', '$arg_1')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Subscript_0', 'coordinates'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['arg__annotation', 'arg__arg'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$arg_1')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/assign_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$arg_1')

    def test_087_llm_client_raw_llm_task_handler__FunctionDef_1(self):
        """expand_children('llm_client/raw_llm_task_handler.jsonl', '$FunctionDef_1'): 7 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'llm_client/raw_llm_task_handler.jsonl', '$FunctionDef_1')
        self.assertEqual(len(children), 7)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$arguments_1', '$Expr_2', '_grep_by_file_content_hint', '$Subscript_0', '$Assign_1', '$For_1', '$Return_2'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['FunctionDef__args', 'FunctionDef__body', 'FunctionDef__name', 'FunctionDef__returns', 'FunctionDef__body', 'FunctionDef__body', 'FunctionDef__body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$FunctionDef_1')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'llm_client/raw_llm_task_handler.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$FunctionDef_1')

    def test_088_symbolic_tensor_tensor_util_slice_view__With_1(self):
        """expand_children('symbolic_tensor/tensor_util/slice_view.jsonl', '$With_1'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/slice_view.jsonl', '$With_1')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Expr_16', '$withitem_1'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['With__body', 'With__items'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$With_1')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/slice_view.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$With_1')

    def test_089_symbolic_tensor_function_slice_attention__arguments_0(self):
        """expand_children('symbolic_tensor/function/slice_attention.jsonl', '$arguments_0'): 13 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_attention.jsonl', '$arguments_0')
        self.assertEqual(len(children), 13)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$arg_0', 'False', '$arg_1', 'None', '$arg_2', "''", '$arg_3', "'raw_llm_api'", '$arg_4', 'None', '$arg_5', '$arg_6', '$arg_7'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['arguments__args', 'arguments__defaults', 'arguments__args', 'arguments__defaults', 'arguments__args', 'arguments__defaults', 'arguments__args', 'arguments__defaults', 'arguments__args', 'arguments__defaults', 'arguments__args', 'arguments__args', 'arguments__args'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$arguments_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_attention.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$arguments_0')

    def test_090_llm_client_raw_llm_task_handler__arg_0(self):
        """expand_children('llm_client/raw_llm_task_handler.jsonl', '$arg_0'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'llm_client/raw_llm_task_handler.jsonl', '$arg_0')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['nested'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['arg__arg'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$arg_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))

    def test_091_symbolic_tensor_function_merge_forward__keyword_4(self):
        """expand_children('symbolic_tensor/function/merge_forward.jsonl', '$keyword_4'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/merge_forward.jsonl', '$keyword_4')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['axis'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['keyword__arg'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$keyword_4')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))

    def test_092_symbolic_tensor_function_slice_tensor__arg_7(self):
        """expand_children('symbolic_tensor/function/slice_tensor.jsonl', '$arg_7'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_tensor.jsonl', '$arg_7')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['name'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['arg__arg'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$arg_7')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))

    def test_093_llm_client_raw_llm_query__keyword_4(self):
        """expand_children('llm_client/raw_llm_query.jsonl', '$keyword_4'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'llm_client/raw_llm_query.jsonl', '$keyword_4')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['stream'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['keyword__arg'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$keyword_4')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))

    def test_094_symbolic_tensor_function_st_attention__With_8(self):
        """expand_children('symbolic_tensor/function/st_attention.jsonl', '$With_8'): 7 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_attention.jsonl', '$With_8')
        self.assertEqual(len(children), 7)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Assign_45', '$withitem_8', '$Assign_46', '$Assign_47', '$Expr_46', '$Assign_48', '$Expr_47'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['With__body', 'With__items', 'With__body', 'With__body', 'With__body', 'With__body', 'With__body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$With_8')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_attention.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$With_8')

    def test_095_symbolic_tensor_function_st_moe__Import_0(self):
        """expand_children('symbolic_tensor/function/st_moe.jsonl', '$Import_0'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe.jsonl', '$Import_0')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$alias_0'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['Import__names'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$Import_0')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$Import_0')

    def test_096_symbolic_tensor_function_slice_view__FunctionDef_5(self):
        """expand_children('symbolic_tensor/function/slice_view.jsonl', '$FunctionDef_5'): 14 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_view.jsonl', '$FunctionDef_5')
        self.assertEqual(len(children), 14)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$arguments_5', '$Expr_4', 'slice_backward', '$Subscript_9', '$If_3', '$Assign_8', '$Assign_9', '$Expr_5', '$Assign_10', '$Assign_11', '$Assign_12', '$Assign_13', '$For_5', '$Return_6'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['FunctionDef__args', 'FunctionDef__body', 'FunctionDef__name', 'FunctionDef__returns', 'FunctionDef__body', 'FunctionDef__body', 'FunctionDef__body', 'FunctionDef__body', 'FunctionDef__body', 'FunctionDef__body', 'FunctionDef__body', 'FunctionDef__body', 'FunctionDef__body', 'FunctionDef__body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$FunctionDef_5')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_view.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$FunctionDef_5')

    def test_097_symbolic_tensor_function_fork_tensor__If_6(self):
        """expand_children('symbolic_tensor/function/fork_tensor.jsonl', '$If_6'): 2 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/fork_tensor.jsonl', '$If_6')
        self.assertEqual(len(children), 2)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Assign_39', '$Assign_40'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['If__body', 'If__body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$If_6')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/fork_tensor.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$If_6')

    def test_098_symbolic_tensor_function_st_moe_backward__FunctionDef_8(self):
        """expand_children('symbolic_tensor/function/st_moe_backward.jsonl', '$FunctionDef_8'): 8 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe_backward.jsonl', '$FunctionDef_8')
        self.assertEqual(len(children), 8)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$arguments_8', '$Expr_12', '_flatten_nested_indexes', '$Subscript_21', '$If_5', '$Assign_7', '$For_0', '$Return_12'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['FunctionDef__args', 'FunctionDef__body', 'FunctionDef__name', 'FunctionDef__returns', 'FunctionDef__body', 'FunctionDef__body', 'FunctionDef__body', 'FunctionDef__body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$FunctionDef_8')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe_backward.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$FunctionDef_8')

    def test_099_fs_util_text_merger__If_9(self):
        """expand_children('fs_util/text_merger.jsonl', '$If_9'): 1 children"""
        db = _db()
        children = lexical_scope_expand_children(db, 'fs_util/text_merger.jsonl', '$If_9')
        self.assertEqual(len(children), 1)
        member_tags = [c.member_tag for c in children]
        self.assertEqual(member_tags, ['$Expr_6'])
        relation_tags = [c.relation_tag for c in children]
        self.assertEqual(relation_tags, ['If__body'])
        for rt in relation_tags:
            self.assertIn(rt, LEXICAL_RELATION_TAGS)
        for c in children:
            self.assertEqual(c.owner_tag, '$If_9')
        order_vals = [c.member_order_value for c in children]
        self.assertEqual(order_vals, sorted(order_vals))
        for c in children:
            if c.member_tag.startswith("$"):
                parent = lexical_scope_go_to_parent(db, 'fs_util/text_merger.jsonl', c.member_tag)
                self.assertEqual(parent.owner_tag, '$If_9')


if __name__ == '__main__':
    unittest.main()
