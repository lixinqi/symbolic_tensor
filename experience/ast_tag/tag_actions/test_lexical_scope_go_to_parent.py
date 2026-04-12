"""
test_lexical_scope_go_to_parent: 100 concrete test cases.
Each case: (file_id, member_tag) -> expected (owner_tag, relation_tag, line).
Round-trip: expand_children on the parent includes the original member.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ast_tag_db import AstTagDB
from relation_tag_classification import LEXICAL_RELATION_TAGS
from tag_actions.lexical_scope_go_to_parent import lexical_scope_go_to_parent
from tag_actions.lexical_scope_expand_children import lexical_scope_expand_children


def _db() -> AstTagDB:
    if not hasattr(_db, "_instance"):
        dataset_dir = os.path.join(os.path.dirname(__file__), "..", "test_dataset")
        backend = os.environ.get("AST_TAG_DB_BACKEND", "sqlite")
        if backend == "postgres_age":
            from ast_tag_postgres_age_db import load_jsonl_dataset_into_ast_tag_age_db
            conn_params = os.environ.get("AST_TAG_PG_CONN", "dbname=ast_tag")
            graph_name = os.environ.get("AST_TAG_PG_GRAPH", "ast_tag")
            _db._instance = load_jsonl_dataset_into_ast_tag_age_db(
                dataset_dir, conn_params, graph_name
            )
        else:
            from ast_tag_db import load_jsonl_dataset_into_ast_tag_db
            _db._instance = load_jsonl_dataset_into_ast_tag_db(dataset_dir)
    return _db._instance


class TestLexicalScopeGoToParent(unittest.TestCase):

    def test_000_symbolic_tensor_tensor_util_assign_view__Assign_7(self):
        """go_to_parent('symbolic_tensor/tensor_util/assign_view.jsonl', '$Assign_7') -> '$For_0' via 'For__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/assign_view.jsonl', '$Assign_7')
        self.assertEqual(parent.owner_tag, '$For_0')
        self.assertEqual(parent.relation_tag, 'For__body')
        self.assertEqual(parent.line, 30)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/assign_view.jsonl')
        self.assertEqual(parent.member_tag, '$Assign_7')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Assign_7')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/assign_view.jsonl', '$For_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Assign_7', child_members)

    def test_001_symbolic_tensor_function_symbolic_grad_registry__arg_1(self):
        """go_to_parent('symbolic_tensor/function/symbolic_grad_registry.jsonl', '$arg_1') -> '$arguments_1' via 'arguments__args'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/symbolic_grad_registry.jsonl', '$arg_1')
        self.assertEqual(parent.owner_tag, '$arguments_1')
        self.assertEqual(parent.relation_tag, 'arguments__args')
        self.assertEqual(parent.line, 0)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/symbolic_grad_registry.jsonl')
        self.assertEqual(parent.member_tag, '$arg_1')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$arg_1')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/symbolic_grad_registry.jsonl', '$arguments_1')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$arg_1', child_members)

    def test_002_symbolic_tensor_function_st_moe_backward__Expr_1(self):
        """go_to_parent('symbolic_tensor/function/st_moe_backward.jsonl', '$Expr_1') -> '$FunctionDef_1' via 'FunctionDef__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe_backward.jsonl', '$Expr_1')
        self.assertEqual(parent.owner_tag, '$FunctionDef_1')
        self.assertEqual(parent.relation_tag, 'FunctionDef__body')
        self.assertEqual(parent.line, 29)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_moe_backward.jsonl')
        self.assertEqual(parent.member_tag, '$Expr_1')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Expr_1')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe_backward.jsonl', '$FunctionDef_1')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Expr_1', child_members)

    def test_003_symbolic_tensor_optimizer_st_sgd__alias_18(self):
        """go_to_parent('symbolic_tensor/optimizer/st_sgd.jsonl', '$alias_18') -> '$ImportFrom_6' via 'ImportFrom__names'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/optimizer/st_sgd.jsonl', '$alias_18')
        self.assertEqual(parent.owner_tag, '$ImportFrom_6')
        self.assertEqual(parent.relation_tag, 'ImportFrom__names')
        self.assertEqual(parent.line, 165)
        self.assertEqual(parent.file_id, 'symbolic_tensor/optimizer/st_sgd.jsonl')
        self.assertEqual(parent.member_tag, '$alias_18')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$alias_18')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/optimizer/st_sgd.jsonl', '$ImportFrom_6')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$alias_18', child_members)

    def test_004_symbolic_tensor_function_st_moe_backward__Assign_85(self):
        """go_to_parent('symbolic_tensor/function/st_moe_backward.jsonl', '$Assign_85') -> '$FunctionDef_19' via 'FunctionDef__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe_backward.jsonl', '$Assign_85')
        self.assertEqual(parent.owner_tag, '$FunctionDef_19')
        self.assertEqual(parent.relation_tag, 'FunctionDef__body')
        self.assertEqual(parent.line, 456)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_moe_backward.jsonl')
        self.assertEqual(parent.member_tag, '$Assign_85')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Assign_85')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe_backward.jsonl', '$FunctionDef_19')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Assign_85', child_members)

    def test_005_symbolic_tensor_function_st_copy__ImportFrom_3(self):
        """go_to_parent('symbolic_tensor/function/st_copy.jsonl', '$ImportFrom_3') -> '$If_0' via 'If__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_copy.jsonl', '$ImportFrom_3')
        self.assertEqual(parent.owner_tag, '$If_0')
        self.assertEqual(parent.relation_tag, 'If__body')
        self.assertEqual(parent.line, 32)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_copy.jsonl')
        self.assertEqual(parent.member_tag, '$ImportFrom_3')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$ImportFrom_3')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_copy.jsonl', '$If_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$ImportFrom_3', child_members)

    def test_006_symbolic_tensor_tensor_util_patch_tensor__Import_2(self):
        """go_to_parent('symbolic_tensor/tensor_util/patch_tensor.jsonl', '$Import_2') -> '<module>' via 'Module__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/patch_tensor.jsonl', '$Import_2')
        self.assertEqual(parent.owner_tag, '<module>')
        self.assertEqual(parent.relation_tag, 'Module__body')
        self.assertEqual(parent.line, 0)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/patch_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$Import_2')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Import_2')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/patch_tensor.jsonl', '<module>')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Import_2', child_members)

    def test_007_symbolic_tensor_function_merge_forward__Assign_70(self):
        """go_to_parent('symbolic_tensor/function/merge_forward.jsonl', '$Assign_70') -> '$With_12' via 'With__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/merge_forward.jsonl', '$Assign_70')
        self.assertEqual(parent.owner_tag, '$With_12')
        self.assertEqual(parent.relation_tag, 'With__body')
        self.assertEqual(parent.line, 213)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/merge_forward.jsonl')
        self.assertEqual(parent.member_tag, '$Assign_70')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Assign_70')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/merge_forward.jsonl', '$With_12')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Assign_70', child_members)

    def test_008_symbolic_tensor_function_slice_attention_forward__Expr_60(self):
        """go_to_parent('symbolic_tensor/function/slice_attention_forward.jsonl', '$Expr_60') -> '$With_9' via 'With__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_attention_forward.jsonl', '$Expr_60')
        self.assertEqual(parent.owner_tag, '$With_9')
        self.assertEqual(parent.relation_tag, 'With__body')
        self.assertEqual(parent.line, 196)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/slice_attention_forward.jsonl')
        self.assertEqual(parent.member_tag, '$Expr_60')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Expr_60')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_attention_forward.jsonl', '$With_9')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Expr_60', child_members)

    def test_009_symbolic_tensor_module_st_moe__Assign_17(self):
        """go_to_parent('symbolic_tensor/module/st_moe.jsonl', '$Assign_17') -> '$If_0' via 'If__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/module/st_moe.jsonl', '$Assign_17')
        self.assertEqual(parent.owner_tag, '$If_0')
        self.assertEqual(parent.relation_tag, 'If__body')
        self.assertEqual(parent.line, 65)
        self.assertEqual(parent.file_id, 'symbolic_tensor/module/st_moe.jsonl')
        self.assertEqual(parent.member_tag, '$Assign_17')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Assign_17')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/module/st_moe.jsonl', '$If_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Assign_17', child_members)

    def test_010_symbolic_tensor_tensor_util_load_tensor__Call_41(self):
        """go_to_parent('symbolic_tensor/tensor_util/load_tensor.jsonl', '$Call_41') -> '$withitem_4' via 'withitem__context_expr'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/load_tensor.jsonl', '$Call_41')
        self.assertEqual(parent.owner_tag, '$withitem_4')
        self.assertEqual(parent.relation_tag, 'withitem__context_expr')
        self.assertEqual(parent.line, 0)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/load_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$Call_41')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Call_41')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/load_tensor.jsonl', '$withitem_4')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Call_41', child_members)

    def test_011_test_test_transform_method_time_comparison__Assign_5(self):
        """go_to_parent('test/test_transform_method_time_comparison.jsonl', '$Assign_5') -> '$With_0' via 'With__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'test/test_transform_method_time_comparison.jsonl', '$Assign_5')
        self.assertEqual(parent.owner_tag, '$With_0')
        self.assertEqual(parent.relation_tag, 'With__body')
        self.assertEqual(parent.line, 10)
        self.assertEqual(parent.file_id, 'test/test_transform_method_time_comparison.jsonl')
        self.assertEqual(parent.member_tag, '$Assign_5')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Assign_5')
        children = lexical_scope_expand_children(db, 'test/test_transform_method_time_comparison.jsonl', '$With_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Assign_5', child_members)

    def test_012_symbolic_tensor_tensor_util_slice_tensor__Return_1(self):
        """go_to_parent('symbolic_tensor/tensor_util/slice_tensor.jsonl', '$Return_1') -> '$If_0' via 'If__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/slice_tensor.jsonl', '$Return_1')
        self.assertEqual(parent.owner_tag, '$If_0')
        self.assertEqual(parent.relation_tag, 'If__body')
        self.assertEqual(parent.line, 24)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/slice_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$Return_1')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Return_1')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/slice_tensor.jsonl', '$If_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Return_1', child_members)

    def test_013_symbolic_tensor_function_slice_attention_forward__If_1(self):
        """go_to_parent('symbolic_tensor/function/slice_attention_forward.jsonl', '$If_1') -> '$For_0' via 'For__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_attention_forward.jsonl', '$If_1')
        self.assertEqual(parent.owner_tag, '$For_0')
        self.assertEqual(parent.relation_tag, 'For__body')
        self.assertEqual(parent.line, 47)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/slice_attention_forward.jsonl')
        self.assertEqual(parent.member_tag, '$If_1')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$If_1')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_attention_forward.jsonl', '$For_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$If_1', child_members)

    def test_014_symbolic_tensor_function_fork_tensor__Call_84(self):
        """go_to_parent('symbolic_tensor/function/fork_tensor.jsonl', '$Call_84') -> '$withitem_1' via 'withitem__context_expr'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/fork_tensor.jsonl', '$Call_84')
        self.assertEqual(parent.owner_tag, '$withitem_1')
        self.assertEqual(parent.relation_tag, 'withitem__context_expr')
        self.assertEqual(parent.line, 0)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/fork_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$Call_84')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Call_84')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/fork_tensor.jsonl', '$withitem_1')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Call_84', child_members)

    def test_015_symbolic_tensor_tensor_util_slice_tensor__Assign_28(self):
        """go_to_parent('symbolic_tensor/tensor_util/slice_tensor.jsonl', '$Assign_28') -> '$With_2' via 'With__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/slice_tensor.jsonl', '$Assign_28')
        self.assertEqual(parent.owner_tag, '$With_2')
        self.assertEqual(parent.relation_tag, 'With__body')
        self.assertEqual(parent.line, 123)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/slice_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$Assign_28')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Assign_28')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/slice_tensor.jsonl', '$With_2')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Assign_28', child_members)

    def test_016_symbolic_tensor_tensor_util_load_tensor__ImportFrom_0(self):
        """go_to_parent('symbolic_tensor/tensor_util/load_tensor.jsonl', '$ImportFrom_0') -> '<module>' via 'Module__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/load_tensor.jsonl', '$ImportFrom_0')
        self.assertEqual(parent.owner_tag, '<module>')
        self.assertEqual(parent.relation_tag, 'Module__body')
        self.assertEqual(parent.line, 0)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/load_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$ImportFrom_0')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$ImportFrom_0')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/load_tensor.jsonl', '<module>')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$ImportFrom_0', child_members)

    def test_017_symbolic_tensor_function_with_dense_view__Expr_22(self):
        """go_to_parent('symbolic_tensor/function/with_dense_view.jsonl', '$Expr_22') -> '$With_3' via 'With__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/with_dense_view.jsonl', '$Expr_22')
        self.assertEqual(parent.owner_tag, '$With_3')
        self.assertEqual(parent.relation_tag, 'With__body')
        self.assertEqual(parent.line, 152)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/with_dense_view.jsonl')
        self.assertEqual(parent.member_tag, '$Expr_22')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Expr_22')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/with_dense_view.jsonl', '$With_3')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Expr_22', child_members)

    def test_018_symbolic_tensor_tensor_util_assign_tensor__Assign_0(self):
        """go_to_parent('symbolic_tensor/tensor_util/assign_tensor.jsonl', '$Assign_0') -> '$FunctionDef_0' via 'FunctionDef__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/assign_tensor.jsonl', '$Assign_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef__body')
        self.assertEqual(parent.line, 6)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/assign_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$Assign_0')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Assign_0')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/assign_tensor.jsonl', '$FunctionDef_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Assign_0', child_members)

    def test_019_symbolic_tensor_tensor_util_sparse_to_dense__Assign_8(self):
        """go_to_parent('symbolic_tensor/tensor_util/sparse_to_dense.jsonl', '$Assign_8') -> '$FunctionDef_2' via 'FunctionDef__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', '$Assign_8')
        self.assertEqual(parent.owner_tag, '$FunctionDef_2')
        self.assertEqual(parent.relation_tag, 'FunctionDef__body')
        self.assertEqual(parent.line, 29)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl')
        self.assertEqual(parent.member_tag, '$Assign_8')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Assign_8')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', '$FunctionDef_2')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Assign_8', child_members)

    def test_020_symbolic_tensor_function_st_moe__Assign_16(self):
        """go_to_parent('symbolic_tensor/function/st_moe.jsonl', '$Assign_16') -> '$If_2' via 'If__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe.jsonl', '$Assign_16')
        self.assertEqual(parent.owner_tag, '$If_2')
        self.assertEqual(parent.relation_tag, 'If__body')
        self.assertEqual(parent.line, 68)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_moe.jsonl')
        self.assertEqual(parent.member_tag, '$Assign_16')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Assign_16')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe.jsonl', '$If_2')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Assign_16', child_members)

    def test_021_symbolic_tensor_tensor_util_assign_view__Assign_27(self):
        """go_to_parent('symbolic_tensor/tensor_util/assign_view.jsonl', '$Assign_27') -> '$With_7' via 'With__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/assign_view.jsonl', '$Assign_27')
        self.assertEqual(parent.owner_tag, '$With_7')
        self.assertEqual(parent.relation_tag, 'With__body')
        self.assertEqual(parent.line, 118)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/assign_view.jsonl')
        self.assertEqual(parent.member_tag, '$Assign_27')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Assign_27')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/assign_view.jsonl', '$With_7')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Assign_27', child_members)

    def test_022_symbolic_tensor_function_st_stack__Assign_5(self):
        """go_to_parent('symbolic_tensor/function/st_stack.jsonl', '$Assign_5') -> '$FunctionDef_4' via 'FunctionDef__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_stack.jsonl', '$Assign_5')
        self.assertEqual(parent.owner_tag, '$FunctionDef_4')
        self.assertEqual(parent.relation_tag, 'FunctionDef__body')
        self.assertEqual(parent.line, 34)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_stack.jsonl')
        self.assertEqual(parent.member_tag, '$Assign_5')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Assign_5')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_stack.jsonl', '$FunctionDef_4')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Assign_5', child_members)

    def test_023_symbolic_tensor_function_merge_backward__Assign_19(self):
        """go_to_parent('symbolic_tensor/function/merge_backward.jsonl', '$Assign_19') -> '$For_0' via 'For__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/merge_backward.jsonl', '$Assign_19')
        self.assertEqual(parent.owner_tag, '$For_0')
        self.assertEqual(parent.relation_tag, 'For__body')
        self.assertEqual(parent.line, 73)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/merge_backward.jsonl')
        self.assertEqual(parent.member_tag, '$Assign_19')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Assign_19')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/merge_backward.jsonl', '$For_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Assign_19', child_members)

    def test_024_symbolic_tensor_data_loader_sole_file_batch_data_loader__arguments_3(self):
        """go_to_parent('symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', '$arguments_3') -> '$FunctionDef_3' via 'FunctionDef__args'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', '$arguments_3')
        self.assertEqual(parent.owner_tag, '$FunctionDef_3')
        self.assertEqual(parent.relation_tag, 'FunctionDef__args')
        self.assertEqual(parent.line, 43)
        self.assertEqual(parent.file_id, 'symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl')
        self.assertEqual(parent.member_tag, '$arguments_3')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$arguments_3')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', '$FunctionDef_3')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$arguments_3', child_members)

    def test_025_symbolic_tensor_tensor_util_assign_tensor__Expr_26(self):
        """go_to_parent('symbolic_tensor/tensor_util/assign_tensor.jsonl', '$Expr_26') -> '$Try_0' via 'Try__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/assign_tensor.jsonl', '$Expr_26')
        self.assertEqual(parent.owner_tag, '$Try_0')
        self.assertEqual(parent.relation_tag, 'Try__body')
        self.assertEqual(parent.line, 91)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/assign_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$Expr_26')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Expr_26')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/assign_tensor.jsonl', '$Try_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Expr_26', child_members)

    def test_026_symbolic_tensor_function_st_stack__Call_283(self):
        """go_to_parent('symbolic_tensor/function/st_stack.jsonl', '$Call_283') -> '$withitem_14' via 'withitem__context_expr'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_stack.jsonl', '$Call_283')
        self.assertEqual(parent.owner_tag, '$withitem_14')
        self.assertEqual(parent.relation_tag, 'withitem__context_expr')
        self.assertEqual(parent.line, 0)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_stack.jsonl')
        self.assertEqual(parent.member_tag, '$Call_283')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Call_283')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_stack.jsonl', '$withitem_14')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Call_283', child_members)

    def test_027_symbolic_tensor_function_get_causal_attention_mask__Assign_8(self):
        """go_to_parent('symbolic_tensor/function/get_causal_attention_mask.jsonl', '$Assign_8') -> '$If_0' via 'If__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/get_causal_attention_mask.jsonl', '$Assign_8')
        self.assertEqual(parent.owner_tag, '$If_0')
        self.assertEqual(parent.relation_tag, 'If__body')
        self.assertEqual(parent.line, 18)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/get_causal_attention_mask.jsonl')
        self.assertEqual(parent.member_tag, '$Assign_8')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Assign_8')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/get_causal_attention_mask.jsonl', '$If_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Assign_8', child_members)

    def test_028_symbolic_tensor_tensor_util_make_none_tensor__ImportFrom_0(self):
        """go_to_parent('symbolic_tensor/tensor_util/make_none_tensor.jsonl', '$ImportFrom_0') -> '<module>' via 'Module__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', '$ImportFrom_0')
        self.assertEqual(parent.owner_tag, '<module>')
        self.assertEqual(parent.relation_tag, 'Module__body')
        self.assertEqual(parent.line, 0)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/make_none_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$ImportFrom_0')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$ImportFrom_0')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', '<module>')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$ImportFrom_0', child_members)

    def test_029_symbolic_tensor_function_st_moe_backward__Assign_132(self):
        """go_to_parent('symbolic_tensor/function/st_moe_backward.jsonl', '$Assign_132') -> '$With_5' via 'With__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe_backward.jsonl', '$Assign_132')
        self.assertEqual(parent.owner_tag, '$With_5')
        self.assertEqual(parent.relation_tag, 'With__body')
        self.assertEqual(parent.line, 659)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_moe_backward.jsonl')
        self.assertEqual(parent.member_tag, '$Assign_132')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Assign_132')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe_backward.jsonl', '$With_5')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Assign_132', child_members)

    def test_030_symbolic_tensor_function_st_attention__Assign_30(self):
        """go_to_parent('symbolic_tensor/function/st_attention.jsonl', '$Assign_30') -> '$With_4' via 'With__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_attention.jsonl', '$Assign_30')
        self.assertEqual(parent.owner_tag, '$With_4')
        self.assertEqual(parent.relation_tag, 'With__body')
        self.assertEqual(parent.line, 123)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_attention.jsonl')
        self.assertEqual(parent.member_tag, '$Assign_30')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Assign_30')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_attention.jsonl', '$With_4')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Assign_30', child_members)

    def test_031_symbolic_tensor_function_get_edit_distance_ratio__arguments_6(self):
        """go_to_parent('symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$arguments_6') -> '$FunctionDef_6' via 'FunctionDef__args'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$arguments_6')
        self.assertEqual(parent.owner_tag, '$FunctionDef_6')
        self.assertEqual(parent.relation_tag, 'FunctionDef__args')
        self.assertEqual(parent.line, 102)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl')
        self.assertEqual(parent.member_tag, '$arguments_6')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$arguments_6')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$FunctionDef_6')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$arguments_6', child_members)

    def test_032_symbolic_tensor_tensor_util_load_tensor__If_3(self):
        """go_to_parent('symbolic_tensor/tensor_util/load_tensor.jsonl', '$If_3') -> '$FunctionDef_5' via 'FunctionDef__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/load_tensor.jsonl', '$If_3')
        self.assertEqual(parent.owner_tag, '$FunctionDef_5')
        self.assertEqual(parent.relation_tag, 'FunctionDef__body')
        self.assertEqual(parent.line, 75)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/load_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$If_3')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$If_3')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/load_tensor.jsonl', '$FunctionDef_5')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$If_3', child_members)

    def test_033_symbolic_tensor_tensor_util_todo_tensor_like__With_4(self):
        """go_to_parent('symbolic_tensor/tensor_util/todo_tensor_like.jsonl', '$With_4') -> '$If_3' via 'If__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl', '$With_4')
        self.assertEqual(parent.owner_tag, '$If_3')
        self.assertEqual(parent.relation_tag, 'If__body')
        self.assertEqual(parent.line, 38)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl')
        self.assertEqual(parent.member_tag, '$With_4')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$With_4')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl', '$If_3')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$With_4', child_members)

    def test_034_symbolic_tensor_function_get_edit_distance_ratio__Assign_26(self):
        """go_to_parent('symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$Assign_26') -> '$FunctionDef_6' via 'FunctionDef__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$Assign_26')
        self.assertEqual(parent.owner_tag, '$FunctionDef_6')
        self.assertEqual(parent.relation_tag, 'FunctionDef__body')
        self.assertEqual(parent.line, 102)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl')
        self.assertEqual(parent.member_tag, '$Assign_26')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Assign_26')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$FunctionDef_6')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Assign_26', child_members)

    def test_035_symbolic_tensor_function_st_copy__arg_3(self):
        """go_to_parent('symbolic_tensor/function/st_copy.jsonl', '$arg_3') -> '$arguments_1' via 'arguments__args'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_copy.jsonl', '$arg_3')
        self.assertEqual(parent.owner_tag, '$arguments_1')
        self.assertEqual(parent.relation_tag, 'arguments__args')
        self.assertEqual(parent.line, 0)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_copy.jsonl')
        self.assertEqual(parent.member_tag, '$arg_3')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$arg_3')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_copy.jsonl', '$arguments_1')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$arg_3', child_members)

    def test_036_symbolic_tensor_function_select_qkv_indexes__alias_2(self):
        """go_to_parent('symbolic_tensor/function/select_qkv_indexes.jsonl', '$alias_2') -> '$ImportFrom_0' via 'ImportFrom__names'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/select_qkv_indexes.jsonl', '$alias_2')
        self.assertEqual(parent.owner_tag, '$ImportFrom_0')
        self.assertEqual(parent.relation_tag, 'ImportFrom__names')
        self.assertEqual(parent.line, 3)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/select_qkv_indexes.jsonl')
        self.assertEqual(parent.member_tag, '$alias_2')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$alias_2')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/select_qkv_indexes.jsonl', '$ImportFrom_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$alias_2', child_members)

    def test_037_symbolic_tensor_module_st_moe__arg_10(self):
        """go_to_parent('symbolic_tensor/module/st_moe.jsonl', '$arg_10') -> '$arguments_0' via 'arguments__args'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/module/st_moe.jsonl', '$arg_10')
        self.assertEqual(parent.owner_tag, '$arguments_0')
        self.assertEqual(parent.relation_tag, 'arguments__args')
        self.assertEqual(parent.line, 0)
        self.assertEqual(parent.file_id, 'symbolic_tensor/module/st_moe.jsonl')
        self.assertEqual(parent.member_tag, '$arg_10')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$arg_10')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/module/st_moe.jsonl', '$arguments_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$arg_10', child_members)

    def test_038_symbolic_tensor_function_select_qkv_indexes__For_2(self):
        """go_to_parent('symbolic_tensor/function/select_qkv_indexes.jsonl', '$For_2') -> '$FunctionDef_4' via 'FunctionDef__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/select_qkv_indexes.jsonl', '$For_2')
        self.assertEqual(parent.owner_tag, '$FunctionDef_4')
        self.assertEqual(parent.relation_tag, 'FunctionDef__body')
        self.assertEqual(parent.line, 56)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/select_qkv_indexes.jsonl')
        self.assertEqual(parent.member_tag, '$For_2')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$For_2')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/select_qkv_indexes.jsonl', '$FunctionDef_4')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$For_2', child_members)

    def test_039_symbolic_tensor_function_slice_view__Call_156(self):
        """go_to_parent('symbolic_tensor/function/slice_view.jsonl', '$Call_156') -> '$withitem_6' via 'withitem__context_expr'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_view.jsonl', '$Call_156')
        self.assertEqual(parent.owner_tag, '$withitem_6')
        self.assertEqual(parent.relation_tag, 'withitem__context_expr')
        self.assertEqual(parent.line, 0)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/slice_view.jsonl')
        self.assertEqual(parent.member_tag, '$Call_156')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Call_156')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_view.jsonl', '$withitem_6')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Call_156', child_members)

    def test_040_symbolic_tensor_tensor_util_slice_tensor__FunctionDef_0(self):
        """go_to_parent('symbolic_tensor/tensor_util/slice_tensor.jsonl', '$FunctionDef_0') -> '<module>' via 'Module__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/slice_tensor.jsonl', '$FunctionDef_0')
        self.assertEqual(parent.owner_tag, '<module>')
        self.assertEqual(parent.relation_tag, 'Module__body')
        self.assertEqual(parent.line, 0)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/slice_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$FunctionDef_0')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$FunctionDef_0')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/slice_tensor.jsonl', '<module>')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$FunctionDef_0', child_members)

    def test_041_test_test_attention_vs_traditional__Import_0(self):
        """go_to_parent('test/test_attention_vs_traditional.jsonl', '$Import_0') -> '<module>' via 'Module__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'test/test_attention_vs_traditional.jsonl', '$Import_0')
        self.assertEqual(parent.owner_tag, '<module>')
        self.assertEqual(parent.relation_tag, 'Module__body')
        self.assertEqual(parent.line, 0)
        self.assertEqual(parent.file_id, 'test/test_attention_vs_traditional.jsonl')
        self.assertEqual(parent.member_tag, '$Import_0')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Import_0')
        children = lexical_scope_expand_children(db, 'test/test_attention_vs_traditional.jsonl', '<module>')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Import_0', child_members)

    def test_042_symbolic_tensor_function_coding_agent__alias_8(self):
        """go_to_parent('symbolic_tensor/function/coding_agent.jsonl', '$alias_8') -> '$ImportFrom_0' via 'ImportFrom__names'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/coding_agent.jsonl', '$alias_8')
        self.assertEqual(parent.owner_tag, '$ImportFrom_0')
        self.assertEqual(parent.relation_tag, 'ImportFrom__names')
        self.assertEqual(parent.line, 6)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/coding_agent.jsonl')
        self.assertEqual(parent.member_tag, '$alias_8')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$alias_8')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/coding_agent.jsonl', '$ImportFrom_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$alias_8', child_members)

    def test_043_symbolic_tensor_function_st_moe__Assign_21(self):
        """go_to_parent('symbolic_tensor/function/st_moe.jsonl', '$Assign_21') -> '$If_5' via 'If__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe.jsonl', '$Assign_21')
        self.assertEqual(parent.owner_tag, '$If_5')
        self.assertEqual(parent.relation_tag, 'If__body')
        self.assertEqual(parent.line, 98)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_moe.jsonl')
        self.assertEqual(parent.member_tag, '$Assign_21')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Assign_21')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe.jsonl', '$If_5')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Assign_21', child_members)

    def test_044_symbolic_tensor_tensor_util_register_tensor_ops__Assign_12(self):
        """go_to_parent('symbolic_tensor/tensor_util/register_tensor_ops.jsonl', '$Assign_12') -> '$If_1' via 'If__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl', '$Assign_12')
        self.assertEqual(parent.owner_tag, '$If_1')
        self.assertEqual(parent.relation_tag, 'If__body')
        self.assertEqual(parent.line, 44)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl')
        self.assertEqual(parent.member_tag, '$Assign_12')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Assign_12')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl', '$If_1')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Assign_12', child_members)

    def test_045_symbolic_tensor_function_st_moe_backward__Expr_30(self):
        """go_to_parent('symbolic_tensor/function/st_moe_backward.jsonl', '$Expr_30') -> '$FunctionDef_18' via 'FunctionDef__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe_backward.jsonl', '$Expr_30')
        self.assertEqual(parent.owner_tag, '$FunctionDef_18')
        self.assertEqual(parent.relation_tag, 'FunctionDef__body')
        self.assertEqual(parent.line, 324)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_moe_backward.jsonl')
        self.assertEqual(parent.member_tag, '$Expr_30')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Expr_30')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe_backward.jsonl', '$FunctionDef_18')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Expr_30', child_members)

    def test_046_symbolic_tensor_function_get_edit_distance_ratio__FunctionDef_7(self):
        """go_to_parent('symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$FunctionDef_7') -> '$If_3' via 'If__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$FunctionDef_7')
        self.assertEqual(parent.owner_tag, '$If_3')
        self.assertEqual(parent.relation_tag, 'If__body')
        self.assertEqual(parent.line, 112)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl')
        self.assertEqual(parent.member_tag, '$FunctionDef_7')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$FunctionDef_7')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$If_3')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$FunctionDef_7', child_members)

    def test_047_symbolic_tensor_optimizer_st_sgd__Return_4(self):
        """go_to_parent('symbolic_tensor/optimizer/st_sgd.jsonl', '$Return_4') -> '$FunctionDef_4' via 'FunctionDef__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/optimizer/st_sgd.jsonl', '$Return_4')
        self.assertEqual(parent.owner_tag, '$FunctionDef_4')
        self.assertEqual(parent.relation_tag, 'FunctionDef__body')
        self.assertEqual(parent.line, 49)
        self.assertEqual(parent.file_id, 'symbolic_tensor/optimizer/st_sgd.jsonl')
        self.assertEqual(parent.member_tag, '$Return_4')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Return_4')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/optimizer/st_sgd.jsonl', '$FunctionDef_4')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Return_4', child_members)

    def test_048_symbolic_tensor_function_get_edit_distance_ratio__With_3(self):
        """go_to_parent('symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$With_3') -> '$If_3' via 'If__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$With_3')
        self.assertEqual(parent.owner_tag, '$If_3')
        self.assertEqual(parent.relation_tag, 'If__body')
        self.assertEqual(parent.line, 112)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl')
        self.assertEqual(parent.member_tag, '$With_3')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$With_3')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$If_3')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$With_3', child_members)

    def test_049_symbolic_tensor_function_merge__arg_13(self):
        """go_to_parent('symbolic_tensor/function/merge.jsonl', '$arg_13') -> '$arguments_5' via 'arguments__args'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/merge.jsonl', '$arg_13')
        self.assertEqual(parent.owner_tag, '$arguments_5')
        self.assertEqual(parent.relation_tag, 'arguments__args')
        self.assertEqual(parent.line, 0)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/merge.jsonl')
        self.assertEqual(parent.member_tag, '$arg_13')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$arg_13')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/merge.jsonl', '$arguments_5')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$arg_13', child_members)

    def test_050_symbolic_tensor_function_st_stack__Expr_45(self):
        """go_to_parent('symbolic_tensor/function/st_stack.jsonl', '$Expr_45') -> '$With_5' via 'With__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_stack.jsonl', '$Expr_45')
        self.assertEqual(parent.owner_tag, '$With_5')
        self.assertEqual(parent.relation_tag, 'With__body')
        self.assertEqual(parent.line, 220)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_stack.jsonl')
        self.assertEqual(parent.member_tag, '$Expr_45')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Expr_45')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_stack.jsonl', '$With_5')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Expr_45', child_members)

    def test_051_symbolic_tensor_function_st_moe_forward__Assign_33(self):
        """go_to_parent('symbolic_tensor/function/st_moe_forward.jsonl', '$Assign_33') -> '$If_6' via 'If__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe_forward.jsonl', '$Assign_33')
        self.assertEqual(parent.owner_tag, '$If_6')
        self.assertEqual(parent.relation_tag, 'If__body')
        self.assertEqual(parent.line, 201)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_moe_forward.jsonl')
        self.assertEqual(parent.member_tag, '$Assign_33')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Assign_33')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe_forward.jsonl', '$If_6')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Assign_33', child_members)

    def test_052_symbolic_tensor_tensor_util_dump_view__Return_0(self):
        """go_to_parent('symbolic_tensor/tensor_util/dump_view.jsonl', '$Return_0') -> '$FunctionDef_0' via 'FunctionDef__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/dump_view.jsonl', '$Return_0')
        self.assertEqual(parent.owner_tag, '$FunctionDef_0')
        self.assertEqual(parent.relation_tag, 'FunctionDef__body')
        self.assertEqual(parent.line, 5)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/dump_view.jsonl')
        self.assertEqual(parent.member_tag, '$Return_0')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Return_0')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/dump_view.jsonl', '$FunctionDef_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Return_0', child_members)

    def test_053_symbolic_tensor_function_st_stack__arguments_10(self):
        """go_to_parent('symbolic_tensor/function/st_stack.jsonl', '$arguments_10') -> '$FunctionDef_10' via 'FunctionDef__args'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_stack.jsonl', '$arguments_10')
        self.assertEqual(parent.owner_tag, '$FunctionDef_10')
        self.assertEqual(parent.relation_tag, 'FunctionDef__args')
        self.assertEqual(parent.line, 188)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_stack.jsonl')
        self.assertEqual(parent.member_tag, '$arguments_10')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$arguments_10')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_stack.jsonl', '$FunctionDef_10')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$arguments_10', child_members)

    def test_054_symbolic_tensor_function_st_moe__Expr_5(self):
        """go_to_parent('symbolic_tensor/function/st_moe.jsonl', '$Expr_5') -> '$If_4' via 'If__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe.jsonl', '$Expr_5')
        self.assertEqual(parent.owner_tag, '$If_4')
        self.assertEqual(parent.relation_tag, 'If__body')
        self.assertEqual(parent.line, 91)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_moe.jsonl')
        self.assertEqual(parent.member_tag, '$Expr_5')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Expr_5')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe.jsonl', '$If_4')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Expr_5', child_members)

    def test_055_symbolic_tensor_tensor_util_get_diff_tensor__Assign_11(self):
        """go_to_parent('symbolic_tensor/tensor_util/get_diff_tensor.jsonl', '$Assign_11') -> '$For_0' via 'For__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', '$Assign_11')
        self.assertEqual(parent.owner_tag, '$For_0')
        self.assertEqual(parent.relation_tag, 'For__body')
        self.assertEqual(parent.line, 50)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$Assign_11')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Assign_11')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', '$For_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Assign_11', child_members)

    def test_056_symbolic_tensor_function_st_moe_backward__Assign_23(self):
        """go_to_parent('symbolic_tensor/function/st_moe_backward.jsonl', '$Assign_23') -> '$If_13' via 'If__orelse'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe_backward.jsonl', '$Assign_23')
        self.assertEqual(parent.owner_tag, '$If_13')
        self.assertEqual(parent.relation_tag, 'If__orelse')
        self.assertEqual(parent.line, 165)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_moe_backward.jsonl')
        self.assertEqual(parent.member_tag, '$Assign_23')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Assign_23')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe_backward.jsonl', '$If_13')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Assign_23', child_members)

    def test_057_symbolic_tensor_optimizer_st_sgd__Assign_37(self):
        """go_to_parent('symbolic_tensor/optimizer/st_sgd.jsonl', '$Assign_37') -> '$FunctionDef_7' via 'FunctionDef__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/optimizer/st_sgd.jsonl', '$Assign_37')
        self.assertEqual(parent.owner_tag, '$FunctionDef_7')
        self.assertEqual(parent.relation_tag, 'FunctionDef__body')
        self.assertEqual(parent.line, 113)
        self.assertEqual(parent.file_id, 'symbolic_tensor/optimizer/st_sgd.jsonl')
        self.assertEqual(parent.member_tag, '$Assign_37')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Assign_37')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/optimizer/st_sgd.jsonl', '$FunctionDef_7')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Assign_37', child_members)

    def test_058_symbolic_tensor_tensor_util_dense_to_sparse__Assign_12(self):
        """go_to_parent('symbolic_tensor/tensor_util/dense_to_sparse.jsonl', '$Assign_12') -> '$With_1' via 'With__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', '$Assign_12')
        self.assertEqual(parent.owner_tag, '$With_1')
        self.assertEqual(parent.relation_tag, 'With__body')
        self.assertEqual(parent.line, 86)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl')
        self.assertEqual(parent.member_tag, '$Assign_12')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Assign_12')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', '$With_1')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Assign_12', child_members)

    def test_059_symbolic_tensor_function_st_attention__Expr_2(self):
        """go_to_parent('symbolic_tensor/function/st_attention.jsonl', '$Expr_2') -> '$If_1' via 'If__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_attention.jsonl', '$Expr_2')
        self.assertEqual(parent.owner_tag, '$If_1')
        self.assertEqual(parent.relation_tag, 'If__body')
        self.assertEqual(parent.line, 47)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_attention.jsonl')
        self.assertEqual(parent.member_tag, '$Expr_2')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Expr_2')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_attention.jsonl', '$If_1')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Expr_2', child_members)

    def test_060_llm_client_task_handler__alias_6(self):
        """go_to_parent('llm_client/task_handler.jsonl', '$alias_6') -> '$ImportFrom_3' via 'ImportFrom__names'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'llm_client/task_handler.jsonl', '$alias_6')
        self.assertEqual(parent.owner_tag, '$ImportFrom_3')
        self.assertEqual(parent.relation_tag, 'ImportFrom__names')
        self.assertEqual(parent.line, 6)
        self.assertEqual(parent.file_id, 'llm_client/task_handler.jsonl')
        self.assertEqual(parent.member_tag, '$alias_6')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$alias_6')
        children = lexical_scope_expand_children(db, 'llm_client/task_handler.jsonl', '$ImportFrom_3')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$alias_6', child_members)

    def test_061_symbolic_tensor_function_merge_backward__alias_8(self):
        """go_to_parent('symbolic_tensor/function/merge_backward.jsonl', '$alias_8') -> '$ImportFrom_2' via 'ImportFrom__names'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/merge_backward.jsonl', '$alias_8')
        self.assertEqual(parent.owner_tag, '$ImportFrom_2')
        self.assertEqual(parent.relation_tag, 'ImportFrom__names')
        self.assertEqual(parent.line, 8)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/merge_backward.jsonl')
        self.assertEqual(parent.member_tag, '$alias_8')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$alias_8')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/merge_backward.jsonl', '$ImportFrom_2')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$alias_8', child_members)

    def test_062_symbolic_tensor_function_get_edit_distance_ratio__Expr_15(self):
        """go_to_parent('symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$Expr_15') -> '$If_3' via 'If__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$Expr_15')
        self.assertEqual(parent.owner_tag, '$If_3')
        self.assertEqual(parent.relation_tag, 'If__body')
        self.assertEqual(parent.line, 112)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl')
        self.assertEqual(parent.member_tag, '$Expr_15')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Expr_15')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$If_3')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Expr_15', child_members)

    def test_063_symbolic_tensor_tensor_util_register_tensor_ops__FunctionDef_7(self):
        """go_to_parent('symbolic_tensor/tensor_util/register_tensor_ops.jsonl', '$FunctionDef_7') -> '$ClassDef_0' via 'ClassDef__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl', '$FunctionDef_7')
        self.assertEqual(parent.owner_tag, '$ClassDef_0')
        self.assertEqual(parent.relation_tag, 'ClassDef__body')
        self.assertEqual(parent.line, 30)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl')
        self.assertEqual(parent.member_tag, '$FunctionDef_7')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$FunctionDef_7')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl', '$ClassDef_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$FunctionDef_7', child_members)

    def test_064_symbolic_tensor_tensor_util_empty_tensor_like__For_1(self):
        """go_to_parent('symbolic_tensor/tensor_util/empty_tensor_like.jsonl', '$For_1') -> '$With_2' via 'With__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', '$For_1')
        self.assertEqual(parent.owner_tag, '$With_2')
        self.assertEqual(parent.relation_tag, 'With__body')
        self.assertEqual(parent.line, 44)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl')
        self.assertEqual(parent.member_tag, '$For_1')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$For_1')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', '$With_2')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$For_1', child_members)

    def test_065_symbolic_tensor_tensor_util_dump_view__arg_8(self):
        """go_to_parent('symbolic_tensor/tensor_util/dump_view.jsonl', '$arg_8') -> '$arguments_4' via 'arguments__args'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/dump_view.jsonl', '$arg_8')
        self.assertEqual(parent.owner_tag, '$arguments_4')
        self.assertEqual(parent.relation_tag, 'arguments__args')
        self.assertEqual(parent.line, 0)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/dump_view.jsonl')
        self.assertEqual(parent.member_tag, '$arg_8')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$arg_8')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/dump_view.jsonl', '$arguments_4')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$arg_8', child_members)

    def test_066_symbolic_tensor_tensor_util_dump_view__Expr_4(self):
        """go_to_parent('symbolic_tensor/tensor_util/dump_view.jsonl', '$Expr_4') -> '$For_0' via 'For__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/dump_view.jsonl', '$Expr_4')
        self.assertEqual(parent.owner_tag, '$For_0')
        self.assertEqual(parent.relation_tag, 'For__body')
        self.assertEqual(parent.line, 26)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/dump_view.jsonl')
        self.assertEqual(parent.member_tag, '$Expr_4')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Expr_4')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/dump_view.jsonl', '$For_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Expr_4', child_members)

    def test_067_symbolic_tensor_tensor_util_make_tensor__With_12(self):
        """go_to_parent('symbolic_tensor/tensor_util/make_tensor.jsonl', '$With_12') -> '$If_8' via 'If__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/make_tensor.jsonl', '$With_12')
        self.assertEqual(parent.owner_tag, '$If_8')
        self.assertEqual(parent.relation_tag, 'If__body')
        self.assertEqual(parent.line, 111)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/make_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$With_12')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$With_12')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/make_tensor.jsonl', '$If_8')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$With_12', child_members)

    def test_068_symbolic_tensor_function_slice_view__arg_13(self):
        """go_to_parent('symbolic_tensor/function/slice_view.jsonl', '$arg_13') -> '$arguments_5' via 'arguments__args'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_view.jsonl', '$arg_13')
        self.assertEqual(parent.owner_tag, '$arguments_5')
        self.assertEqual(parent.relation_tag, 'arguments__args')
        self.assertEqual(parent.line, 0)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/slice_view.jsonl')
        self.assertEqual(parent.member_tag, '$arg_13')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$arg_13')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_view.jsonl', '$arguments_5')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$arg_13', child_members)

    def test_069_symbolic_tensor_tensor_util_none_tensor_like__Expr_8(self):
        """go_to_parent('symbolic_tensor/tensor_util/none_tensor_like.jsonl', '$Expr_8') -> '$With_0' via 'With__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', '$Expr_8')
        self.assertEqual(parent.owner_tag, '$With_0')
        self.assertEqual(parent.relation_tag, 'With__body')
        self.assertEqual(parent.line, 28)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/none_tensor_like.jsonl')
        self.assertEqual(parent.member_tag, '$Expr_8')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Expr_8')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', '$With_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Expr_8', child_members)

    def test_070_symbolic_tensor_function_merge_backward__With_7(self):
        """go_to_parent('symbolic_tensor/function/merge_backward.jsonl', '$With_7') -> '$If_8' via 'If__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/merge_backward.jsonl', '$With_7')
        self.assertEqual(parent.owner_tag, '$If_8')
        self.assertEqual(parent.relation_tag, 'If__body')
        self.assertEqual(parent.line, 112)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/merge_backward.jsonl')
        self.assertEqual(parent.member_tag, '$With_7')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$With_7')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/merge_backward.jsonl', '$If_8')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$With_7', child_members)

    def test_071_symbolic_tensor_function_slice_attention_backward__arg_7(self):
        """go_to_parent('symbolic_tensor/function/slice_attention_backward.jsonl', '$arg_7') -> '$arguments_1' via 'arguments__args'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_attention_backward.jsonl', '$arg_7')
        self.assertEqual(parent.owner_tag, '$arguments_1')
        self.assertEqual(parent.relation_tag, 'arguments__args')
        self.assertEqual(parent.line, 0)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/slice_attention_backward.jsonl')
        self.assertEqual(parent.member_tag, '$arg_7')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$arg_7')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_attention_backward.jsonl', '$arguments_1')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$arg_7', child_members)

    def test_072_symbolic_tensor_function_get_causal_attention_mask__Assign_10(self):
        """go_to_parent('symbolic_tensor/function/get_causal_attention_mask.jsonl', '$Assign_10') -> '$If_0' via 'If__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/get_causal_attention_mask.jsonl', '$Assign_10')
        self.assertEqual(parent.owner_tag, '$If_0')
        self.assertEqual(parent.relation_tag, 'If__body')
        self.assertEqual(parent.line, 18)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/get_causal_attention_mask.jsonl')
        self.assertEqual(parent.member_tag, '$Assign_10')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Assign_10')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/get_causal_attention_mask.jsonl', '$If_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Assign_10', child_members)

    def test_073_symbolic_tensor_function_slice_tensor__Assign_24(self):
        """go_to_parent('symbolic_tensor/function/slice_tensor.jsonl', '$Assign_24') -> '$With_4' via 'With__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_tensor.jsonl', '$Assign_24')
        self.assertEqual(parent.owner_tag, '$With_4')
        self.assertEqual(parent.relation_tag, 'With__body')
        self.assertEqual(parent.line, 103)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/slice_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$Assign_24')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Assign_24')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_tensor.jsonl', '$With_4')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Assign_24', child_members)

    def test_074_symbolic_tensor_function_st_stack__alias_12(self):
        """go_to_parent('symbolic_tensor/function/st_stack.jsonl', '$alias_12') -> '$Import_3' via 'Import__names'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_stack.jsonl', '$alias_12')
        self.assertEqual(parent.owner_tag, '$Import_3')
        self.assertEqual(parent.relation_tag, 'Import__names')
        self.assertEqual(parent.line, 177)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_stack.jsonl')
        self.assertEqual(parent.member_tag, '$alias_12')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$alias_12')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_stack.jsonl', '$Import_3')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$alias_12', child_members)

    def test_075_symbolic_tensor_tensor_util_load_tensor__ImportFrom_1(self):
        """go_to_parent('symbolic_tensor/tensor_util/load_tensor.jsonl', '$ImportFrom_1') -> '<module>' via 'Module__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/load_tensor.jsonl', '$ImportFrom_1')
        self.assertEqual(parent.owner_tag, '<module>')
        self.assertEqual(parent.relation_tag, 'Module__body')
        self.assertEqual(parent.line, 0)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/load_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$ImportFrom_1')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$ImportFrom_1')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/load_tensor.jsonl', '<module>')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$ImportFrom_1', child_members)

    def test_076_symbolic_tensor_function_st_moe_backward__FunctionDef_4(self):
        """go_to_parent('symbolic_tensor/function/st_moe_backward.jsonl', '$FunctionDef_4') -> '<module>' via 'Module__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe_backward.jsonl', '$FunctionDef_4')
        self.assertEqual(parent.owner_tag, '<module>')
        self.assertEqual(parent.relation_tag, 'Module__body')
        self.assertEqual(parent.line, 0)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_moe_backward.jsonl')
        self.assertEqual(parent.member_tag, '$FunctionDef_4')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$FunctionDef_4')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe_backward.jsonl', '<module>')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$FunctionDef_4', child_members)

    def test_077_symbolic_tensor_function_get_edit_distance_ratio__Expr_7(self):
        """go_to_parent('symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$Expr_7') -> '$With_1' via 'With__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$Expr_7')
        self.assertEqual(parent.owner_tag, '$With_1')
        self.assertEqual(parent.relation_tag, 'With__body')
        self.assertEqual(parent.line, 22)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl')
        self.assertEqual(parent.member_tag, '$Expr_7')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Expr_7')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$With_1')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Expr_7', child_members)

    def test_078_symbolic_tensor_tensor_util_assign_tensor__Assign_11(self):
        """go_to_parent('symbolic_tensor/tensor_util/assign_tensor.jsonl', '$Assign_11') -> '$With_1' via 'With__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/assign_tensor.jsonl', '$Assign_11')
        self.assertEqual(parent.owner_tag, '$With_1')
        self.assertEqual(parent.relation_tag, 'With__body')
        self.assertEqual(parent.line, 64)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/assign_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$Assign_11')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Assign_11')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/assign_tensor.jsonl', '$With_1')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Assign_11', child_members)

    def test_079_symbolic_tensor_function_merge_backward__Assign_66(self):
        """go_to_parent('symbolic_tensor/function/merge_backward.jsonl', '$Assign_66') -> '$With_6' via 'With__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/merge_backward.jsonl', '$Assign_66')
        self.assertEqual(parent.owner_tag, '$With_6')
        self.assertEqual(parent.relation_tag, 'With__body')
        self.assertEqual(parent.line, 202)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/merge_backward.jsonl')
        self.assertEqual(parent.member_tag, '$Assign_66')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Assign_66')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/merge_backward.jsonl', '$With_6')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Assign_66', child_members)

    def test_080_llm_client_coding_agent_task_handler__alias_3(self):
        """go_to_parent('llm_client/coding_agent_task_handler.jsonl', '$alias_3') -> '$ImportFrom_0' via 'ImportFrom__names'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'llm_client/coding_agent_task_handler.jsonl', '$alias_3')
        self.assertEqual(parent.owner_tag, '$ImportFrom_0')
        self.assertEqual(parent.relation_tag, 'ImportFrom__names')
        self.assertEqual(parent.line, 3)
        self.assertEqual(parent.file_id, 'llm_client/coding_agent_task_handler.jsonl')
        self.assertEqual(parent.member_tag, '$alias_3')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$alias_3')
        children = lexical_scope_expand_children(db, 'llm_client/coding_agent_task_handler.jsonl', '$ImportFrom_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$alias_3', child_members)

    def test_081_symbolic_tensor_tensor_util_dump_tensor__Expr_2(self):
        """go_to_parent('symbolic_tensor/tensor_util/dump_tensor.jsonl', '$Expr_2') -> '$For_0' via 'For__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/dump_tensor.jsonl', '$Expr_2')
        self.assertEqual(parent.owner_tag, '$For_0')
        self.assertEqual(parent.relation_tag, 'For__body')
        self.assertEqual(parent.line, 20)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/dump_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$Expr_2')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Expr_2')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/dump_tensor.jsonl', '$For_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Expr_2', child_members)

    def test_082_symbolic_tensor_function_st_moe_forward__Subscript_2(self):
        """go_to_parent('symbolic_tensor/function/st_moe_forward.jsonl', '$Subscript_2') -> '$arg_1' via 'arg__annotation'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe_forward.jsonl', '$Subscript_2')
        self.assertEqual(parent.owner_tag, '$arg_1')
        self.assertEqual(parent.relation_tag, 'arg__annotation')
        self.assertEqual(parent.line, 20)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_moe_forward.jsonl')
        self.assertEqual(parent.member_tag, '$Subscript_2')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Subscript_2')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe_forward.jsonl', '$arg_1')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Subscript_2', child_members)

    def test_083_symbolic_tensor_function_st_moe__If_7(self):
        """go_to_parent('symbolic_tensor/function/st_moe.jsonl', '$If_7') -> '$If_6' via 'If__orelse'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe.jsonl', '$If_7')
        self.assertEqual(parent.owner_tag, '$If_6')
        self.assertEqual(parent.relation_tag, 'If__orelse')
        self.assertEqual(parent.line, 104)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_moe.jsonl')
        self.assertEqual(parent.member_tag, '$If_7')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$If_7')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe.jsonl', '$If_6')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$If_7', child_members)

    def test_084_symbolic_tensor_function_st_moe_backward__Expr_39(self):
        """go_to_parent('symbolic_tensor/function/st_moe_backward.jsonl', '$Expr_39') -> '$For_7' via 'For__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe_backward.jsonl', '$Expr_39')
        self.assertEqual(parent.owner_tag, '$For_7')
        self.assertEqual(parent.relation_tag, 'For__body')
        self.assertEqual(parent.line, 402)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_moe_backward.jsonl')
        self.assertEqual(parent.member_tag, '$Expr_39')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Expr_39')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe_backward.jsonl', '$For_7')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Expr_39', child_members)

    def test_085_symbolic_tensor_optimizer_st_sgd__For_4(self):
        """go_to_parent('symbolic_tensor/optimizer/st_sgd.jsonl', '$For_4') -> '$FunctionDef_7' via 'FunctionDef__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/optimizer/st_sgd.jsonl', '$For_4')
        self.assertEqual(parent.owner_tag, '$FunctionDef_7')
        self.assertEqual(parent.relation_tag, 'FunctionDef__body')
        self.assertEqual(parent.line, 113)
        self.assertEqual(parent.file_id, 'symbolic_tensor/optimizer/st_sgd.jsonl')
        self.assertEqual(parent.member_tag, '$For_4')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$For_4')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/optimizer/st_sgd.jsonl', '$FunctionDef_7')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$For_4', child_members)

    def test_086_symbolic_tensor_tensor_util_assign_tensor__Expr_10(self):
        """go_to_parent('symbolic_tensor/tensor_util/assign_tensor.jsonl', '$Expr_10') -> '$With_1' via 'With__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/assign_tensor.jsonl', '$Expr_10')
        self.assertEqual(parent.owner_tag, '$With_1')
        self.assertEqual(parent.relation_tag, 'With__body')
        self.assertEqual(parent.line, 64)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/assign_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$Expr_10')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Expr_10')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/assign_tensor.jsonl', '$With_1')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Expr_10', child_members)

    def test_087_symbolic_tensor_function_get_edit_distance_ratio__Assign_35(self):
        """go_to_parent('symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$Assign_35') -> '$With_4' via 'With__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$Assign_35')
        self.assertEqual(parent.owner_tag, '$With_4')
        self.assertEqual(parent.relation_tag, 'With__body')
        self.assertEqual(parent.line, 143)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl')
        self.assertEqual(parent.member_tag, '$Assign_35')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Assign_35')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$With_4')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Assign_35', child_members)

    def test_088_symbolic_tensor_tensor_util_slice_view__Expr_38(self):
        """go_to_parent('symbolic_tensor/tensor_util/slice_view.jsonl', '$Expr_38') -> '$With_17' via 'With__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/slice_view.jsonl', '$Expr_38')
        self.assertEqual(parent.owner_tag, '$With_17')
        self.assertEqual(parent.relation_tag, 'With__body')
        self.assertEqual(parent.line, 183)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/slice_view.jsonl')
        self.assertEqual(parent.member_tag, '$Expr_38')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Expr_38')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/slice_view.jsonl', '$With_17')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Expr_38', child_members)

    def test_089_symbolic_tensor_tensor_util_get_diff_tensor__Assign_31(self):
        """go_to_parent('symbolic_tensor/tensor_util/get_diff_tensor.jsonl', '$Assign_31') -> '$With_4' via 'With__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', '$Assign_31')
        self.assertEqual(parent.owner_tag, '$With_4')
        self.assertEqual(parent.relation_tag, 'With__body')
        self.assertEqual(parent.line, 132)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$Assign_31')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Assign_31')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', '$With_4')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Assign_31', child_members)

    def test_090_llm_client_raw_llm_task_handler__Assign_8(self):
        """go_to_parent('llm_client/raw_llm_task_handler.jsonl', '$Assign_8') -> '$For_3' via 'For__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'llm_client/raw_llm_task_handler.jsonl', '$Assign_8')
        self.assertEqual(parent.owner_tag, '$For_3')
        self.assertEqual(parent.relation_tag, 'For__body')
        self.assertEqual(parent.line, 32)
        self.assertEqual(parent.file_id, 'llm_client/raw_llm_task_handler.jsonl')
        self.assertEqual(parent.member_tag, '$Assign_8')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Assign_8')
        children = lexical_scope_expand_children(db, 'llm_client/raw_llm_task_handler.jsonl', '$For_3')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Assign_8', child_members)

    def test_091_symbolic_tensor_function_merge__withitem_2(self):
        """go_to_parent('symbolic_tensor/function/merge.jsonl', '$withitem_2') -> '$With_2' via 'With__items'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/merge.jsonl', '$withitem_2')
        self.assertEqual(parent.owner_tag, '$With_2')
        self.assertEqual(parent.relation_tag, 'With__items')
        self.assertEqual(parent.line, 94)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/merge.jsonl')
        self.assertEqual(parent.member_tag, '$withitem_2')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$withitem_2')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/merge.jsonl', '$With_2')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$withitem_2', child_members)

    def test_092_symbolic_tensor_function_st_moe_backward__alias_15(self):
        """go_to_parent('symbolic_tensor/function/st_moe_backward.jsonl', '$alias_15') -> '$ImportFrom_4' via 'ImportFrom__names'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe_backward.jsonl', '$alias_15')
        self.assertEqual(parent.owner_tag, '$ImportFrom_4')
        self.assertEqual(parent.relation_tag, 'ImportFrom__names')
        self.assertEqual(parent.line, 10)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_moe_backward.jsonl')
        self.assertEqual(parent.member_tag, '$alias_15')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$alias_15')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe_backward.jsonl', '$ImportFrom_4')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$alias_15', child_members)

    def test_093_symbolic_tensor_tensor_util_empty_tensor_like__Assign_4(self):
        """go_to_parent('symbolic_tensor/tensor_util/empty_tensor_like.jsonl', '$Assign_4') -> '$With_0' via 'With__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', '$Assign_4')
        self.assertEqual(parent.owner_tag, '$With_0')
        self.assertEqual(parent.relation_tag, 'With__body')
        self.assertEqual(parent.line, 34)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl')
        self.assertEqual(parent.member_tag, '$Assign_4')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Assign_4')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', '$With_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Assign_4', child_members)

    def test_094_symbolic_tensor_function_merge_forward__Assign_44(self):
        """go_to_parent('symbolic_tensor/function/merge_forward.jsonl', '$Assign_44') -> '$With_6' via 'With__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/merge_forward.jsonl', '$Assign_44')
        self.assertEqual(parent.owner_tag, '$With_6')
        self.assertEqual(parent.relation_tag, 'With__body')
        self.assertEqual(parent.line, 147)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/merge_forward.jsonl')
        self.assertEqual(parent.member_tag, '$Assign_44')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Assign_44')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/merge_forward.jsonl', '$With_6')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Assign_44', child_members)

    def test_095_symbolic_tensor_function_slice_attention__If_4(self):
        """go_to_parent('symbolic_tensor/function/slice_attention.jsonl', '$If_4') -> '<module>' via 'Module__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_attention.jsonl', '$If_4')
        self.assertEqual(parent.owner_tag, '<module>')
        self.assertEqual(parent.relation_tag, 'Module__body')
        self.assertEqual(parent.line, 0)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/slice_attention.jsonl')
        self.assertEqual(parent.member_tag, '$If_4')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$If_4')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_attention.jsonl', '<module>')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$If_4', child_members)

    def test_096_symbolic_tensor_function_select_qkv_indexes__Import_1(self):
        """go_to_parent('symbolic_tensor/function/select_qkv_indexes.jsonl', '$Import_1') -> '<module>' via 'Module__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/select_qkv_indexes.jsonl', '$Import_1')
        self.assertEqual(parent.owner_tag, '<module>')
        self.assertEqual(parent.relation_tag, 'Module__body')
        self.assertEqual(parent.line, 0)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/select_qkv_indexes.jsonl')
        self.assertEqual(parent.member_tag, '$Import_1')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Import_1')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/select_qkv_indexes.jsonl', '<module>')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Import_1', child_members)

    def test_097_symbolic_tensor_function_st_moe_backward__arg_31(self):
        """go_to_parent('symbolic_tensor/function/st_moe_backward.jsonl', '$arg_31') -> '$arguments_14' via 'arguments__args'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe_backward.jsonl', '$arg_31')
        self.assertEqual(parent.owner_tag, '$arguments_14')
        self.assertEqual(parent.relation_tag, 'arguments__args')
        self.assertEqual(parent.line, 0)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_moe_backward.jsonl')
        self.assertEqual(parent.member_tag, '$arg_31')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$arg_31')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe_backward.jsonl', '$arguments_14')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$arg_31', child_members)

    def test_098_symbolic_tensor_tensor_util_make_tensor__Assign_19(self):
        """go_to_parent('symbolic_tensor/tensor_util/make_tensor.jsonl', '$Assign_19') -> '$With_2' via 'With__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/make_tensor.jsonl', '$Assign_19')
        self.assertEqual(parent.owner_tag, '$With_2')
        self.assertEqual(parent.relation_tag, 'With__body')
        self.assertEqual(parent.line, 124)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/make_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$Assign_19')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Assign_19')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/make_tensor.jsonl', '$With_2')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Assign_19', child_members)

    def test_099_symbolic_tensor_tensor_util_patch_tensor__Import_4(self):
        """go_to_parent('symbolic_tensor/tensor_util/patch_tensor.jsonl', '$Import_4') -> '<module>' via 'Module__body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/patch_tensor.jsonl', '$Import_4')
        self.assertEqual(parent.owner_tag, '<module>')
        self.assertEqual(parent.relation_tag, 'Module__body')
        self.assertEqual(parent.line, 0)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/patch_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$Import_4')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$Import_4')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/patch_tensor.jsonl', '<module>')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$Import_4', child_members)


if __name__ == '__main__':
    unittest.main()
