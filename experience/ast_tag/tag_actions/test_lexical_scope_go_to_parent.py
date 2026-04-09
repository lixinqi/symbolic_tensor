"""
test_lexical_scope_go_to_parent: 100 concrete test cases.
Each case: (file_id, member_tag) -> expected (owner_tag, relation_tag, line).
Round-trip: expand_children on the parent includes the original member.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ast_tag_db import load_jsonl_dataset_into_ast_tag_db, AstTagDB
from relation_tag_classification import LEXICAL_RELATION_TAGS
from tag_actions.lexical_scope_go_to_parent import lexical_scope_go_to_parent
from tag_actions.lexical_scope_expand_children import lexical_scope_expand_children


def _db() -> AstTagDB:
    if not hasattr(_db, "_instance"):
        dataset_dir = os.path.join(os.path.dirname(__file__), "..", "test_dataset")
        _db._instance = load_jsonl_dataset_into_ast_tag_db(dataset_dir)
    return _db._instance


class TestLexicalScopeGoToParent(unittest.TestCase):

    def test_000_symbolic_tensor_function_merge_backward__if_11(self):
        """go_to_parent('symbolic_tensor/function/merge_backward.jsonl', '$if_11') -> '$functiondef_4' via 'contains'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/merge_backward.jsonl', '$if_11')
        self.assertEqual(parent.owner_tag, '$functiondef_4')
        self.assertEqual(parent.relation_tag, 'contains')
        self.assertEqual(parent.line, 124)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/merge_backward.jsonl')
        self.assertEqual(parent.member_tag, '$if_11')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$if_11')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/merge_backward.jsonl', '$functiondef_4')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$if_11', child_members)

    def test_001_symbolic_tensor_function_slice_tensor__expr_21(self):
        """go_to_parent('symbolic_tensor/function/slice_tensor.jsonl', '$expr_21') -> '$with_3' via 'with_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_tensor.jsonl', '$expr_21')
        self.assertEqual(parent.owner_tag, '$with_3')
        self.assertEqual(parent.relation_tag, 'with_body')
        self.assertEqual(parent.line, 83)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/slice_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$expr_21')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$expr_21')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_tensor.jsonl', '$with_3')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$expr_21', child_members)

    def test_002_symbolic_tensor_function_st_moe__assign_4(self):
        """go_to_parent('symbolic_tensor/function/st_moe.jsonl', '$assign_4') -> '$for_0' via 'for_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe.jsonl', '$assign_4')
        self.assertEqual(parent.owner_tag, '$for_0')
        self.assertEqual(parent.relation_tag, 'for_body')
        self.assertEqual(parent.line, 43)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_moe.jsonl')
        self.assertEqual(parent.member_tag, '$assign_4')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$assign_4')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe.jsonl', '$for_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$assign_4', child_members)

    def test_003_symbolic_tensor_tensor_util_sparse_to_dense__assign_28(self):
        """go_to_parent('symbolic_tensor/tensor_util/sparse_to_dense.jsonl', '$assign_28') -> '$functiondef_5' via 'contains'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', '$assign_28')
        self.assertEqual(parent.owner_tag, '$functiondef_5')
        self.assertEqual(parent.relation_tag, 'contains')
        self.assertEqual(parent.line, 74)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl')
        self.assertEqual(parent.member_tag, '$assign_28')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$assign_28')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', '$functiondef_5')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$assign_28', child_members)

    def test_004_symbolic_tensor_function_get_query_tensor__assign_27(self):
        """go_to_parent('symbolic_tensor/function/get_query_tensor.jsonl', '$assign_27') -> '$with_5' via 'with_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/get_query_tensor.jsonl', '$assign_27')
        self.assertEqual(parent.owner_tag, '$with_5')
        self.assertEqual(parent.relation_tag, 'with_body')
        self.assertEqual(parent.line, 131)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/get_query_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$assign_27')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$assign_27')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/get_query_tensor.jsonl', '$with_5')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$assign_27', child_members)

    def test_005_symbolic_tensor_tensor_util_slice_tensor__return_7(self):
        """go_to_parent('symbolic_tensor/tensor_util/slice_tensor.jsonl', '$return_7') -> '$functiondef_2' via 'contains'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/slice_tensor.jsonl', '$return_7')
        self.assertEqual(parent.owner_tag, '$functiondef_2')
        self.assertEqual(parent.relation_tag, 'contains')
        self.assertEqual(parent.line, 35)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/slice_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$return_7')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$return_7')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/slice_tensor.jsonl', '$functiondef_2')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$return_7', child_members)

    def test_006_symbolic_tensor_tensor_util_empty_tensor_like__assign_4(self):
        """go_to_parent('symbolic_tensor/tensor_util/empty_tensor_like.jsonl', '$assign_4') -> '$with_0' via 'with_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', '$assign_4')
        self.assertEqual(parent.owner_tag, '$with_0')
        self.assertEqual(parent.relation_tag, 'with_body')
        self.assertEqual(parent.line, 34)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl')
        self.assertEqual(parent.member_tag, '$assign_4')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$assign_4')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', '$with_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$assign_4', child_members)

    def test_007_symbolic_tensor_tensor_util_sparse_to_dense__call_51(self):
        """go_to_parent('symbolic_tensor/tensor_util/sparse_to_dense.jsonl', '$call_51') -> '$expr_11' via 'expr_stmt'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', '$call_51')
        self.assertEqual(parent.owner_tag, '$expr_11')
        self.assertEqual(parent.relation_tag, 'expr_stmt')
        self.assertEqual(parent.line, 132)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl')
        self.assertEqual(parent.member_tag, '$call_51')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$call_51')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', '$expr_11')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$call_51', child_members)

    def test_008_symbolic_tensor_tensor_util_dump_view__assign_14(self):
        """go_to_parent('symbolic_tensor/tensor_util/dump_view.jsonl', '$assign_14') -> '$with_2' via 'with_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/dump_view.jsonl', '$assign_14')
        self.assertEqual(parent.owner_tag, '$with_2')
        self.assertEqual(parent.relation_tag, 'with_body')
        self.assertEqual(parent.line, 68)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/dump_view.jsonl')
        self.assertEqual(parent.member_tag, '$assign_14')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$assign_14')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/dump_view.jsonl', '$with_2')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$assign_14', child_members)

    def test_009_symbolic_tensor_function_slice_attention_backward__if_9(self):
        """go_to_parent('symbolic_tensor/function/slice_attention_backward.jsonl', '$if_9') -> '$functiondef_4' via 'contains'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_attention_backward.jsonl', '$if_9')
        self.assertEqual(parent.owner_tag, '$functiondef_4')
        self.assertEqual(parent.relation_tag, 'contains')
        self.assertEqual(parent.line, 179)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/slice_attention_backward.jsonl')
        self.assertEqual(parent.member_tag, '$if_9')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$if_9')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_attention_backward.jsonl', '$functiondef_4')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$if_9', child_members)

    def test_010_sparse_util_convert_nested_list_coordinates_to_pairs_coordinates__docstring_5(self):
        """go_to_parent('sparse_util/convert_nested_list_coordinates_to_pairs_coordinates.jsonl', '$docstring_5') -> '$expr_4' via 'docstring'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'sparse_util/convert_nested_list_coordinates_to_pairs_coordinates.jsonl', '$docstring_5')
        self.assertEqual(parent.owner_tag, '$expr_4')
        self.assertEqual(parent.relation_tag, 'docstring')
        self.assertEqual(parent.line, 26)
        self.assertEqual(parent.file_id, 'sparse_util/convert_nested_list_coordinates_to_pairs_coordinates.jsonl')
        self.assertEqual(parent.member_tag, '$docstring_5')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$docstring_5')
        children = lexical_scope_expand_children(db, 'sparse_util/convert_nested_list_coordinates_to_pairs_coordinates.jsonl', '$expr_4')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$docstring_5', child_members)

    def test_011_symbolic_tensor_module_st_moe__assign_5(self):
        """go_to_parent('symbolic_tensor/module/st_moe.jsonl', '$assign_5') -> '$functiondef_0' via 'contains'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/module/st_moe.jsonl', '$assign_5')
        self.assertEqual(parent.owner_tag, '$functiondef_0')
        self.assertEqual(parent.relation_tag, 'contains')
        self.assertEqual(parent.line, 29)
        self.assertEqual(parent.file_id, 'symbolic_tensor/module/st_moe.jsonl')
        self.assertEqual(parent.member_tag, '$assign_5')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$assign_5')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/module/st_moe.jsonl', '$functiondef_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$assign_5', child_members)

    def test_012_llm_client_coding_agent_task_handler__assign_3(self):
        """go_to_parent('llm_client/coding_agent_task_handler.jsonl', '$assign_3') -> '$asyncfunctiondef_0' via 'contains'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'llm_client/coding_agent_task_handler.jsonl', '$assign_3')
        self.assertEqual(parent.owner_tag, '$asyncfunctiondef_0')
        self.assertEqual(parent.relation_tag, 'contains')
        self.assertEqual(parent.line, 17)
        self.assertEqual(parent.file_id, 'llm_client/coding_agent_task_handler.jsonl')
        self.assertEqual(parent.member_tag, '$assign_3')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$assign_3')
        children = lexical_scope_expand_children(db, 'llm_client/coding_agent_task_handler.jsonl', '$asyncfunctiondef_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$assign_3', child_members)

    def test_013_symbolic_tensor_function_st_attention__assign_2(self):
        """go_to_parent('symbolic_tensor/function/st_attention.jsonl', '$assign_2') -> '$functiondef_2' via 'contains'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_attention.jsonl', '$assign_2')
        self.assertEqual(parent.owner_tag, '$functiondef_2')
        self.assertEqual(parent.relation_tag, 'contains')
        self.assertEqual(parent.line, 54)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_attention.jsonl')
        self.assertEqual(parent.member_tag, '$assign_2')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$assign_2')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_attention.jsonl', '$functiondef_2')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$assign_2', child_members)

    def test_014_symbolic_tensor_tensor_util_patch_tensor__augassign_3(self):
        """go_to_parent('symbolic_tensor/tensor_util/patch_tensor.jsonl', '$augassign_3') -> '$if_4' via 'if_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/patch_tensor.jsonl', '$augassign_3')
        self.assertEqual(parent.owner_tag, '$if_4')
        self.assertEqual(parent.relation_tag, 'if_body')
        self.assertEqual(parent.line, 56)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/patch_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$augassign_3')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$augassign_3')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/patch_tensor.jsonl', '$if_4')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$augassign_3', child_members)

    def test_015_symbolic_tensor_function_st_moe__return_0(self):
        """go_to_parent('symbolic_tensor/function/st_moe.jsonl', '$return_0') -> '$functiondef_0' via 'contains'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe.jsonl', '$return_0')
        self.assertEqual(parent.owner_tag, '$functiondef_0')
        self.assertEqual(parent.relation_tag, 'contains')
        self.assertEqual(parent.line, 22)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_moe.jsonl')
        self.assertEqual(parent.member_tag, '$return_0')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$return_0')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe.jsonl', '$functiondef_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$return_0', child_members)

    def test_016_fs_util_text_merger__call_3(self):
        """go_to_parent('fs_util/text_merger.jsonl', '$call_3') -> '$comp_1' via 'comprehension_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'fs_util/text_merger.jsonl', '$call_3')
        self.assertEqual(parent.owner_tag, '$comp_1')
        self.assertEqual(parent.relation_tag, 'comprehension_body')
        self.assertEqual(parent.line, 20)
        self.assertEqual(parent.file_id, 'fs_util/text_merger.jsonl')
        self.assertEqual(parent.member_tag, '$call_3')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$call_3')
        children = lexical_scope_expand_children(db, 'fs_util/text_merger.jsonl', '$comp_1')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$call_3', child_members)

    def test_017_symbolic_tensor_data_loader_sole_file_batch_data_loader__call_35(self):
        """go_to_parent('symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', '$call_35') -> '$expr_15' via 'expr_stmt'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', '$call_35')
        self.assertEqual(parent.owner_tag, '$expr_15')
        self.assertEqual(parent.relation_tag, 'expr_stmt')
        self.assertEqual(parent.line, 81)
        self.assertEqual(parent.file_id, 'symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl')
        self.assertEqual(parent.member_tag, '$call_35')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$call_35')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', '$expr_15')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$call_35', child_members)

    def test_018_symbolic_tensor_tensor_util_make_tensor__expr_55(self):
        """go_to_parent('symbolic_tensor/tensor_util/make_tensor.jsonl', '$expr_55') -> '$with_15' via 'with_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/make_tensor.jsonl', '$expr_55')
        self.assertEqual(parent.owner_tag, '$with_15')
        self.assertEqual(parent.relation_tag, 'with_body')
        self.assertEqual(parent.line, 187)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/make_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$expr_55')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$expr_55')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/make_tensor.jsonl', '$with_15')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$expr_55', child_members)

    def test_019_llm_client_raw_llm_task_handler__docstring_1(self):
        """go_to_parent('llm_client/raw_llm_task_handler.jsonl', '$docstring_1') -> '$expr_0' via 'docstring'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'llm_client/raw_llm_task_handler.jsonl', '$docstring_1')
        self.assertEqual(parent.owner_tag, '$expr_0')
        self.assertEqual(parent.relation_tag, 'docstring')
        self.assertEqual(parent.line, 7)
        self.assertEqual(parent.file_id, 'llm_client/raw_llm_task_handler.jsonl')
        self.assertEqual(parent.member_tag, '$docstring_1')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$docstring_1')
        children = lexical_scope_expand_children(db, 'llm_client/raw_llm_task_handler.jsonl', '$expr_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$docstring_1', child_members)

    def test_020_symbolic_tensor_tensor_util_dump_tensor__with_1(self):
        """go_to_parent('symbolic_tensor/tensor_util/dump_tensor.jsonl', '$with_1') -> '$functiondef_1' via 'contains'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/dump_tensor.jsonl', '$with_1')
        self.assertEqual(parent.owner_tag, '$functiondef_1')
        self.assertEqual(parent.relation_tag, 'contains')
        self.assertEqual(parent.line, 9)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/dump_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$with_1')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$with_1')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/dump_tensor.jsonl', '$functiondef_1')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$with_1', child_members)

    def test_021_symbolic_tensor_module_st_moe__importfrom_4(self):
        """go_to_parent('symbolic_tensor/module/st_moe.jsonl', '$importfrom_4') -> '$if_0' via 'if_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/module/st_moe.jsonl', '$importfrom_4')
        self.assertEqual(parent.owner_tag, '$if_0')
        self.assertEqual(parent.relation_tag, 'if_body')
        self.assertEqual(parent.line, 65)
        self.assertEqual(parent.file_id, 'symbolic_tensor/module/st_moe.jsonl')
        self.assertEqual(parent.member_tag, '$importfrom_4')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$importfrom_4')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/module/st_moe.jsonl', '$if_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$importfrom_4', child_members)

    def test_022_symbolic_tensor_tensor_util_pack_tensor__call_26(self):
        """go_to_parent('symbolic_tensor/tensor_util/pack_tensor.jsonl', '$call_26') -> '$expr_16' via 'expr_stmt'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/pack_tensor.jsonl', '$call_26')
        self.assertEqual(parent.owner_tag, '$expr_16')
        self.assertEqual(parent.relation_tag, 'expr_stmt')
        self.assertEqual(parent.line, 46)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/pack_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$call_26')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$call_26')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/pack_tensor.jsonl', '$expr_16')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$call_26', child_members)

    def test_023_symbolic_tensor_function_slice_and_concat_attention_forward__assign_43(self):
        """go_to_parent('symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', '$assign_43') -> '$with_7' via 'with_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', '$assign_43')
        self.assertEqual(parent.owner_tag, '$with_7')
        self.assertEqual(parent.relation_tag, 'with_body')
        self.assertEqual(parent.line, 139)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl')
        self.assertEqual(parent.member_tag, '$assign_43')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$assign_43')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', '$with_7')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$assign_43', child_members)

    def test_024_symbolic_tensor_tensor_util_todo_tensor_like__return_1(self):
        """go_to_parent('symbolic_tensor/tensor_util/todo_tensor_like.jsonl', '$return_1') -> '$functiondef_0' via 'contains'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl', '$return_1')
        self.assertEqual(parent.owner_tag, '$functiondef_0')
        self.assertEqual(parent.relation_tag, 'contains')
        self.assertEqual(parent.line, 6)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl')
        self.assertEqual(parent.member_tag, '$return_1')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$return_1')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl', '$functiondef_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$return_1', child_members)

    def test_025_symbolic_tensor_tensor_util_sparse_to_dense__expr_42(self):
        """go_to_parent('symbolic_tensor/tensor_util/sparse_to_dense.jsonl', '$expr_42') -> '$with_5' via 'with_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', '$expr_42')
        self.assertEqual(parent.owner_tag, '$with_5')
        self.assertEqual(parent.relation_tag, 'with_body')
        self.assertEqual(parent.line, 190)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl')
        self.assertEqual(parent.member_tag, '$expr_42')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$expr_42')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', '$with_5')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$expr_42', child_members)

    def test_026_sparse_util_group_random_select__assign_2(self):
        """go_to_parent('sparse_util/group_random_select.jsonl', '$assign_2') -> '$functiondef_0' via 'contains'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'sparse_util/group_random_select.jsonl', '$assign_2')
        self.assertEqual(parent.owner_tag, '$functiondef_0')
        self.assertEqual(parent.relation_tag, 'contains')
        self.assertEqual(parent.line, 2)
        self.assertEqual(parent.file_id, 'sparse_util/group_random_select.jsonl')
        self.assertEqual(parent.member_tag, '$assign_2')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$assign_2')
        children = lexical_scope_expand_children(db, 'sparse_util/group_random_select.jsonl', '$functiondef_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$assign_2', child_members)

    def test_027_symbolic_tensor_tensor_util_empty_tensor_like__assign_3(self):
        """go_to_parent('symbolic_tensor/tensor_util/empty_tensor_like.jsonl', '$assign_3') -> '$with_0' via 'with_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', '$assign_3')
        self.assertEqual(parent.owner_tag, '$with_0')
        self.assertEqual(parent.relation_tag, 'with_body')
        self.assertEqual(parent.line, 34)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl')
        self.assertEqual(parent.member_tag, '$assign_3')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$assign_3')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', '$with_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$assign_3', child_members)

    def test_028_symbolic_tensor_tensor_util_make_none_tensor__expr_0(self):
        """go_to_parent('symbolic_tensor/tensor_util/make_none_tensor.jsonl', '$expr_0') -> '$functiondef_0' via 'contains'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', '$expr_0')
        self.assertEqual(parent.owner_tag, '$functiondef_0')
        self.assertEqual(parent.relation_tag, 'contains')
        self.assertEqual(parent.line, 4)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/make_none_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$expr_0')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$expr_0')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', '$functiondef_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$expr_0', child_members)

    def test_029_symbolic_tensor_tensor_util_none_tensor_like__expr_11(self):
        """go_to_parent('symbolic_tensor/tensor_util/none_tensor_like.jsonl', '$expr_11') -> '$with_0' via 'with_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', '$expr_11')
        self.assertEqual(parent.owner_tag, '$with_0')
        self.assertEqual(parent.relation_tag, 'with_body')
        self.assertEqual(parent.line, 28)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/none_tensor_like.jsonl')
        self.assertEqual(parent.member_tag, '$expr_11')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$expr_11')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', '$with_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$expr_11', child_members)

    def test_030_fs_util_text_merger__if_0(self):
        """go_to_parent('fs_util/text_merger.jsonl', '$if_0') -> '$functiondef_2' via 'contains'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'fs_util/text_merger.jsonl', '$if_0')
        self.assertEqual(parent.owner_tag, '$functiondef_2')
        self.assertEqual(parent.relation_tag, 'contains')
        self.assertEqual(parent.line, 21)
        self.assertEqual(parent.file_id, 'fs_util/text_merger.jsonl')
        self.assertEqual(parent.member_tag, '$if_0')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$if_0')
        children = lexical_scope_expand_children(db, 'fs_util/text_merger.jsonl', '$functiondef_2')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$if_0', child_members)

    def test_031_symbolic_tensor_tensor_util_load_tensor__call_100(self):
        """go_to_parent('symbolic_tensor/tensor_util/load_tensor.jsonl', '$call_100') -> '$with_8' via 'with_context'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/load_tensor.jsonl', '$call_100')
        self.assertEqual(parent.owner_tag, '$with_8')
        self.assertEqual(parent.relation_tag, 'with_context')
        self.assertEqual(parent.line, 128)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/load_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$call_100')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$call_100')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/load_tensor.jsonl', '$with_8')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$call_100', child_members)

    def test_032_symbolic_tensor_tensor_util_dump_tensor__call_26(self):
        """go_to_parent('symbolic_tensor/tensor_util/dump_tensor.jsonl', '$call_26') -> '$expr_8' via 'expr_stmt'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/dump_tensor.jsonl', '$call_26')
        self.assertEqual(parent.owner_tag, '$expr_8')
        self.assertEqual(parent.relation_tag, 'expr_stmt')
        self.assertEqual(parent.line, 41)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/dump_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$call_26')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$call_26')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/dump_tensor.jsonl', '$expr_8')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$call_26', child_members)

    def test_033_llm_client_coding_agent_task_handler__import_1(self):
        """go_to_parent('llm_client/coding_agent_task_handler.jsonl', '$import_1') -> '<module>' via 'contains'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'llm_client/coding_agent_task_handler.jsonl', '$import_1')
        self.assertEqual(parent.owner_tag, '<module>')
        self.assertEqual(parent.relation_tag, 'contains')
        self.assertEqual(parent.line, 0)
        self.assertEqual(parent.file_id, 'llm_client/coding_agent_task_handler.jsonl')
        self.assertEqual(parent.member_tag, '$import_1')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$import_1')
        children = lexical_scope_expand_children(db, 'llm_client/coding_agent_task_handler.jsonl', '<module>')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$import_1', child_members)

    def test_034_symbolic_tensor_module_with_dense_view__import_1(self):
        """go_to_parent('symbolic_tensor/module/with_dense_view.jsonl', '$import_1') -> '<module>' via 'contains'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/module/with_dense_view.jsonl', '$import_1')
        self.assertEqual(parent.owner_tag, '<module>')
        self.assertEqual(parent.relation_tag, 'contains')
        self.assertEqual(parent.line, 0)
        self.assertEqual(parent.file_id, 'symbolic_tensor/module/with_dense_view.jsonl')
        self.assertEqual(parent.member_tag, '$import_1')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$import_1')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/module/with_dense_view.jsonl', '<module>')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$import_1', child_members)

    def test_035_symbolic_tensor_tensor_util_dump_view__if_3(self):
        """go_to_parent('symbolic_tensor/tensor_util/dump_view.jsonl', '$if_3') -> '$if_2' via 'else_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/dump_view.jsonl', '$if_3')
        self.assertEqual(parent.owner_tag, '$if_2')
        self.assertEqual(parent.relation_tag, 'else_body')
        self.assertEqual(parent.line, 49)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/dump_view.jsonl')
        self.assertEqual(parent.member_tag, '$if_3')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$if_3')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/dump_view.jsonl', '$if_2')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$if_3', child_members)

    def test_036_symbolic_tensor_function_slice_attention_forward__import_3(self):
        """go_to_parent('symbolic_tensor/function/slice_attention_forward.jsonl', '$import_3') -> '$if_3' via 'if_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_attention_forward.jsonl', '$import_3')
        self.assertEqual(parent.owner_tag, '$if_3')
        self.assertEqual(parent.relation_tag, 'if_body')
        self.assertEqual(parent.line, 75)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/slice_attention_forward.jsonl')
        self.assertEqual(parent.member_tag, '$import_3')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$import_3')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_attention_forward.jsonl', '$if_3')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$import_3', child_members)

    def test_037_symbolic_tensor_data_loader_sole_file_batch_data_loader__functiondef_3(self):
        """go_to_parent('symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', '$functiondef_3') -> '$classdef_0' via 'contains'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', '$functiondef_3')
        self.assertEqual(parent.owner_tag, '$classdef_0')
        self.assertEqual(parent.relation_tag, 'contains')
        self.assertEqual(parent.line, 16)
        self.assertEqual(parent.file_id, 'symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl')
        self.assertEqual(parent.member_tag, '$functiondef_3')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$functiondef_3')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', '$classdef_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$functiondef_3', child_members)

    def test_038_fs_util_text_merger__if_3(self):
        """go_to_parent('fs_util/text_merger.jsonl', '$if_3') -> '$if_2' via 'if_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'fs_util/text_merger.jsonl', '$if_3')
        self.assertEqual(parent.owner_tag, '$if_2')
        self.assertEqual(parent.relation_tag, 'if_body')
        self.assertEqual(parent.line, 40)
        self.assertEqual(parent.file_id, 'fs_util/text_merger.jsonl')
        self.assertEqual(parent.member_tag, '$if_3')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$if_3')
        children = lexical_scope_expand_children(db, 'fs_util/text_merger.jsonl', '$if_2')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$if_3', child_members)

    def test_039_symbolic_tensor_optimizer_st_sgd__expr_19(self):
        """go_to_parent('symbolic_tensor/optimizer/st_sgd.jsonl', '$expr_19') -> '$if_17' via 'if_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/optimizer/st_sgd.jsonl', '$expr_19')
        self.assertEqual(parent.owner_tag, '$if_17')
        self.assertEqual(parent.relation_tag, 'if_body')
        self.assertEqual(parent.line, 159)
        self.assertEqual(parent.file_id, 'symbolic_tensor/optimizer/st_sgd.jsonl')
        self.assertEqual(parent.member_tag, '$expr_19')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$expr_19')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/optimizer/st_sgd.jsonl', '$if_17')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$expr_19', child_members)

    def test_040_symbolic_tensor_function_slice_attention_forward__call_65(self):
        """go_to_parent('symbolic_tensor/function/slice_attention_forward.jsonl', '$call_65') -> '$expr_18' via 'expr_stmt'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_attention_forward.jsonl', '$call_65')
        self.assertEqual(parent.owner_tag, '$expr_18')
        self.assertEqual(parent.relation_tag, 'expr_stmt')
        self.assertEqual(parent.line, 115)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/slice_attention_forward.jsonl')
        self.assertEqual(parent.member_tag, '$call_65')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$call_65')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_attention_forward.jsonl', '$expr_18')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$call_65', child_members)

    def test_041_symbolic_tensor_function_slice_attention__call_39(self):
        """go_to_parent('symbolic_tensor/function/slice_attention.jsonl', '$call_39') -> '$expr_15' via 'expr_stmt'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_attention.jsonl', '$call_39')
        self.assertEqual(parent.owner_tag, '$expr_15')
        self.assertEqual(parent.relation_tag, 'expr_stmt')
        self.assertEqual(parent.line, 97)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/slice_attention.jsonl')
        self.assertEqual(parent.member_tag, '$call_39')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$call_39')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_attention.jsonl', '$expr_15')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$call_39', child_members)

    def test_042_symbolic_tensor_function_st_stack__call_250(self):
        """go_to_parent('symbolic_tensor/function/st_stack.jsonl', '$call_250') -> '$expr_83' via 'expr_stmt'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_stack.jsonl', '$call_250')
        self.assertEqual(parent.owner_tag, '$expr_83')
        self.assertEqual(parent.relation_tag, 'expr_stmt')
        self.assertEqual(parent.line, 304)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_stack.jsonl')
        self.assertEqual(parent.member_tag, '$call_250')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$call_250')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_stack.jsonl', '$expr_83')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$call_250', child_members)

    def test_043_llm_client_coding_agent_query__import_1(self):
        """go_to_parent('llm_client/coding_agent_query.jsonl', '$import_1') -> '<module>' via 'contains'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'llm_client/coding_agent_query.jsonl', '$import_1')
        self.assertEqual(parent.owner_tag, '<module>')
        self.assertEqual(parent.relation_tag, 'contains')
        self.assertEqual(parent.line, 0)
        self.assertEqual(parent.file_id, 'llm_client/coding_agent_query.jsonl')
        self.assertEqual(parent.member_tag, '$import_1')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$import_1')
        children = lexical_scope_expand_children(db, 'llm_client/coding_agent_query.jsonl', '<module>')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$import_1', child_members)

    def test_044_symbolic_tensor_function_st_attention__assign_29(self):
        """go_to_parent('symbolic_tensor/function/st_attention.jsonl', '$assign_29') -> '$with_4' via 'with_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_attention.jsonl', '$assign_29')
        self.assertEqual(parent.owner_tag, '$with_4')
        self.assertEqual(parent.relation_tag, 'with_body')
        self.assertEqual(parent.line, 123)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_attention.jsonl')
        self.assertEqual(parent.member_tag, '$assign_29')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$assign_29')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_attention.jsonl', '$with_4')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$assign_29', child_members)

    def test_045_symbolic_tensor_tensor_util_dump_view__expr_18(self):
        """go_to_parent('symbolic_tensor/tensor_util/dump_view.jsonl', '$expr_18') -> '$for_2' via 'for_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/dump_view.jsonl', '$expr_18')
        self.assertEqual(parent.owner_tag, '$for_2')
        self.assertEqual(parent.relation_tag, 'for_body')
        self.assertEqual(parent.line, 74)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/dump_view.jsonl')
        self.assertEqual(parent.member_tag, '$expr_18')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$expr_18')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/dump_view.jsonl', '$for_2')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$expr_18', child_members)

    def test_046_sparse_util_group_random_select__assign_0(self):
        """go_to_parent('sparse_util/group_random_select.jsonl', '$assign_0') -> '$functiondef_0' via 'contains'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'sparse_util/group_random_select.jsonl', '$assign_0')
        self.assertEqual(parent.owner_tag, '$functiondef_0')
        self.assertEqual(parent.relation_tag, 'contains')
        self.assertEqual(parent.line, 2)
        self.assertEqual(parent.file_id, 'sparse_util/group_random_select.jsonl')
        self.assertEqual(parent.member_tag, '$assign_0')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$assign_0')
        children = lexical_scope_expand_children(db, 'sparse_util/group_random_select.jsonl', '$functiondef_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$assign_0', child_members)

    def test_047_symbolic_tensor_function_symbolic_grad_registry__assign_1(self):
        """go_to_parent('symbolic_tensor/function/symbolic_grad_registry.jsonl', '$assign_1') -> '$if_0' via 'if_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/symbolic_grad_registry.jsonl', '$assign_1')
        self.assertEqual(parent.owner_tag, '$if_0')
        self.assertEqual(parent.relation_tag, 'if_body')
        self.assertEqual(parent.line, 18)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/symbolic_grad_registry.jsonl')
        self.assertEqual(parent.member_tag, '$assign_1')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$assign_1')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/symbolic_grad_registry.jsonl', '$if_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$assign_1', child_members)

    def test_048_llm_client_task_handler__subscript_0(self):
        """go_to_parent('llm_client/task_handler.jsonl', '$subscript_0') -> '$functiondef_0' via 'param_annotation'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'llm_client/task_handler.jsonl', '$subscript_0')
        self.assertEqual(parent.owner_tag, '$functiondef_0')
        self.assertEqual(parent.relation_tag, 'param_annotation')
        self.assertEqual(parent.line, 8)
        self.assertEqual(parent.file_id, 'llm_client/task_handler.jsonl')
        self.assertEqual(parent.member_tag, '$subscript_0')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$subscript_0')
        children = lexical_scope_expand_children(db, 'llm_client/task_handler.jsonl', '$functiondef_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$subscript_0', child_members)

    def test_049_llm_client_raw_llm_task_handler__assign_6(self):
        """go_to_parent('llm_client/raw_llm_task_handler.jsonl', '$assign_6') -> '$for_3' via 'for_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'llm_client/raw_llm_task_handler.jsonl', '$assign_6')
        self.assertEqual(parent.owner_tag, '$for_3')
        self.assertEqual(parent.relation_tag, 'for_body')
        self.assertEqual(parent.line, 32)
        self.assertEqual(parent.file_id, 'llm_client/raw_llm_task_handler.jsonl')
        self.assertEqual(parent.member_tag, '$assign_6')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$assign_6')
        children = lexical_scope_expand_children(db, 'llm_client/raw_llm_task_handler.jsonl', '$for_3')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$assign_6', child_members)

    def test_050_test_test_attention_vs_traditional__return_2(self):
        """go_to_parent('test/test_attention_vs_traditional.jsonl', '$return_2') -> '$if_1' via 'if_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'test/test_attention_vs_traditional.jsonl', '$return_2')
        self.assertEqual(parent.owner_tag, '$if_1')
        self.assertEqual(parent.relation_tag, 'if_body')
        self.assertEqual(parent.line, 32)
        self.assertEqual(parent.file_id, 'test/test_attention_vs_traditional.jsonl')
        self.assertEqual(parent.member_tag, '$return_2')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$return_2')
        children = lexical_scope_expand_children(db, 'test/test_attention_vs_traditional.jsonl', '$if_1')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$return_2', child_members)

    def test_051_symbolic_tensor_function_st_moe_backward__call_319(self):
        """go_to_parent('symbolic_tensor/function/st_moe_backward.jsonl', '$call_319') -> '$expr_107' via 'expr_stmt'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe_backward.jsonl', '$call_319')
        self.assertEqual(parent.owner_tag, '$expr_107')
        self.assertEqual(parent.relation_tag, 'expr_stmt')
        self.assertEqual(parent.line, 720)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_moe_backward.jsonl')
        self.assertEqual(parent.member_tag, '$call_319')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$call_319')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe_backward.jsonl', '$expr_107')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$call_319', child_members)

    def test_052_symbolic_tensor_tensor_util_dense_to_sparse__assign_4(self):
        """go_to_parent('symbolic_tensor/tensor_util/dense_to_sparse.jsonl', '$assign_4') -> '$if_0' via 'if_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', '$assign_4')
        self.assertEqual(parent.owner_tag, '$if_0')
        self.assertEqual(parent.relation_tag, 'if_body')
        self.assertEqual(parent.line, 31)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl')
        self.assertEqual(parent.member_tag, '$assign_4')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$assign_4')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', '$if_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$assign_4', child_members)

    def test_053_symbolic_tensor_tensor_util_make_none_tensor__docstring_1(self):
        """go_to_parent('symbolic_tensor/tensor_util/make_none_tensor.jsonl', '$docstring_1') -> '$expr_0' via 'docstring'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', '$docstring_1')
        self.assertEqual(parent.owner_tag, '$expr_0')
        self.assertEqual(parent.relation_tag, 'docstring')
        self.assertEqual(parent.line, 5)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/make_none_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$docstring_1')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$docstring_1')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', '$expr_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$docstring_1', child_members)

    def test_054_symbolic_tensor_function_merge_forward__call_86(self):
        """go_to_parent('symbolic_tensor/function/merge_forward.jsonl', '$call_86') -> '$expr_30' via 'expr_stmt'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/merge_forward.jsonl', '$call_86')
        self.assertEqual(parent.owner_tag, '$expr_30')
        self.assertEqual(parent.relation_tag, 'expr_stmt')
        self.assertEqual(parent.line, 143)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/merge_forward.jsonl')
        self.assertEqual(parent.member_tag, '$call_86')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$call_86')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/merge_forward.jsonl', '$expr_30')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$call_86', child_members)

    def test_055_symbolic_tensor_tensor_util_pack_tensor__with_1(self):
        """go_to_parent('symbolic_tensor/tensor_util/pack_tensor.jsonl', '$with_1') -> '$if_0' via 'if_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/pack_tensor.jsonl', '$with_1')
        self.assertEqual(parent.owner_tag, '$if_0')
        self.assertEqual(parent.relation_tag, 'if_body')
        self.assertEqual(parent.line, 15)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/pack_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$with_1')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$with_1')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/pack_tensor.jsonl', '$if_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$with_1', child_members)

    def test_056_symbolic_tensor_module_st_moe__assign_8(self):
        """go_to_parent('symbolic_tensor/module/st_moe.jsonl', '$assign_8') -> '$functiondef_0' via 'contains'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/module/st_moe.jsonl', '$assign_8')
        self.assertEqual(parent.owner_tag, '$functiondef_0')
        self.assertEqual(parent.relation_tag, 'contains')
        self.assertEqual(parent.line, 29)
        self.assertEqual(parent.file_id, 'symbolic_tensor/module/st_moe.jsonl')
        self.assertEqual(parent.member_tag, '$assign_8')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$assign_8')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/module/st_moe.jsonl', '$functiondef_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$assign_8', child_members)

    def test_057_llm_client_coding_agent_query__assign_4(self):
        """go_to_parent('llm_client/coding_agent_query.jsonl', '$assign_4') -> '$if_3' via 'if_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'llm_client/coding_agent_query.jsonl', '$assign_4')
        self.assertEqual(parent.owner_tag, '$if_3')
        self.assertEqual(parent.relation_tag, 'if_body')
        self.assertEqual(parent.line, 33)
        self.assertEqual(parent.file_id, 'llm_client/coding_agent_query.jsonl')
        self.assertEqual(parent.member_tag, '$assign_4')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$assign_4')
        children = lexical_scope_expand_children(db, 'llm_client/coding_agent_query.jsonl', '$if_3')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$assign_4', child_members)

    def test_058_symbolic_tensor_function_st_moe_backward__call_174(self):
        """go_to_parent('symbolic_tensor/function/st_moe_backward.jsonl', '$call_174') -> '$expr_63' via 'expr_stmt'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe_backward.jsonl', '$call_174')
        self.assertEqual(parent.owner_tag, '$expr_63')
        self.assertEqual(parent.relation_tag, 'expr_stmt')
        self.assertEqual(parent.line, 537)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_moe_backward.jsonl')
        self.assertEqual(parent.member_tag, '$call_174')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$call_174')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe_backward.jsonl', '$expr_63')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$call_174', child_members)

    def test_059_symbolic_tensor_function_with_dense_view__expr_27(self):
        """go_to_parent('symbolic_tensor/function/with_dense_view.jsonl', '$expr_27') -> '$with_5' via 'with_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/with_dense_view.jsonl', '$expr_27')
        self.assertEqual(parent.owner_tag, '$with_5')
        self.assertEqual(parent.relation_tag, 'with_body')
        self.assertEqual(parent.line, 168)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/with_dense_view.jsonl')
        self.assertEqual(parent.member_tag, '$expr_27')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$expr_27')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/with_dense_view.jsonl', '$with_5')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$expr_27', child_members)

    def test_060_symbolic_tensor_tensor_util_dump_view__call_28(self):
        """go_to_parent('symbolic_tensor/tensor_util/dump_view.jsonl', '$call_28') -> '$expr_11' via 'expr_stmt'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/dump_view.jsonl', '$call_28')
        self.assertEqual(parent.owner_tag, '$expr_11')
        self.assertEqual(parent.relation_tag, 'expr_stmt')
        self.assertEqual(parent.line, 56)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/dump_view.jsonl')
        self.assertEqual(parent.member_tag, '$call_28')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$call_28')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/dump_view.jsonl', '$expr_11')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$call_28', child_members)

    def test_061_symbolic_tensor_tensor_util_assign_view__expr_3(self):
        """go_to_parent('symbolic_tensor/tensor_util/assign_view.jsonl', '$expr_3') -> '$if_0' via 'if_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/assign_view.jsonl', '$expr_3')
        self.assertEqual(parent.owner_tag, '$if_0')
        self.assertEqual(parent.relation_tag, 'if_body')
        self.assertEqual(parent.line, 35)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/assign_view.jsonl')
        self.assertEqual(parent.member_tag, '$expr_3')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$expr_3')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/assign_view.jsonl', '$if_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$expr_3', child_members)

    def test_062_symbolic_tensor_function_st_attention__with_0(self):
        """go_to_parent('symbolic_tensor/function/st_attention.jsonl', '$with_0') -> '$functiondef_2' via 'contains'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_attention.jsonl', '$with_0')
        self.assertEqual(parent.owner_tag, '$functiondef_2')
        self.assertEqual(parent.relation_tag, 'contains')
        self.assertEqual(parent.line, 54)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_attention.jsonl')
        self.assertEqual(parent.member_tag, '$with_0')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$with_0')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_attention.jsonl', '$functiondef_2')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$with_0', child_members)

    def test_063_symbolic_tensor_function_get_edit_distance_ratio__call_54(self):
        """go_to_parent('symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$call_54') -> '$expr_17' via 'expr_stmt'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$call_54')
        self.assertEqual(parent.owner_tag, '$expr_17')
        self.assertEqual(parent.relation_tag, 'expr_stmt')
        self.assertEqual(parent.line, 119)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl')
        self.assertEqual(parent.member_tag, '$call_54')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$call_54')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$expr_17')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$call_54', child_members)

    def test_064_symbolic_tensor_tensor_util_slice_tensor__docstring_3(self):
        """go_to_parent('symbolic_tensor/tensor_util/slice_tensor.jsonl', '$docstring_3') -> '$expr_1' via 'docstring'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/slice_tensor.jsonl', '$docstring_3')
        self.assertEqual(parent.owner_tag, '$expr_1')
        self.assertEqual(parent.relation_tag, 'docstring')
        self.assertEqual(parent.line, 23)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/slice_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$docstring_3')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$docstring_3')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/slice_tensor.jsonl', '$expr_1')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$docstring_3', child_members)

    def test_065_symbolic_tensor_function_symbolic_grad_registry__expr_2(self):
        """go_to_parent('symbolic_tensor/function/symbolic_grad_registry.jsonl', '$expr_2') -> '$functiondef_2' via 'contains'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/symbolic_grad_registry.jsonl', '$expr_2')
        self.assertEqual(parent.owner_tag, '$functiondef_2')
        self.assertEqual(parent.relation_tag, 'contains')
        self.assertEqual(parent.line, 24)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/symbolic_grad_registry.jsonl')
        self.assertEqual(parent.member_tag, '$expr_2')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$expr_2')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/symbolic_grad_registry.jsonl', '$functiondef_2')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$expr_2', child_members)

    def test_066_test_test_st_attention_followed_by_st_moe__assign_6(self):
        """go_to_parent('test/test_st_attention_followed_by_st_moe.jsonl', '$assign_6') -> '$if_3' via 'if_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'test/test_st_attention_followed_by_st_moe.jsonl', '$assign_6')
        self.assertEqual(parent.owner_tag, '$if_3')
        self.assertEqual(parent.relation_tag, 'if_body')
        self.assertEqual(parent.line, 45)
        self.assertEqual(parent.file_id, 'test/test_st_attention_followed_by_st_moe.jsonl')
        self.assertEqual(parent.member_tag, '$assign_6')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$assign_6')
        children = lexical_scope_expand_children(db, 'test/test_st_attention_followed_by_st_moe.jsonl', '$if_3')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$assign_6', child_members)

    def test_067_symbolic_tensor_function_merge_backward__call_42(self):
        """go_to_parent('symbolic_tensor/function/merge_backward.jsonl', '$call_42') -> '$expr_7' via 'expr_stmt'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/merge_backward.jsonl', '$call_42')
        self.assertEqual(parent.owner_tag, '$expr_7')
        self.assertEqual(parent.relation_tag, 'expr_stmt')
        self.assertEqual(parent.line, 118)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/merge_backward.jsonl')
        self.assertEqual(parent.member_tag, '$call_42')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$call_42')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/merge_backward.jsonl', '$expr_7')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$call_42', child_members)

    def test_068_fs_util_get_nested_list_file_pathes__assign_3(self):
        """go_to_parent('fs_util/get_nested_list_file_pathes.jsonl', '$assign_3') -> '$functiondef_1' via 'contains'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'fs_util/get_nested_list_file_pathes.jsonl', '$assign_3')
        self.assertEqual(parent.owner_tag, '$functiondef_1')
        self.assertEqual(parent.relation_tag, 'contains')
        self.assertEqual(parent.line, 18)
        self.assertEqual(parent.file_id, 'fs_util/get_nested_list_file_pathes.jsonl')
        self.assertEqual(parent.member_tag, '$assign_3')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$assign_3')
        children = lexical_scope_expand_children(db, 'fs_util/get_nested_list_file_pathes.jsonl', '$functiondef_1')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$assign_3', child_members)

    def test_069_symbolic_tensor_tensor_util_load_tensor__expr_15(self):
        """go_to_parent('symbolic_tensor/tensor_util/load_tensor.jsonl', '$expr_15') -> '$with_3' via 'with_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/load_tensor.jsonl', '$expr_15')
        self.assertEqual(parent.owner_tag, '$with_3')
        self.assertEqual(parent.relation_tag, 'with_body')
        self.assertEqual(parent.line, 84)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/load_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$expr_15')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$expr_15')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/load_tensor.jsonl', '$with_3')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$expr_15', child_members)

    def test_070_symbolic_tensor_function_st_copy__call_17(self):
        """go_to_parent('symbolic_tensor/function/st_copy.jsonl', '$call_17') -> '$expr_8' via 'expr_stmt'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_copy.jsonl', '$call_17')
        self.assertEqual(parent.owner_tag, '$expr_8')
        self.assertEqual(parent.relation_tag, 'expr_stmt')
        self.assertEqual(parent.line, 54)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_copy.jsonl')
        self.assertEqual(parent.member_tag, '$call_17')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$call_17')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_copy.jsonl', '$expr_8')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$call_17', child_members)

    def test_071_llm_client_raw_llm_task_handler__try_0(self):
        """go_to_parent('llm_client/raw_llm_task_handler.jsonl', '$try_0') -> '$for_2' via 'for_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'llm_client/raw_llm_task_handler.jsonl', '$try_0')
        self.assertEqual(parent.owner_tag, '$for_2')
        self.assertEqual(parent.relation_tag, 'for_body')
        self.assertEqual(parent.line, 18)
        self.assertEqual(parent.file_id, 'llm_client/raw_llm_task_handler.jsonl')
        self.assertEqual(parent.member_tag, '$try_0')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$try_0')
        children = lexical_scope_expand_children(db, 'llm_client/raw_llm_task_handler.jsonl', '$for_2')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$try_0', child_members)

    def test_072_symbolic_tensor_tensor_util_empty_tensor_like__expr_6(self):
        """go_to_parent('symbolic_tensor/tensor_util/empty_tensor_like.jsonl', '$expr_6') -> '$if_3' via 'if_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', '$expr_6')
        self.assertEqual(parent.owner_tag, '$if_3')
        self.assertEqual(parent.relation_tag, 'if_body')
        self.assertEqual(parent.line, 30)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl')
        self.assertEqual(parent.member_tag, '$expr_6')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$expr_6')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', '$if_3')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$expr_6', child_members)

    def test_073_symbolic_tensor_tensor_util_load_tensor__if_0(self):
        """go_to_parent('symbolic_tensor/tensor_util/load_tensor.jsonl', '$if_0') -> '$functiondef_3' via 'contains'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/load_tensor.jsonl', '$if_0')
        self.assertEqual(parent.owner_tag, '$functiondef_3')
        self.assertEqual(parent.relation_tag, 'contains')
        self.assertEqual(parent.line, 31)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/load_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$if_0')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$if_0')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/load_tensor.jsonl', '$functiondef_3')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$if_0', child_members)

    def test_074_symbolic_tensor_tensor_util_st_patched__expr_16(self):
        """go_to_parent('symbolic_tensor/tensor_util/st_patched.jsonl', '$expr_16') -> '$if_0' via 'if_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/st_patched.jsonl', '$expr_16')
        self.assertEqual(parent.owner_tag, '$if_0')
        self.assertEqual(parent.relation_tag, 'if_body')
        self.assertEqual(parent.line, 20)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/st_patched.jsonl')
        self.assertEqual(parent.member_tag, '$expr_16')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$expr_16')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/st_patched.jsonl', '$if_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$expr_16', child_members)

    def test_075_symbolic_tensor_tensor_util_make_tensor__call_112(self):
        """go_to_parent('symbolic_tensor/tensor_util/make_tensor.jsonl', '$call_112') -> '$expr_39' via 'expr_stmt'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/make_tensor.jsonl', '$call_112')
        self.assertEqual(parent.owner_tag, '$expr_39')
        self.assertEqual(parent.relation_tag, 'expr_stmt')
        self.assertEqual(parent.line, 154)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/make_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$call_112')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$call_112')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/make_tensor.jsonl', '$expr_39')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$call_112', child_members)

    def test_076_symbolic_tensor_function_merge__assign_23(self):
        """go_to_parent('symbolic_tensor/function/merge.jsonl', '$assign_23') -> '$with_2' via 'with_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/merge.jsonl', '$assign_23')
        self.assertEqual(parent.owner_tag, '$with_2')
        self.assertEqual(parent.relation_tag, 'with_body')
        self.assertEqual(parent.line, 94)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/merge.jsonl')
        self.assertEqual(parent.member_tag, '$assign_23')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$assign_23')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/merge.jsonl', '$with_2')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$assign_23', child_members)

    def test_077_symbolic_tensor_function_slice_view__expr_31(self):
        """go_to_parent('symbolic_tensor/function/slice_view.jsonl', '$expr_31') -> '$with_3' via 'with_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_view.jsonl', '$expr_31')
        self.assertEqual(parent.owner_tag, '$with_3')
        self.assertEqual(parent.relation_tag, 'with_body')
        self.assertEqual(parent.line, 147)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/slice_view.jsonl')
        self.assertEqual(parent.member_tag, '$expr_31')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$expr_31')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_view.jsonl', '$with_3')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$expr_31', child_members)

    def test_078_symbolic_tensor_tensor_util_st_patched__call_50(self):
        """go_to_parent('symbolic_tensor/tensor_util/st_patched.jsonl', '$call_50') -> '$expr_20' via 'expr_stmt'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/st_patched.jsonl', '$call_50')
        self.assertEqual(parent.owner_tag, '$expr_20')
        self.assertEqual(parent.relation_tag, 'expr_stmt')
        self.assertEqual(parent.line, 73)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/st_patched.jsonl')
        self.assertEqual(parent.member_tag, '$call_50')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$call_50')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/st_patched.jsonl', '$expr_20')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$call_50', child_members)

    def test_079_symbolic_tensor_function_slice_attention__importfrom_0(self):
        """go_to_parent('symbolic_tensor/function/slice_attention.jsonl', '$importfrom_0') -> '<module>' via 'contains'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/slice_attention.jsonl', '$importfrom_0')
        self.assertEqual(parent.owner_tag, '<module>')
        self.assertEqual(parent.relation_tag, 'contains')
        self.assertEqual(parent.line, 0)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/slice_attention.jsonl')
        self.assertEqual(parent.member_tag, '$importfrom_0')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$importfrom_0')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/slice_attention.jsonl', '<module>')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$importfrom_0', child_members)

    def test_080_symbolic_tensor_tensor_util_make_tensor__assign_24(self):
        """go_to_parent('symbolic_tensor/tensor_util/make_tensor.jsonl', '$assign_24') -> '$with_10' via 'with_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/make_tensor.jsonl', '$assign_24')
        self.assertEqual(parent.owner_tag, '$with_10')
        self.assertEqual(parent.relation_tag, 'with_body')
        self.assertEqual(parent.line, 155)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/make_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$assign_24')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$assign_24')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/make_tensor.jsonl', '$with_10')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$assign_24', child_members)

    def test_081_symbolic_tensor_module_with_dense_view__classdef_0(self):
        """go_to_parent('symbolic_tensor/module/with_dense_view.jsonl', '$classdef_0') -> '<module>' via 'contains'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/module/with_dense_view.jsonl', '$classdef_0')
        self.assertEqual(parent.owner_tag, '<module>')
        self.assertEqual(parent.relation_tag, 'contains')
        self.assertEqual(parent.line, 0)
        self.assertEqual(parent.file_id, 'symbolic_tensor/module/with_dense_view.jsonl')
        self.assertEqual(parent.member_tag, '$classdef_0')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$classdef_0')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/module/with_dense_view.jsonl', '<module>')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$classdef_0', child_members)

    def test_082_symbolic_tensor_function_select_qkv_indexes__assign_40(self):
        """go_to_parent('symbolic_tensor/function/select_qkv_indexes.jsonl', '$assign_40') -> '$with_3' via 'with_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/select_qkv_indexes.jsonl', '$assign_40')
        self.assertEqual(parent.owner_tag, '$with_3')
        self.assertEqual(parent.relation_tag, 'with_body')
        self.assertEqual(parent.line, 169)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/select_qkv_indexes.jsonl')
        self.assertEqual(parent.member_tag, '$assign_40')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$assign_40')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/select_qkv_indexes.jsonl', '$with_3')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$assign_40', child_members)

    def test_083_symbolic_tensor_function_get_causal_attention_mask__expr_2(self):
        """go_to_parent('symbolic_tensor/function/get_causal_attention_mask.jsonl', '$expr_2') -> '$if_1' via 'if_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/get_causal_attention_mask.jsonl', '$expr_2')
        self.assertEqual(parent.owner_tag, '$if_1')
        self.assertEqual(parent.relation_tag, 'if_body')
        self.assertEqual(parent.line, 21)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/get_causal_attention_mask.jsonl')
        self.assertEqual(parent.member_tag, '$expr_2')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$expr_2')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/get_causal_attention_mask.jsonl', '$if_1')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$expr_2', child_members)

    def test_084_symbolic_tensor_function_st_moe__assign_32(self):
        """go_to_parent('symbolic_tensor/function/st_moe.jsonl', '$assign_32') -> '$with_2' via 'with_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/st_moe.jsonl', '$assign_32')
        self.assertEqual(parent.owner_tag, '$with_2')
        self.assertEqual(parent.relation_tag, 'with_body')
        self.assertEqual(parent.line, 145)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/st_moe.jsonl')
        self.assertEqual(parent.member_tag, '$assign_32')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$assign_32')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/st_moe.jsonl', '$with_2')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$assign_32', child_members)

    def test_085_symbolic_tensor_module_st_moe__call_9(self):
        """go_to_parent('symbolic_tensor/module/st_moe.jsonl', '$call_9') -> '$expr_4' via 'expr_stmt'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/module/st_moe.jsonl', '$call_9')
        self.assertEqual(parent.owner_tag, '$expr_4')
        self.assertEqual(parent.relation_tag, 'expr_stmt')
        self.assertEqual(parent.line, 76)
        self.assertEqual(parent.file_id, 'symbolic_tensor/module/st_moe.jsonl')
        self.assertEqual(parent.member_tag, '$call_9')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$call_9')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/module/st_moe.jsonl', '$expr_4')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$call_9', child_members)

    def test_086_symbolic_tensor_tensor_util_slice_view__call_89(self):
        """go_to_parent('symbolic_tensor/tensor_util/slice_view.jsonl', '$call_89') -> '$with_5' via 'with_context'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/slice_view.jsonl', '$call_89')
        self.assertEqual(parent.owner_tag, '$with_5')
        self.assertEqual(parent.relation_tag, 'with_context')
        self.assertEqual(parent.line, 143)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/slice_view.jsonl')
        self.assertEqual(parent.member_tag, '$call_89')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$call_89')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/slice_view.jsonl', '$with_5')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$call_89', child_members)

    def test_087_symbolic_tensor_optimizer_st_sgd__expr_23(self):
        """go_to_parent('symbolic_tensor/optimizer/st_sgd.jsonl', '$expr_23') -> '$if_20' via 'if_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/optimizer/st_sgd.jsonl', '$expr_23')
        self.assertEqual(parent.owner_tag, '$if_20')
        self.assertEqual(parent.relation_tag, 'if_body')
        self.assertEqual(parent.line, 178)
        self.assertEqual(parent.file_id, 'symbolic_tensor/optimizer/st_sgd.jsonl')
        self.assertEqual(parent.member_tag, '$expr_23')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$expr_23')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/optimizer/st_sgd.jsonl', '$if_20')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$expr_23', child_members)

    def test_088_symbolic_tensor_tensor_util_pack_tensor__expr_15(self):
        """go_to_parent('symbolic_tensor/tensor_util/pack_tensor.jsonl', '$expr_15') -> '$with_2' via 'with_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/pack_tensor.jsonl', '$expr_15')
        self.assertEqual(parent.owner_tag, '$with_2')
        self.assertEqual(parent.relation_tag, 'with_body')
        self.assertEqual(parent.line, 42)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/pack_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$expr_15')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$expr_15')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/pack_tensor.jsonl', '$with_2')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$expr_15', child_members)

    def test_089_fs_util_pack_dir__import_0(self):
        """go_to_parent('fs_util/pack_dir.jsonl', '$import_0') -> '<module>' via 'contains'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'fs_util/pack_dir.jsonl', '$import_0')
        self.assertEqual(parent.owner_tag, '<module>')
        self.assertEqual(parent.relation_tag, 'contains')
        self.assertEqual(parent.line, 0)
        self.assertEqual(parent.file_id, 'fs_util/pack_dir.jsonl')
        self.assertEqual(parent.member_tag, '$import_0')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$import_0')
        children = lexical_scope_expand_children(db, 'fs_util/pack_dir.jsonl', '<module>')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$import_0', child_members)

    def test_090_llm_client_coding_agent_task_handler__assign_0(self):
        """go_to_parent('llm_client/coding_agent_task_handler.jsonl', '$assign_0') -> '$functiondef_0' via 'contains'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'llm_client/coding_agent_task_handler.jsonl', '$assign_0')
        self.assertEqual(parent.owner_tag, '$functiondef_0')
        self.assertEqual(parent.relation_tag, 'contains')
        self.assertEqual(parent.line, 6)
        self.assertEqual(parent.file_id, 'llm_client/coding_agent_task_handler.jsonl')
        self.assertEqual(parent.member_tag, '$assign_0')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$assign_0')
        children = lexical_scope_expand_children(db, 'llm_client/coding_agent_task_handler.jsonl', '$functiondef_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$assign_0', child_members)

    def test_091_symbolic_tensor_tensor_util_make_tensor__expr_37(self):
        """go_to_parent('symbolic_tensor/tensor_util/make_tensor.jsonl', '$expr_37') -> '$with_8' via 'with_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/make_tensor.jsonl', '$expr_37')
        self.assertEqual(parent.owner_tag, '$with_8')
        self.assertEqual(parent.relation_tag, 'with_body')
        self.assertEqual(parent.line, 150)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/make_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$expr_37')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$expr_37')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/make_tensor.jsonl', '$with_8')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$expr_37', child_members)

    def test_092_sparse_util_convert_nested_list_coordinates_to_pairs_coordinates__annassign_0(self):
        """go_to_parent('sparse_util/convert_nested_list_coordinates_to_pairs_coordinates.jsonl', '$annassign_0') -> '$functiondef_2' via 'contains'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'sparse_util/convert_nested_list_coordinates_to_pairs_coordinates.jsonl', '$annassign_0')
        self.assertEqual(parent.owner_tag, '$functiondef_2')
        self.assertEqual(parent.relation_tag, 'contains')
        self.assertEqual(parent.line, 23)
        self.assertEqual(parent.file_id, 'sparse_util/convert_nested_list_coordinates_to_pairs_coordinates.jsonl')
        self.assertEqual(parent.member_tag, '$annassign_0')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$annassign_0')
        children = lexical_scope_expand_children(db, 'sparse_util/convert_nested_list_coordinates_to_pairs_coordinates.jsonl', '$functiondef_2')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$annassign_0', child_members)

    def test_093_symbolic_tensor_function_get_edit_distance_ratio__call_68(self):
        """go_to_parent('symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$call_68') -> '$expr_21' via 'expr_stmt'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$call_68')
        self.assertEqual(parent.owner_tag, '$expr_21')
        self.assertEqual(parent.relation_tag, 'expr_stmt')
        self.assertEqual(parent.line, 139)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl')
        self.assertEqual(parent.member_tag, '$call_68')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$call_68')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$expr_21')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$call_68', child_members)

    def test_094_symbolic_tensor_function_get_edit_distance_ratio__expr_40(self):
        """go_to_parent('symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$expr_40') -> '$if_3' via 'if_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$expr_40')
        self.assertEqual(parent.owner_tag, '$if_3')
        self.assertEqual(parent.relation_tag, 'if_body')
        self.assertEqual(parent.line, 112)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl')
        self.assertEqual(parent.member_tag, '$expr_40')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$expr_40')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', '$if_3')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$expr_40', child_members)

    def test_095_symbolic_tensor_function_merge_backward__assert_2(self):
        """go_to_parent('symbolic_tensor/function/merge_backward.jsonl', '$assert_2') -> '$for_0' via 'for_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/merge_backward.jsonl', '$assert_2')
        self.assertEqual(parent.owner_tag, '$for_0')
        self.assertEqual(parent.relation_tag, 'for_body')
        self.assertEqual(parent.line, 73)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/merge_backward.jsonl')
        self.assertEqual(parent.member_tag, '$assert_2')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$assert_2')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/merge_backward.jsonl', '$for_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$assert_2', child_members)

    def test_096_symbolic_tensor_tensor_util_patch_tensor__if_0(self):
        """go_to_parent('symbolic_tensor/tensor_util/patch_tensor.jsonl', '$if_0') -> '$for_0' via 'for_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/patch_tensor.jsonl', '$if_0')
        self.assertEqual(parent.owner_tag, '$for_0')
        self.assertEqual(parent.relation_tag, 'for_body')
        self.assertEqual(parent.line, 35)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/patch_tensor.jsonl')
        self.assertEqual(parent.member_tag, '$if_0')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$if_0')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/patch_tensor.jsonl', '$for_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$if_0', child_members)

    def test_097_symbolic_tensor_tensor_util_register_tensor_ops__return_6(self):
        """go_to_parent('symbolic_tensor/tensor_util/register_tensor_ops.jsonl', '$return_6') -> '$functiondef_6' via 'contains'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl', '$return_6')
        self.assertEqual(parent.owner_tag, '$functiondef_6')
        self.assertEqual(parent.relation_tag, 'contains')
        self.assertEqual(parent.line, 26)
        self.assertEqual(parent.file_id, 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl')
        self.assertEqual(parent.member_tag, '$return_6')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$return_6')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl', '$functiondef_6')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$return_6', child_members)

    def test_098_symbolic_tensor_function_coding_agent__return_8(self):
        """go_to_parent('symbolic_tensor/function/coding_agent.jsonl', '$return_8') -> '$functiondef_7' via 'contains'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'symbolic_tensor/function/coding_agent.jsonl', '$return_8')
        self.assertEqual(parent.owner_tag, '$functiondef_7')
        self.assertEqual(parent.relation_tag, 'contains')
        self.assertEqual(parent.line, 202)
        self.assertEqual(parent.file_id, 'symbolic_tensor/function/coding_agent.jsonl')
        self.assertEqual(parent.member_tag, '$return_8')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$return_8')
        children = lexical_scope_expand_children(db, 'symbolic_tensor/function/coding_agent.jsonl', '$functiondef_7')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$return_8', child_members)

    def test_099_test_test_transform_method_time_comparison__assign_1(self):
        """go_to_parent('test/test_transform_method_time_comparison.jsonl', '$assign_1') -> '$with_0' via 'with_body'"""
        db = _db()
        parent = lexical_scope_go_to_parent(db, 'test/test_transform_method_time_comparison.jsonl', '$assign_1')
        self.assertEqual(parent.owner_tag, '$with_0')
        self.assertEqual(parent.relation_tag, 'with_body')
        self.assertEqual(parent.line, 10)
        self.assertEqual(parent.file_id, 'test/test_transform_method_time_comparison.jsonl')
        self.assertEqual(parent.member_tag, '$assign_1')
        self.assertIn(parent.relation_tag, LEXICAL_RELATION_TAGS)
        self.assertNotEqual(parent.owner_tag, '$assign_1')
        children = lexical_scope_expand_children(db, 'test/test_transform_method_time_comparison.jsonl', '$with_0')
        child_members = [ch.member_tag for ch in children]
        self.assertIn('$assign_1', child_members)


if __name__ == "__main__":
    unittest.main()
