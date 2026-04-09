"""
test_dynamic_scope_find_all_references: 100 concrete test cases.
Each case: symbol_name -> expected ref count, expected file_ids, expected relation_tags.
Round-trip: go_to_definition finds the original definition.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ast_tag_db import load_jsonl_dataset_into_ast_tag_db, AstTagDB
from relation_tag_classification import DYNAMIC_RELATION_TAGS
from tag_actions.dynamic_scope_find_all_references import dynamic_scope_find_all_references
from tag_actions.dynamic_scope_go_to_definition import dynamic_scope_go_to_definition


def _db() -> AstTagDB:
    if not hasattr(_db, "_instance"):
        dataset_dir = os.path.join(os.path.dirname(__file__), "..", "test_dataset")
        _db._instance = load_jsonl_dataset_into_ast_tag_db(dataset_dir)
    return _db._instance


class TestDynamicScopeFindAllReferences(unittest.TestCase):

    def test_000__resolve_grad_output(self):
        """find_all_references('_resolve_grad_output'): 3 refs across 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_resolve_grad_output')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 3)
        for r in refs:
            self.assertEqual(r.member_tag, '_resolve_grad_output')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 3)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls', 'imports'])
        defs = dynamic_scope_go_to_definition(db, '_resolve_grad_output')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/slice_view.jsonl', def_file_ids)

    def test_001_SimpleMerger(self):
        """find_all_references('SimpleMerger'): 3 refs across 3 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'SimpleMerger')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 3)
        for r in refs:
            self.assertEqual(r.member_tag, 'SimpleMerger')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 3)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/st_attention.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['call_pos_arg', 'keyword_arg'])
        defs = dynamic_scope_go_to_definition(db, 'SimpleMerger')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/st_attention.jsonl', def_file_ids)

    def test_002__save_st_attrs(self):
        """find_all_references('_save_st_attrs'): 3 refs across 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_save_st_attrs')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 3)
        for r in refs:
            self.assertEqual(r.member_tag, '_save_st_attrs')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 3)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls', 'imports'])
        defs = dynamic_scope_go_to_definition(db, '_save_st_attrs')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/slice_view.jsonl', def_file_ids)

    def test_003__write_storage(self):
        """find_all_references('_write_storage'): 4 refs across 4 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_write_storage')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 4)
        for r in refs:
            self.assertEqual(r.member_tag, '_write_storage')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 4)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_stack.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, '_write_storage')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/merge_forward.jsonl', def_file_ids)

    def test_004_pop(self):
        """find_all_references('pop'): 22 refs across 17 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'pop')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 22)
        for r in refs:
            self.assertEqual(r.member_tag, 'pop')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 22)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['llm_client/coding_agent_query.jsonl', 'llm_client/coding_agent_task_handler.jsonl', 'symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_query_tensor.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/symbolic_grad_registry.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['attr_name'])
        defs = dynamic_scope_go_to_definition(db, 'pop')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/symbolic_grad_registry.jsonl', def_file_ids)

    def test_005__get_raw_storage_path(self):
        """find_all_references('_get_raw_storage_path'): 4 refs across 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_get_raw_storage_path')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 4)
        for r in refs:
            self.assertEqual(r.member_tag, '_get_raw_storage_path')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 4)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, '_get_raw_storage_path')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/tensor_util/assign_view.jsonl', def_file_ids)

    def test_006_read_storage(self):
        """find_all_references('read_storage'): 167 refs across 21 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'read_storage')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 167)
        for r in refs:
            self.assertEqual(r.member_tag, 'read_storage')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 167)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'test/test_attention_vs_traditional.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, 'read_storage')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/fork_tensor.jsonl', def_file_ids)

    def test_007_SimpleMerger(self):
        """find_all_references('SimpleMerger'): 3 refs across 3 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'SimpleMerger')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 3)
        for r in refs:
            self.assertEqual(r.member_tag, 'SimpleMerger')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 3)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/st_attention.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['call_pos_arg', 'keyword_arg'])
        defs = dynamic_scope_go_to_definition(db, 'SimpleMerger')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/merge_forward.jsonl', def_file_ids)

    def test_008_TextMerger(self):
        """find_all_references('TextMerger'): 36 refs across 7 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'TextMerger')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 36)
        for r in refs:
            self.assertEqual(r.member_tag, 'TextMerger')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 36)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'test/test_attention_vs_traditional.jsonl', 'test/test_st_attention_followed_by_st_moe.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['assigns', 'attr_value', 'bool_op_operand', 'imports'])
        defs = dynamic_scope_go_to_definition(db, 'TextMerger')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('fs_util/text_merger.jsonl', def_file_ids)

    def test_009__write_storage(self):
        """find_all_references('_write_storage'): 4 refs across 4 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_write_storage')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 4)
        for r in refs:
            self.assertEqual(r.member_tag, '_write_storage')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 4)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_stack.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, '_write_storage')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/st_moe_backward.jsonl', def_file_ids)

    def test_010_backward(self):
        """find_all_references('backward'): 3 refs across 3 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'backward')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 3)
        for r in refs:
            self.assertEqual(r.member_tag, 'backward')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 3)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['attr_name'])
        defs = dynamic_scope_go_to_definition(db, 'backward')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/merge.jsonl', def_file_ids)

    def test_011_merge_backward(self):
        """find_all_references('merge_backward'): 9 refs across 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'merge_backward')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 9)
        for r in refs:
            self.assertEqual(r.member_tag, 'merge_backward')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 9)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls', 'imports'])
        defs = dynamic_scope_go_to_definition(db, 'merge_backward')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/merge_backward.jsonl', def_file_ids)

    def test_012__flatten_nested(self):
        """find_all_references('_flatten_nested'): 4 refs across 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_flatten_nested')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 4)
        for r in refs:
            self.assertEqual(r.member_tag, '_flatten_nested')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 4)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['llm_client/coding_agent_task_handler.jsonl', 'llm_client/raw_llm_task_handler.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, '_flatten_nested')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('llm_client/raw_llm_task_handler.jsonl', def_file_ids)

    def test_013_read_storage(self):
        """find_all_references('read_storage'): 167 refs across 21 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'read_storage')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 167)
        for r in refs:
            self.assertEqual(r.member_tag, 'read_storage')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 167)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'test/test_attention_vs_traditional.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, 'read_storage')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/tensor_util/patch_tensor.jsonl', def_file_ids)

    def test_014_forward(self):
        """find_all_references('forward'): 2 refs across 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'forward')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r.member_tag, 'forward')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 2)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/st_copy.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['attr_name'])
        defs = dynamic_scope_go_to_definition(db, 'forward')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/merge.jsonl', def_file_ids)

    def test_015_run_test(self):
        """find_all_references('run_test'): 830 refs across 43 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'run_test')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 830)
        for r in refs:
            self.assertEqual(r.member_tag, 'run_test')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 830)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_causal_attention_mask.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/select_qkv_indexes.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/dump_tensor.jsonl', 'symbolic_tensor/tensor_util/dump_view.jsonl', 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl', 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl', 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', 'symbolic_tensor/tensor_util/pack_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, 'run_test')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/tensor_util/empty_tensor_like.jsonl', def_file_ids)

    def test_016_run_test(self):
        """find_all_references('run_test'): 830 refs across 43 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'run_test')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 830)
        for r in refs:
            self.assertEqual(r.member_tag, 'run_test')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 830)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_causal_attention_mask.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/select_qkv_indexes.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/dump_tensor.jsonl', 'symbolic_tensor/tensor_util/dump_view.jsonl', 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl', 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl', 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', 'symbolic_tensor/tensor_util/pack_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, 'run_test')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/tensor_util/make_tensor.jsonl', def_file_ids)

    def test_017_run_test(self):
        """find_all_references('run_test'): 830 refs across 43 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'run_test')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 830)
        for r in refs:
            self.assertEqual(r.member_tag, 'run_test')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 830)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_causal_attention_mask.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/select_qkv_indexes.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/dump_tensor.jsonl', 'symbolic_tensor/tensor_util/dump_view.jsonl', 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl', 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl', 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', 'symbolic_tensor/tensor_util/pack_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, 'run_test')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/module/st_moe.jsonl', def_file_ids)

    def test_018_read_storage(self):
        """find_all_references('read_storage'): 167 refs across 21 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'read_storage')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 167)
        for r in refs:
            self.assertEqual(r.member_tag, 'read_storage')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 167)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'test/test_attention_vs_traditional.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, 'read_storage')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/get_edit_distance_ratio.jsonl', def_file_ids)

    def test_019_read_storage(self):
        """find_all_references('read_storage'): 167 refs across 21 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'read_storage')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 167)
        for r in refs:
            self.assertEqual(r.member_tag, 'read_storage')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 167)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'test/test_attention_vs_traditional.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, 'read_storage')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/slice_attention.jsonl', def_file_ids)

    def test_020_st_stack(self):
        """find_all_references('st_stack'): 2 refs across 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'st_stack')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r.member_tag, 'st_stack')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 2)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/st_stack.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, 'st_stack')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/st_stack.jsonl', def_file_ids)

    def test_021_zero_first(self):
        """find_all_references('zero_first'): 1 refs across 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'zero_first')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r.member_tag, 'zero_first')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 1)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/module/with_dense_view.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['keyword_arg'])
        defs = dynamic_scope_go_to_definition(db, 'zero_first')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/module/with_dense_view.jsonl', def_file_ids)

    def test_022__get_storage_path(self):
        """find_all_references('_get_storage_path'): 23 refs across 12 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_get_storage_path')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 23)
        for r in refs:
            self.assertEqual(r.member_tag, '_get_storage_path')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 23)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['fs_util/get_nested_list_file_pathes.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls', 'imports'])
        defs = dynamic_scope_go_to_definition(db, '_get_storage_path')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/tensor_util/assign_tensor.jsonl', def_file_ids)

    def test_023_run_test(self):
        """find_all_references('run_test'): 830 refs across 43 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'run_test')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 830)
        for r in refs:
            self.assertEqual(r.member_tag, 'run_test')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 830)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_causal_attention_mask.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/select_qkv_indexes.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/dump_tensor.jsonl', 'symbolic_tensor/tensor_util/dump_view.jsonl', 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl', 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl', 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', 'symbolic_tensor/tensor_util/pack_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, 'run_test')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('test/test_st_attention_followed_by_st_moe.jsonl', def_file_ids)

    def test_024_read_out(self):
        """find_all_references('read_out'): 131 refs across 7 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'read_out')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 131)
        for r in refs:
            self.assertEqual(r.member_tag, 'read_out')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 131)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_stack.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, 'read_out')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/slice_view.jsonl', def_file_ids)

    def test_025__unflatten(self):
        """find_all_references('_unflatten'): 4 refs across 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_unflatten')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 4)
        for r in refs:
            self.assertEqual(r.member_tag, '_unflatten')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 4)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, '_unflatten')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/tensor_util/load_tensor.jsonl', def_file_ids)

    def test_026_default_prompt_for_grad_input(self):
        """find_all_references('default_prompt_for_grad_input'): 3 refs across 3 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'default_prompt_for_grad_input')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 3)
        for r in refs:
            self.assertEqual(r.member_tag, 'default_prompt_for_grad_input')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 3)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['bool_op_operand', 'imports'])
        defs = dynamic_scope_go_to_definition(db, 'default_prompt_for_grad_input')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/st_moe_backward.jsonl', def_file_ids)

    def test_027__flatten_nested(self):
        """find_all_references('_flatten_nested'): 4 refs across 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_flatten_nested')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 4)
        for r in refs:
            self.assertEqual(r.member_tag, '_flatten_nested')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 4)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['llm_client/coding_agent_task_handler.jsonl', 'llm_client/raw_llm_task_handler.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, '_flatten_nested')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('llm_client/coding_agent_task_handler.jsonl', def_file_ids)

    def test_028_run_test(self):
        """find_all_references('run_test'): 830 refs across 43 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'run_test')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 830)
        for r in refs:
            self.assertEqual(r.member_tag, 'run_test')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 830)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_causal_attention_mask.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/select_qkv_indexes.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/dump_tensor.jsonl', 'symbolic_tensor/tensor_util/dump_view.jsonl', 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl', 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl', 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', 'symbolic_tensor/tensor_util/pack_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, 'run_test')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/tensor_util/sparse_to_dense.jsonl', def_file_ids)

    def test_029__sparse_to_dense_impl(self):
        """find_all_references('_sparse_to_dense_impl'): 4 refs across 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_sparse_to_dense_impl')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 4)
        for r in refs:
            self.assertEqual(r.member_tag, '_sparse_to_dense_impl')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 4)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls', 'imports'])
        defs = dynamic_scope_go_to_definition(db, '_sparse_to_dense_impl')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/tensor_util/sparse_to_dense.jsonl', def_file_ids)

    def test_030_run_test(self):
        """find_all_references('run_test'): 830 refs across 43 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'run_test')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 830)
        for r in refs:
            self.assertEqual(r.member_tag, 'run_test')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 830)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_causal_attention_mask.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/select_qkv_indexes.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/dump_tensor.jsonl', 'symbolic_tensor/tensor_util/dump_view.jsonl', 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl', 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl', 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', 'symbolic_tensor/tensor_util/pack_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, 'run_test')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/slice_attention.jsonl', def_file_ids)

    def test_031_read_out(self):
        """find_all_references('read_out'): 131 refs across 7 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'read_out')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 131)
        for r in refs:
            self.assertEqual(r.member_tag, 'read_out')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 131)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_stack.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, 'read_out')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', def_file_ids)

    def test_032_st_assign_view(self):
        """find_all_references('st_assign_view'): 2 refs across 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'st_assign_view')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r.member_tag, 'st_assign_view')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 2)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/tensor_util/register_tensor_ops.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['assigns', 'attr_name'])
        defs = dynamic_scope_go_to_definition(db, 'st_assign_view')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/tensor_util/register_tensor_ops.jsonl', def_file_ids)

    def test_033_st_stack_forward(self):
        """find_all_references('st_stack_forward'): 12 refs across 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'st_stack_forward')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 12)
        for r in refs:
            self.assertEqual(r.member_tag, 'st_stack_forward')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 12)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/st_stack.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, 'st_stack_forward')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/st_stack.jsonl', def_file_ids)

    def test_034_forward(self):
        """find_all_references('forward'): 2 refs across 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'forward')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r.member_tag, 'forward')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 2)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/st_copy.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['attr_name'])
        defs = dynamic_scope_go_to_definition(db, 'forward')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/with_dense_view.jsonl', def_file_ids)

    def test_035_run_test(self):
        """find_all_references('run_test'): 830 refs across 43 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'run_test')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 830)
        for r in refs:
            self.assertEqual(r.member_tag, 'run_test')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 830)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_causal_attention_mask.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/select_qkv_indexes.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/dump_tensor.jsonl', 'symbolic_tensor/tensor_util/dump_view.jsonl', 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl', 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl', 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', 'symbolic_tensor/tensor_util/pack_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, 'run_test')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/tensor_util/load_tensor.jsonl', def_file_ids)

    def test_036_read_output(self):
        """find_all_references('read_output'): 4 refs across 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'read_output')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 4)
        for r in refs:
            self.assertEqual(r.member_tag, 'read_output')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 4)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/coding_agent.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, 'read_output')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/coding_agent.jsonl', def_file_ids)

    def test_037_unpack(self):
        """find_all_references('unpack'): 28 refs across 7 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'unpack')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 28)
        for r in refs:
            self.assertEqual(r.member_tag, 'unpack')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 28)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['fs_util/text_merger.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'test/test_attention_vs_traditional.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['assigns', 'attr_name', 'call_pos_arg'])
        defs = dynamic_scope_go_to_definition(db, 'unpack')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/st_attention.jsonl', def_file_ids)

    def test_038_st_get_diff(self):
        """find_all_references('st_get_diff'): 3 refs across 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'st_get_diff')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 3)
        for r in refs:
            self.assertEqual(r.member_tag, 'st_get_diff')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 3)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['assigns', 'attr_name'])
        defs = dynamic_scope_go_to_definition(db, 'st_get_diff')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/tensor_util/register_tensor_ops.jsonl', def_file_ids)

    def test_039__flatten_nested_paths(self):
        """find_all_references('_flatten_nested_paths'): 3 refs across 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_flatten_nested_paths')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 3)
        for r in refs:
            self.assertEqual(r.member_tag, '_flatten_nested_paths')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 3)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/optimizer/st_sgd.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, '_flatten_nested_paths')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/optimizer/st_sgd.jsonl', def_file_ids)

    def test_040_unpack(self):
        """find_all_references('unpack'): 28 refs across 7 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'unpack')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 28)
        for r in refs:
            self.assertEqual(r.member_tag, 'unpack')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 28)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['fs_util/text_merger.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'test/test_attention_vs_traditional.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['assigns', 'attr_name', 'call_pos_arg'])
        defs = dynamic_scope_go_to_definition(db, 'unpack')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('fs_util/text_merger.jsonl', def_file_ids)

    def test_041_frame_to_str(self):
        """find_all_references('frame_to_str'): 1 refs across 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'frame_to_str')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r.member_tag, 'frame_to_str')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 1)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['fs_util/text_merger.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, 'frame_to_str')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('fs_util/text_merger.jsonl', def_file_ids)

    def test_042_SimpleMerger(self):
        """find_all_references('SimpleMerger'): 3 refs across 3 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'SimpleMerger')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 3)
        for r in refs:
            self.assertEqual(r.member_tag, 'SimpleMerger')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 3)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/st_attention.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['call_pos_arg', 'keyword_arg'])
        defs = dynamic_scope_go_to_definition(db, 'SimpleMerger')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/merge.jsonl', def_file_ids)

    def test_043_backward(self):
        """find_all_references('backward'): 3 refs across 3 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'backward')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 3)
        for r in refs:
            self.assertEqual(r.member_tag, 'backward')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 3)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['attr_name'])
        defs = dynamic_scope_go_to_definition(db, 'backward')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/st_stack.jsonl', def_file_ids)

    def test_044__pad_indexes_to_topk_with_none_experience_indexes(self):
        """find_all_references('_pad_indexes_to_topk_with_none_experience_indexes'): 3 refs across 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_pad_indexes_to_topk_with_none_experience_indexes')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 3)
        for r in refs:
            self.assertEqual(r.member_tag, '_pad_indexes_to_topk_with_none_experience_indexes')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 3)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/st_moe_backward.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, '_pad_indexes_to_topk_with_none_experience_indexes')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/st_moe_backward.jsonl', def_file_ids)

    def test_045__collect(self):
        """find_all_references('_collect'): 2 refs across 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_collect')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r.member_tag, '_collect')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 2)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['sparse_util/convert_nested_list_coordinates_to_pairs_coordinates.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, '_collect')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('sparse_util/convert_nested_list_coordinates_to_pairs_coordinates.jsonl', def_file_ids)

    def test_046_unpack(self):
        """find_all_references('unpack'): 28 refs across 7 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'unpack')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 28)
        for r in refs:
            self.assertEqual(r.member_tag, 'unpack')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 28)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['fs_util/text_merger.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'test/test_attention_vs_traditional.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['assigns', 'attr_name', 'call_pos_arg'])
        defs = dynamic_scope_go_to_definition(db, 'unpack')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/merge_forward.jsonl', def_file_ids)

    def test_047__update_queries(self):
        """find_all_references('_update_queries'): 1 refs across 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_update_queries')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r.member_tag, '_update_queries')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 1)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/optimizer/st_sgd.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['attr_name'])
        defs = dynamic_scope_go_to_definition(db, '_update_queries')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/optimizer/st_sgd.jsonl', def_file_ids)

    def test_048_patch_tensor(self):
        """find_all_references('patch_tensor'): 11 refs across 4 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'patch_tensor')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 11)
        for r in refs:
            self.assertEqual(r.member_tag, 'patch_tensor')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 11)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls', 'imports'])
        defs = dynamic_scope_go_to_definition(db, 'patch_tensor')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/tensor_util/patch_tensor.jsonl', def_file_ids)

    def test_049_fork_tensor_forward(self):
        """find_all_references('fork_tensor_forward'): 6 refs across 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'fork_tensor_forward')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 6)
        for r in refs:
            self.assertEqual(r.member_tag, 'fork_tensor_forward')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 6)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/fork_tensor.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, 'fork_tensor_forward')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/fork_tensor.jsonl', def_file_ids)

    def test_050_register(self):
        """find_all_references('register'): 11 refs across 9 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'register')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 11)
        for r in refs:
            self.assertEqual(r.member_tag, 'register')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 11)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['attr_name'])
        defs = dynamic_scope_go_to_definition(db, 'register')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/symbolic_grad_registry.jsonl', def_file_ids)

    def test_051_StMoeModule(self):
        """find_all_references('StMoeModule'): 6 refs across 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'StMoeModule')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 6)
        for r in refs:
            self.assertEqual(r.member_tag, 'StMoeModule')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 6)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/module/st_moe.jsonl', 'test/test_st_attention_followed_by_st_moe.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls', 'imports'])
        defs = dynamic_scope_go_to_definition(db, 'StMoeModule')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/module/st_moe.jsonl', def_file_ids)

    def test_052_step(self):
        """find_all_references('step'): 4 refs across 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'step')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 4)
        for r in refs:
            self.assertEqual(r.member_tag, 'step')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 4)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/optimizer/st_sgd.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['attr_name'])
        defs = dynamic_scope_go_to_definition(db, 'step')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/optimizer/st_sgd.jsonl', def_file_ids)

    def test_053__expand(self):
        """find_all_references('_expand'): 2 refs across 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_expand')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r.member_tag, '_expand')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 2)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/tensor_util/empty_tensor_like.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, '_expand')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/tensor_util/empty_tensor_like.jsonl', def_file_ids)

    def test_054_read_storage(self):
        """find_all_references('read_storage'): 167 refs across 21 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'read_storage')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 167)
        for r in refs:
            self.assertEqual(r.member_tag, 'read_storage')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 167)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'test/test_attention_vs_traditional.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, 'read_storage')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/tensor_util/assign_tensor.jsonl', def_file_ids)

    def test_055_read_storage(self):
        """find_all_references('read_storage'): 167 refs across 21 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'read_storage')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 167)
        for r in refs:
            self.assertEqual(r.member_tag, 'read_storage')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 167)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'test/test_attention_vs_traditional.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, 'read_storage')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('test/test_attention_vs_traditional.jsonl', def_file_ids)

    def test_056__dense_to_sparse_impl(self):
        """find_all_references('_dense_to_sparse_impl'): 3 refs across 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_dense_to_sparse_impl')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 3)
        for r in refs:
            self.assertEqual(r.member_tag, '_dense_to_sparse_impl')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 3)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls', 'imports'])
        defs = dynamic_scope_go_to_definition(db, '_dense_to_sparse_impl')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/tensor_util/dense_to_sparse.jsonl', def_file_ids)

    def test_057_st_moe_backward_grad_experience(self):
        """find_all_references('st_moe_backward_grad_experience'): 1 refs across 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'st_moe_backward_grad_experience')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r.member_tag, 'st_moe_backward_grad_experience')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 1)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/st_moe_backward.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, 'st_moe_backward_grad_experience')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/st_moe_backward.jsonl', def_file_ids)

    def test_058__flat_index_from_coordinates(self):
        """find_all_references('_flat_index_from_coordinates'): 1 refs across 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_flat_index_from_coordinates')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r.member_tag, '_flat_index_from_coordinates')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 1)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/tensor_util/dump_view.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, '_flat_index_from_coordinates')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/tensor_util/dump_view.jsonl', def_file_ids)

    def test_059_assign_tensor(self):
        """find_all_references('assign_tensor'): 28 refs across 10 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'assign_tensor')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 28)
        for r in refs:
            self.assertEqual(r.member_tag, 'assign_tensor')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 28)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls', 'imports'])
        defs = dynamic_scope_go_to_definition(db, 'assign_tensor')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/tensor_util/assign_tensor.jsonl', def_file_ids)

    def test_060_get_edit_distance_ratio_backward_impl(self):
        """find_all_references('get_edit_distance_ratio_backward_impl'): 3 refs across 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'get_edit_distance_ratio_backward_impl')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 3)
        for r in refs:
            self.assertEqual(r.member_tag, 'get_edit_distance_ratio_backward_impl')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 3)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/get_edit_distance_ratio.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, 'get_edit_distance_ratio_backward_impl')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/get_edit_distance_ratio.jsonl', def_file_ids)

    def test_061_exact_match(self):
        """find_all_references('exact_match'): 1 refs across 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'exact_match')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r.member_tag, 'exact_match')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 1)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/select_qkv_indexes.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['keyword_arg'])
        defs = dynamic_scope_go_to_definition(db, 'exact_match')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/select_qkv_indexes.jsonl', def_file_ids)

    def test_062_read_storage(self):
        """find_all_references('read_storage'): 167 refs across 21 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'read_storage')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 167)
        for r in refs:
            self.assertEqual(r.member_tag, 'read_storage')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 167)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'test/test_attention_vs_traditional.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, 'read_storage')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/merge.jsonl', def_file_ids)

    def test_063_slice_tensor(self):
        """find_all_references('slice_tensor'): 23 refs across 9 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'slice_tensor')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 23)
        for r in refs:
            self.assertEqual(r.member_tag, 'slice_tensor')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 23)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_query_tensor.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['call_pos_arg', 'calls', 'imports'])
        defs = dynamic_scope_go_to_definition(db, 'slice_tensor')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/tensor_util/slice_tensor.jsonl', def_file_ids)

    def test_064__get_storage_path(self):
        """find_all_references('_get_storage_path'): 23 refs across 12 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_get_storage_path')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 23)
        for r in refs:
            self.assertEqual(r.member_tag, '_get_storage_path')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 23)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['fs_util/get_nested_list_file_pathes.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls', 'imports'])
        defs = dynamic_scope_go_to_definition(db, '_get_storage_path')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/tensor_util/slice_view.jsonl', def_file_ids)

    def test_065_read_storage(self):
        """find_all_references('read_storage'): 167 refs across 21 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'read_storage')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 167)
        for r in refs:
            self.assertEqual(r.member_tag, 'read_storage')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 167)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'test/test_attention_vs_traditional.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, 'read_storage')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/st_moe_backward.jsonl', def_file_ids)

    def test_066_get_query_tensor(self):
        """find_all_references('get_query_tensor'): 6 refs across 3 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'get_query_tensor')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 6)
        for r in refs:
            self.assertEqual(r.member_tag, 'get_query_tensor')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 6)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/get_query_tensor.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls', 'imports'])
        defs = dynamic_scope_go_to_definition(db, 'get_query_tensor')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/get_query_tensor.jsonl', def_file_ids)

    def test_067_StMoe(self):
        """find_all_references('StMoe'): 2 refs across 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'StMoe')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r.member_tag, 'StMoe')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 2)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/st_moe.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['attr_value'])
        defs = dynamic_scope_go_to_definition(db, 'StMoe')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/st_moe.jsonl', def_file_ids)

    def test_068_slice_view(self):
        """find_all_references('slice_view'): 67 refs across 15 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'slice_view')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 67)
        for r in refs:
            self.assertEqual(r.member_tag, 'slice_view')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 67)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['call_pos_arg', 'calls', 'imports'])
        defs = dynamic_scope_go_to_definition(db, 'slice_view')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/tensor_util/slice_view.jsonl', def_file_ids)

    def test_069_run_test(self):
        """find_all_references('run_test'): 830 refs across 43 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'run_test')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 830)
        for r in refs:
            self.assertEqual(r.member_tag, 'run_test')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 830)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_causal_attention_mask.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/select_qkv_indexes.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/dump_tensor.jsonl', 'symbolic_tensor/tensor_util/dump_view.jsonl', 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl', 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl', 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', 'symbolic_tensor/tensor_util/pack_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, 'run_test')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/get_causal_attention_mask.jsonl', def_file_ids)

    def test_070_run_test(self):
        """find_all_references('run_test'): 830 refs across 43 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'run_test')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 830)
        for r in refs:
            self.assertEqual(r.member_tag, 'run_test')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 830)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_causal_attention_mask.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/select_qkv_indexes.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/dump_tensor.jsonl', 'symbolic_tensor/tensor_util/dump_view.jsonl', 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl', 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl', 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', 'symbolic_tensor/tensor_util/pack_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, 'run_test')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/slice_view.jsonl', def_file_ids)

    def test_071_SliceAttention(self):
        """find_all_references('SliceAttention'): 4 refs across 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'SliceAttention')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 4)
        for r in refs:
            self.assertEqual(r.member_tag, 'SliceAttention')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 4)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/slice_attention.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['attr_value'])
        defs = dynamic_scope_go_to_definition(db, 'SliceAttention')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/slice_attention.jsonl', def_file_ids)

    def test_072_st_assign(self):
        """find_all_references('st_assign'): 2 refs across 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'st_assign')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r.member_tag, 'st_assign')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 2)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/tensor_util/register_tensor_ops.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['assigns', 'attr_name'])
        defs = dynamic_scope_go_to_definition(db, 'st_assign')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/tensor_util/register_tensor_ops.jsonl', def_file_ids)

    def test_073_backward(self):
        """find_all_references('backward'): 3 refs across 3 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'backward')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 3)
        for r in refs:
            self.assertEqual(r.member_tag, 'backward')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 3)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['attr_name'])
        defs = dynamic_scope_go_to_definition(db, 'backward')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/slice_tensor.jsonl', def_file_ids)

    def test_074_unpack(self):
        """find_all_references('unpack'): 28 refs across 7 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'unpack')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 28)
        for r in refs:
            self.assertEqual(r.member_tag, 'unpack')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 28)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['fs_util/text_merger.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'test/test_attention_vs_traditional.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['assigns', 'attr_name', 'call_pos_arg'])
        defs = dynamic_scope_go_to_definition(db, 'unpack')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/merge.jsonl', def_file_ids)

    def test_075__get_storage_path(self):
        """find_all_references('_get_storage_path'): 23 refs across 12 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_get_storage_path')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 23)
        for r in refs:
            self.assertEqual(r.member_tag, '_get_storage_path')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 23)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['fs_util/get_nested_list_file_pathes.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls', 'imports'])
        defs = dynamic_scope_go_to_definition(db, '_get_storage_path')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/st_stack.jsonl', def_file_ids)

    def test_076_make_none_tensor(self):
        """find_all_references('make_none_tensor'): 23 refs across 10 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'make_none_tensor')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 23)
        for r in refs:
            self.assertEqual(r.member_tag, 'make_none_tensor')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 23)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl', 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls', 'imports'])
        defs = dynamic_scope_go_to_definition(db, 'make_none_tensor')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/tensor_util/make_none_tensor.jsonl', def_file_ids)

    def test_077_slice_tensor(self):
        """find_all_references('slice_tensor'): 23 refs across 9 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'slice_tensor')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 23)
        for r in refs:
            self.assertEqual(r.member_tag, 'slice_tensor')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 23)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_query_tensor.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['call_pos_arg', 'calls', 'imports'])
        defs = dynamic_scope_go_to_definition(db, 'slice_tensor')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/slice_tensor.jsonl', def_file_ids)

    def test_078_merge_forward(self):
        """find_all_references('merge_forward'): 19 refs across 3 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'merge_forward')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 19)
        for r in refs:
            self.assertEqual(r.member_tag, 'merge_forward')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 19)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls', 'imports'])
        defs = dynamic_scope_go_to_definition(db, 'merge_forward')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/merge_forward.jsonl', def_file_ids)

    def test_079_backward(self):
        """find_all_references('backward'): 3 refs across 3 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'backward')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 3)
        for r in refs:
            self.assertEqual(r.member_tag, 'backward')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 3)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['attr_name'])
        defs = dynamic_scope_go_to_definition(db, 'backward')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/st_moe.jsonl', def_file_ids)

    def test_080_backward(self):
        """find_all_references('backward'): 3 refs across 3 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'backward')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 3)
        for r in refs:
            self.assertEqual(r.member_tag, 'backward')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 3)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['attr_name'])
        defs = dynamic_scope_go_to_definition(db, 'backward')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/slice_view.jsonl', def_file_ids)

    def test_081_read_storage(self):
        """find_all_references('read_storage'): 167 refs across 21 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'read_storage')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 167)
        for r in refs:
            self.assertEqual(r.member_tag, 'read_storage')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 167)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'test/test_attention_vs_traditional.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, 'read_storage')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('test/test_gain_st_sgd.jsonl', def_file_ids)

    def test_082_run_test(self):
        """find_all_references('run_test'): 830 refs across 43 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'run_test')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 830)
        for r in refs:
            self.assertEqual(r.member_tag, 'run_test')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 830)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_causal_attention_mask.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/select_qkv_indexes.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/dump_tensor.jsonl', 'symbolic_tensor/tensor_util/dump_view.jsonl', 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl', 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl', 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', 'symbolic_tensor/tensor_util/pack_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, 'run_test')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/tensor_util/make_none_tensor.jsonl', def_file_ids)

    def test_083_forward(self):
        """find_all_references('forward'): 2 refs across 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'forward')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r.member_tag, 'forward')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 2)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/st_copy.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['attr_name'])
        defs = dynamic_scope_go_to_definition(db, 'forward')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/slice_attention.jsonl', def_file_ids)

    def test_084_run_test(self):
        """find_all_references('run_test'): 830 refs across 43 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'run_test')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 830)
        for r in refs:
            self.assertEqual(r.member_tag, 'run_test')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 830)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_causal_attention_mask.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/select_qkv_indexes.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/dump_tensor.jsonl', 'symbolic_tensor/tensor_util/dump_view.jsonl', 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl', 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl', 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', 'symbolic_tensor/tensor_util/pack_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, 'run_test')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/tensor_util/none_tensor_like.jsonl', def_file_ids)

    def test_085_WithDenseViewFunction(self):
        """find_all_references('WithDenseViewFunction'): 3 refs across 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'WithDenseViewFunction')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 3)
        for r in refs:
            self.assertEqual(r.member_tag, 'WithDenseViewFunction')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 3)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/with_dense_view.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['attr_value'])
        defs = dynamic_scope_go_to_definition(db, 'WithDenseViewFunction')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/with_dense_view.jsonl', def_file_ids)

    def test_086__get_coordinates(self):
        """find_all_references('_get_coordinates'): 1 refs across 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_get_coordinates')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r.member_tag, '_get_coordinates')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 1)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/tensor_util/dump_view.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, '_get_coordinates')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/tensor_util/dump_view.jsonl', def_file_ids)

    def test_087_forward(self):
        """find_all_references('forward'): 2 refs across 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'forward')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r.member_tag, 'forward')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 2)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/st_copy.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['attr_name'])
        defs = dynamic_scope_go_to_definition(db, 'forward')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/st_copy.jsonl', def_file_ids)

    def test_088__scalar_slice_indices(self):
        """find_all_references('_scalar_slice_indices'): 5 refs across 5 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_scalar_slice_indices')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 5)
        for r in refs:
            self.assertEqual(r.member_tag, '_scalar_slice_indices')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 5)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, '_scalar_slice_indices')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/coding_agent.jsonl', def_file_ids)

    def test_089_read_storage(self):
        """find_all_references('read_storage'): 167 refs across 21 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'read_storage')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 167)
        for r in refs:
            self.assertEqual(r.member_tag, 'read_storage')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 167)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'test/test_attention_vs_traditional.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, 'read_storage')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/st_attention.jsonl', def_file_ids)

    def test_090__run_all(self):
        """find_all_references('_run_all'): 2 refs across 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_run_all')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r.member_tag, '_run_all')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 2)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['llm_client/coding_agent_task_handler.jsonl', 'llm_client/raw_llm_task_handler.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, '_run_all')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('llm_client/coding_agent_task_handler.jsonl', def_file_ids)

    def test_091__read_storage(self):
        """find_all_references('_read_storage'): 16 refs across 7 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_read_storage')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 16)
        for r in refs:
            self.assertEqual(r.member_tag, '_read_storage')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 16)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'test/test_st_attention_followed_by_st_moe.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls', 'imports'])
        defs = dynamic_scope_go_to_definition(db, '_read_storage')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/merge_backward.jsonl', def_file_ids)

    def test_092_GetEditDistanceRatio(self):
        """find_all_references('GetEditDistanceRatio'): 3 refs across 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'GetEditDistanceRatio')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 3)
        for r in refs:
            self.assertEqual(r.member_tag, 'GetEditDistanceRatio')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 3)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/get_edit_distance_ratio.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['attr_value'])
        defs = dynamic_scope_go_to_definition(db, 'GetEditDistanceRatio')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/get_edit_distance_ratio.jsonl', def_file_ids)

    def test_093__do_one(self):
        """find_all_references('_do_one'): 2 refs across 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_do_one')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r.member_tag, '_do_one')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 2)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['llm_client/coding_agent_task_handler.jsonl', 'llm_client/raw_llm_task_handler.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, '_do_one')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('llm_client/coding_agent_task_handler.jsonl', def_file_ids)

    def test_094_dense_to_sparse(self):
        """find_all_references('dense_to_sparse'): 16 refs across 3 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'dense_to_sparse')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 16)
        for r in refs:
            self.assertEqual(r.member_tag, 'dense_to_sparse')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 16)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls', 'imports'])
        defs = dynamic_scope_go_to_definition(db, 'dense_to_sparse')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/tensor_util/dense_to_sparse.jsonl', def_file_ids)

    def test_095_backward(self):
        """find_all_references('backward'): 3 refs across 3 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'backward')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 3)
        for r in refs:
            self.assertEqual(r.member_tag, 'backward')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 3)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['attr_name'])
        defs = dynamic_scope_go_to_definition(db, 'backward')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/slice_attention.jsonl', def_file_ids)

    def test_096__build_nested_result(self):
        """find_all_references('_build_nested_result'): 7 refs across 3 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_build_nested_result')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 7)
        for r in refs:
            self.assertEqual(r.member_tag, '_build_nested_result')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 7)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls'])
        defs = dynamic_scope_go_to_definition(db, '_build_nested_result')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/coding_agent.jsonl', def_file_ids)

    def test_097_st_fork(self):
        """find_all_references('st_fork'): 2 refs across 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'st_fork')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r.member_tag, 'st_fork')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 2)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/tensor_util/register_tensor_ops.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['assigns', 'attr_name'])
        defs = dynamic_scope_go_to_definition(db, 'st_fork')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/tensor_util/register_tensor_ops.jsonl', def_file_ids)

    def test_098__restore_st_attrs(self):
        """find_all_references('_restore_st_attrs'): 3 refs across 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_restore_st_attrs')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 3)
        for r in refs:
            self.assertEqual(r.member_tag, '_restore_st_attrs')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 3)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['calls', 'imports'])
        defs = dynamic_scope_go_to_definition(db, '_restore_st_attrs')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/slice_view.jsonl', def_file_ids)

    def test_099_default_retrieval_method(self):
        """find_all_references('default_retrieval_method'): 3 refs across 3 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'default_retrieval_method')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 3)
        for r in refs:
            self.assertEqual(r.member_tag, 'default_retrieval_method')
        for r in refs:
            self.assertIn(r.relation_tag, DYNAMIC_RELATION_TAGS)
        self.assertGreaterEqual(len(refs), 3)
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/select_qkv_indexes.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl'])
        ref_rtags = sorted(set(r.relation_tag for r in refs))
        self.assertEqual(ref_rtags, ['bool_op_operand', 'imports'])
        defs = dynamic_scope_go_to_definition(db, 'default_retrieval_method')
        if defs:
            def_file_ids = set(d.file_id for d in defs)
            self.assertIn('symbolic_tensor/function/select_qkv_indexes.jsonl', def_file_ids)


if __name__ == "__main__":
    unittest.main()
