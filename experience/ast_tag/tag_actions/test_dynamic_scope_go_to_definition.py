"""
test_dynamic_scope_go_to_definition: 100 concrete test cases.
Each case: symbol_name -> expected definition file_ids, expected count.
Round-trip: find_all_references includes the usage site.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ast_tag_db import load_jsonl_dataset_into_ast_tag_db, AstTagDB
from relation_tag_classification import DYNAMIC_RELATION_TAGS
from tag_actions.dynamic_scope_go_to_definition import dynamic_scope_go_to_definition
from tag_actions.dynamic_scope_find_all_references import dynamic_scope_find_all_references


def _db() -> AstTagDB:
    if not hasattr(_db, "_instance"):
        dataset_dir = os.path.join(os.path.dirname(__file__), "..", "test_dataset")
        _db._instance = load_jsonl_dataset_into_ast_tag_db(dataset_dir)
    return _db._instance


class TestDynamicScopeGoToDefinition(unittest.TestCase):

    def test_000__replace_last_tensor_with_slice(self):
        """go_to_definition('_replace_last_tensor_with_slice'): 2 defs, used via calls in symbolic_tensor/function/st_moe_backward.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_replace_last_tensor_with_slice')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 2)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '_replace_last_tensor_with_slice')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_replace_last_tensor_with_slice')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/st_moe_backward.jsonl', ref_file_ids)

    def test_001_forward(self):
        """go_to_definition('forward'): 13 defs, used via attr_name in symbolic_tensor/function/get_edit_distance_ratio.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'forward')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 13)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'forward')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'forward')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/get_edit_distance_ratio.jsonl', ref_file_ids)

    def test_002_CodingAgentTaskHandler(self):
        """go_to_definition('CodingAgentTaskHandler'): 1 defs, used via imports in llm_client/task_handler.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'CodingAgentTaskHandler')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'CodingAgentTaskHandler')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['llm_client/coding_agent_task_handler.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'CodingAgentTaskHandler')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('llm_client/task_handler.jsonl', ref_file_ids)

    def test_003_slice_attention_forward(self):
        """go_to_definition('slice_attention_forward'): 1 defs, used via imports in symbolic_tensor/function/slice_attention.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'slice_attention_forward')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'slice_attention_forward')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/slice_attention_forward.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'slice_attention_forward')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/slice_attention.jsonl', ref_file_ids)

    def test_004__write_storage(self):
        """go_to_definition('_write_storage'): 4 defs, used via calls in symbolic_tensor/function/merge_forward.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_write_storage')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 4)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '_write_storage')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_stack.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_write_storage')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/merge_forward.jsonl', ref_file_ids)

    def test_005__read_storage(self):
        """go_to_definition('_read_storage'): 6 defs, used via calls in symbolic_tensor/function/get_edit_distance_ratio.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_read_storage')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 6)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '_read_storage')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_stack.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_read_storage')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/get_edit_distance_ratio.jsonl', ref_file_ids)

    def test_006__reshape_flat_to_nested(self):
        """go_to_definition('_reshape_flat_to_nested'): 1 defs, used via calls in symbolic_tensor/tensor_util/get_diff_tensor.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_reshape_flat_to_nested')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '_reshape_flat_to_nested')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/get_diff_tensor.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_reshape_flat_to_nested')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/tensor_util/get_diff_tensor.jsonl', ref_file_ids)

    def test_007_forward(self):
        """go_to_definition('forward'): 13 defs, used via attr_name in symbolic_tensor/function/get_edit_distance_ratio.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'forward')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 13)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'forward')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'forward')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/get_edit_distance_ratio.jsonl', ref_file_ids)

    def test_008___init__(self):
        """go_to_definition('__init__'): 5 defs, used via attr_name in symbolic_tensor/module/st_moe.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '__init__')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 5)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '__init__')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl'])
        refs = dynamic_scope_find_all_references(db, '__init__')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/module/st_moe.jsonl', ref_file_ids)

    def test_009_read_storage(self):
        """go_to_definition('read_storage'): 23 defs, used via calls in symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'read_storage')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 23)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'read_storage')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'test/test_attention_vs_traditional.jsonl', 'test/test_gain_st_sgd.jsonl', 'test/test_st_attention_followed_by_st_moe.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'read_storage')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', ref_file_ids)

    def test_010_run_test(self):
        """go_to_definition('run_test'): 44 defs, used via calls in symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'run_test')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 44)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'run_test')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_causal_attention_mask.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/select_qkv_indexes.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/dump_tensor.jsonl', 'symbolic_tensor/tensor_util/dump_view.jsonl', 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl', 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl', 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', 'symbolic_tensor/tensor_util/pack_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl', 'test/test_st_attention_followed_by_st_moe.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'run_test')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', ref_file_ids)

    def test_011___init__(self):
        """go_to_definition('__init__'): 5 defs, used via attr_name in symbolic_tensor/module/st_moe.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '__init__')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 5)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '__init__')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl'])
        refs = dynamic_scope_find_all_references(db, '__init__')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/module/st_moe.jsonl', ref_file_ids)

    def test_012_run_test(self):
        """go_to_definition('run_test'): 44 defs, used via calls in symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'run_test')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 44)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'run_test')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_causal_attention_mask.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/select_qkv_indexes.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/dump_tensor.jsonl', 'symbolic_tensor/tensor_util/dump_view.jsonl', 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl', 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl', 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', 'symbolic_tensor/tensor_util/pack_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl', 'test/test_st_attention_followed_by_st_moe.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'run_test')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', ref_file_ids)

    def test_013_run_test(self):
        """go_to_definition('run_test'): 44 defs, used via calls in symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'run_test')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 44)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'run_test')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_causal_attention_mask.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/select_qkv_indexes.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/dump_tensor.jsonl', 'symbolic_tensor/tensor_util/dump_view.jsonl', 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl', 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl', 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', 'symbolic_tensor/tensor_util/pack_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl', 'test/test_st_attention_followed_by_st_moe.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'run_test')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', ref_file_ids)

    def test_014__select_random_indexes_with_none_experience_indexes(self):
        """go_to_definition('_select_random_indexes_with_none_experience_indexes'): 1 defs, used via calls in symbolic_tensor/function/st_moe_backward.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_select_random_indexes_with_none_experience_indexes')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '_select_random_indexes_with_none_experience_indexes')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/st_moe_backward.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_select_random_indexes_with_none_experience_indexes')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/st_moe_backward.jsonl', ref_file_ids)

    def test_015__get_storage_path(self):
        """go_to_definition('_get_storage_path'): 12 defs, used via calls in fs_util/get_nested_list_file_pathes.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_get_storage_path')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 12)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '_get_storage_path')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['fs_util/get_nested_list_file_pathes.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_get_storage_path')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('fs_util/get_nested_list_file_pathes.jsonl', ref_file_ids)

    def test_016_dense_to_sparse(self):
        """go_to_definition('dense_to_sparse'): 1 defs, used via imports in symbolic_tensor/function/with_dense_view.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'dense_to_sparse')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'dense_to_sparse')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/dense_to_sparse.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'dense_to_sparse')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/with_dense_view.jsonl', ref_file_ids)

    def test_017_backward(self):
        """go_to_definition('backward'): 11 defs, used via attr_name in symbolic_tensor/function/get_edit_distance_ratio.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'backward')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 11)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'backward')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'backward')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/get_edit_distance_ratio.jsonl', ref_file_ids)

    def test_018_fork_tensor_backward(self):
        """go_to_definition('fork_tensor_backward'): 1 defs, used via calls in symbolic_tensor/function/fork_tensor.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'fork_tensor_backward')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'fork_tensor_backward')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/fork_tensor.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'fork_tensor_backward')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/fork_tensor.jsonl', ref_file_ids)

    def test_019_SoleFileBatchDataLoader(self):
        """go_to_definition('SoleFileBatchDataLoader'): 1 defs, used via calls in symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'SoleFileBatchDataLoader')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'SoleFileBatchDataLoader')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'SoleFileBatchDataLoader')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', ref_file_ids)

    def test_020_get_causal_attention_mask(self):
        """go_to_definition('get_causal_attention_mask'): 1 defs, used via calls in symbolic_tensor/function/get_causal_attention_mask.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'get_causal_attention_mask')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'get_causal_attention_mask')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/get_causal_attention_mask.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'get_causal_attention_mask')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/get_causal_attention_mask.jsonl', ref_file_ids)

    def test_021_read_storage(self):
        """go_to_definition('read_storage'): 23 defs, used via calls in symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'read_storage')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 23)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'read_storage')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'test/test_attention_vs_traditional.jsonl', 'test/test_gain_st_sgd.jsonl', 'test/test_st_attention_followed_by_st_moe.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'read_storage')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', ref_file_ids)

    def test_022___init__(self):
        """go_to_definition('__init__'): 5 defs, used via attr_name in symbolic_tensor/module/st_moe.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '__init__')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 5)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '__init__')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl'])
        refs = dynamic_scope_find_all_references(db, '__init__')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/module/st_moe.jsonl', ref_file_ids)

    def test_023_default_prompt_for_grad_exp_key(self):
        """go_to_definition('default_prompt_for_grad_exp_key'): 1 defs, used via imports in symbolic_tensor/function/st_moe.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'default_prompt_for_grad_exp_key')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'default_prompt_for_grad_exp_key')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/st_moe_backward.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'default_prompt_for_grad_exp_key')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/st_moe.jsonl', ref_file_ids)

    def test_024__get_store(self):
        """go_to_definition('_get_store'): 1 defs, used via calls in symbolic_tensor/function/symbolic_grad_registry.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_get_store')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '_get_store')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/symbolic_grad_registry.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_get_store')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/symbolic_grad_registry.jsonl', ref_file_ids)

    def test_025__force_todo_nested(self):
        """go_to_definition('_force_todo_nested'): 1 defs, used via calls in symbolic_tensor/function/st_moe_backward.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_force_todo_nested')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '_force_todo_nested')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/st_moe_backward.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_force_todo_nested')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/st_moe_backward.jsonl', ref_file_ids)

    def test_026__get_storage_path(self):
        """go_to_definition('_get_storage_path'): 12 defs, used via calls in fs_util/get_nested_list_file_pathes.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_get_storage_path')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 12)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '_get_storage_path')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['fs_util/get_nested_list_file_pathes.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_get_storage_path')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('fs_util/get_nested_list_file_pathes.jsonl', ref_file_ids)

    def test_027__scalar_slice_indices(self):
        """go_to_definition('_scalar_slice_indices'): 5 defs, used via calls in symbolic_tensor/function/coding_agent.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_scalar_slice_indices')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 5)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '_scalar_slice_indices')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_scalar_slice_indices')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/coding_agent.jsonl', ref_file_ids)

    def test_028__ensure_newline(self):
        """go_to_definition('_ensure_newline'): 1 defs, used via calls in symbolic_tensor/tensor_util/get_diff_tensor.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_ensure_newline')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '_ensure_newline')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/get_diff_tensor.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_ensure_newline')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/tensor_util/get_diff_tensor.jsonl', ref_file_ids)

    def test_029_none_tensor_like(self):
        """go_to_definition('none_tensor_like'): 1 defs, used via imports in symbolic_tensor/function/slice_and_concat_attention_forward.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'none_tensor_like')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'none_tensor_like')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/none_tensor_like.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'none_tensor_like')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', ref_file_ids)

    def test_030__str_to_digit_list(self):
        """go_to_definition('_str_to_digit_list'): 4 defs, used via calls in symbolic_tensor/tensor_util/dump_tensor.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_str_to_digit_list')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 4)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '_str_to_digit_list')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/dump_tensor.jsonl', 'symbolic_tensor/tensor_util/dump_view.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_str_to_digit_list')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/tensor_util/dump_tensor.jsonl', ref_file_ids)

    def test_031_forward(self):
        """go_to_definition('forward'): 13 defs, used via attr_name in symbolic_tensor/function/get_edit_distance_ratio.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'forward')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 13)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'forward')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'forward')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/get_edit_distance_ratio.jsonl', ref_file_ids)

    def test_032_StStack(self):
        """go_to_definition('StStack'): 1 defs, used via attr_value in symbolic_tensor/function/st_stack.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'StStack')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'StStack')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/st_stack.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'StStack')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/st_stack.jsonl', ref_file_ids)

    def test_033_run_test(self):
        """go_to_definition('run_test'): 44 defs, used via calls in symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'run_test')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 44)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'run_test')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_causal_attention_mask.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/select_qkv_indexes.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/dump_tensor.jsonl', 'symbolic_tensor/tensor_util/dump_view.jsonl', 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl', 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl', 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', 'symbolic_tensor/tensor_util/pack_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl', 'test/test_st_attention_followed_by_st_moe.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'run_test')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', ref_file_ids)

    def test_034_forward(self):
        """go_to_definition('forward'): 13 defs, used via attr_name in symbolic_tensor/function/get_edit_distance_ratio.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'forward')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 13)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'forward')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'forward')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/get_edit_distance_ratio.jsonl', ref_file_ids)

    def test_035__get_storage_path(self):
        """go_to_definition('_get_storage_path'): 12 defs, used via calls in fs_util/get_nested_list_file_pathes.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_get_storage_path')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 12)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '_get_storage_path')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['fs_util/get_nested_list_file_pathes.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_get_storage_path')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('fs_util/get_nested_list_file_pathes.jsonl', ref_file_ids)

    def test_036_st_assign(self):
        """go_to_definition('st_assign'): 1 defs, used via attr_name in symbolic_tensor/tensor_util/register_tensor_ops.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'st_assign')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'st_assign')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/register_tensor_ops.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'st_assign')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/tensor_util/register_tensor_ops.jsonl', ref_file_ids)

    def test_037_run_test(self):
        """go_to_definition('run_test'): 44 defs, used via calls in symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'run_test')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 44)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'run_test')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_causal_attention_mask.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/select_qkv_indexes.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/dump_tensor.jsonl', 'symbolic_tensor/tensor_util/dump_view.jsonl', 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl', 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl', 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', 'symbolic_tensor/tensor_util/pack_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl', 'test/test_st_attention_followed_by_st_moe.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'run_test')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', ref_file_ids)

    def test_038__build_nested_result(self):
        """go_to_definition('_build_nested_result'): 3 defs, used via calls in symbolic_tensor/function/coding_agent.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_build_nested_result')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 3)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '_build_nested_result')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_build_nested_result')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/coding_agent.jsonl', ref_file_ids)

    def test_039_unpack(self):
        """go_to_definition('unpack'): 4 defs, used via call_pos_arg in fs_util/text_merger.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'unpack')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 4)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'unpack')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['fs_util/text_merger.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/st_attention.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'unpack')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('fs_util/text_merger.jsonl', ref_file_ids)

    def test_040__sparse_to_dense_impl(self):
        """go_to_definition('_sparse_to_dense_impl'): 1 defs, used via imports in symbolic_tensor/function/with_dense_view.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_sparse_to_dense_impl')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '_sparse_to_dense_impl')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_sparse_to_dense_impl')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/with_dense_view.jsonl', ref_file_ids)

    def test_041_get_last_step_stats(self):
        """go_to_definition('get_last_step_stats'): 1 defs, used via attr_name in symbolic_tensor/optimizer/st_sgd.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'get_last_step_stats')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'get_last_step_stats')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/optimizer/st_sgd.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'get_last_step_stats')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/optimizer/st_sgd.jsonl', ref_file_ids)

    def test_042_run_test(self):
        """go_to_definition('run_test'): 44 defs, used via calls in symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'run_test')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 44)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'run_test')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_causal_attention_mask.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/select_qkv_indexes.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/dump_tensor.jsonl', 'symbolic_tensor/tensor_util/dump_view.jsonl', 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl', 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl', 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', 'symbolic_tensor/tensor_util/pack_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl', 'test/test_st_attention_followed_by_st_moe.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'run_test')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', ref_file_ids)

    def test_043_my_prompt(self):
        """go_to_definition('my_prompt'): 1 defs, used via keyword_arg in symbolic_tensor/function/get_query_tensor.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'my_prompt')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'my_prompt')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/get_query_tensor.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'my_prompt')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/get_query_tensor.jsonl', ref_file_ids)

    def test_044_pop(self):
        """go_to_definition('pop'): 1 defs, used via attr_name in llm_client/coding_agent_query.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'pop')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'pop')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/symbolic_grad_registry.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'pop')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('llm_client/coding_agent_query.jsonl', ref_file_ids)

    def test_045__build_per_dim(self):
        """go_to_definition('_build_per_dim'): 1 defs, used via imports in symbolic_tensor/function/slice_tensor.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_build_per_dim')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '_build_per_dim')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/slice_view.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_build_per_dim')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/slice_tensor.jsonl', ref_file_ids)

    def test_046_storage_path(self):
        """go_to_definition('storage_path'): 1 defs, used via calls in symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'storage_path')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'storage_path')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'storage_path')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', ref_file_ids)

    def test_047_dump_view(self):
        """go_to_definition('dump_view'): 1 defs, used via imports in symbolic_tensor/function/coding_agent.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'dump_view')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'dump_view')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/dump_view.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'dump_view')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/coding_agent.jsonl', ref_file_ids)

    def test_048_st_file_paths(self):
        """go_to_definition('st_file_paths'): 1 defs, used via attr_name in symbolic_tensor/tensor_util/register_tensor_ops.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'st_file_paths')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'st_file_paths')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/register_tensor_ops.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'st_file_paths')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/tensor_util/register_tensor_ops.jsonl', ref_file_ids)

    def test_049_read_storage(self):
        """go_to_definition('read_storage'): 23 defs, used via calls in symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'read_storage')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 23)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'read_storage')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'test/test_attention_vs_traditional.jsonl', 'test/test_gain_st_sgd.jsonl', 'test/test_st_attention_followed_by_st_moe.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'read_storage')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', ref_file_ids)

    def test_050___init__(self):
        """go_to_definition('__init__'): 5 defs, used via attr_name in symbolic_tensor/module/st_moe.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '__init__')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 5)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '__init__')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl'])
        refs = dynamic_scope_find_all_references(db, '__init__')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/module/st_moe.jsonl', ref_file_ids)

    def test_051_TextMerger(self):
        """go_to_definition('TextMerger'): 1 defs, used via imports in symbolic_tensor/function/merge.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'TextMerger')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'TextMerger')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['fs_util/text_merger.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'TextMerger')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/merge.jsonl', ref_file_ids)

    def test_052__st_value_slicer(self):
        """go_to_definition('_st_value_slicer'): 1 defs, used via assigns in symbolic_tensor/tensor_util/register_tensor_ops.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_st_value_slicer')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '_st_value_slicer')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/register_tensor_ops.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_st_value_slicer')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/tensor_util/register_tensor_ops.jsonl', ref_file_ids)

    def test_053_st_pack(self):
        """go_to_definition('st_pack'): 1 defs, used via attr_name in symbolic_tensor/tensor_util/register_tensor_ops.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'st_pack')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'st_pack')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/register_tensor_ops.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'st_pack')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/tensor_util/register_tensor_ops.jsonl', ref_file_ids)

    def test_054_Copy(self):
        """go_to_definition('Copy'): 1 defs, used via attr_value in symbolic_tensor/function/st_copy.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'Copy')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'Copy')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/st_copy.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'Copy')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/st_copy.jsonl', ref_file_ids)

    def test_055_read_storage(self):
        """go_to_definition('read_storage'): 23 defs, used via calls in symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'read_storage')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 23)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'read_storage')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'test/test_attention_vs_traditional.jsonl', 'test/test_gain_st_sgd.jsonl', 'test/test_st_attention_followed_by_st_moe.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'read_storage')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', ref_file_ids)

    def test_056_backward(self):
        """go_to_definition('backward'): 11 defs, used via attr_name in symbolic_tensor/function/get_edit_distance_ratio.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'backward')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 11)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'backward')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'backward')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/get_edit_distance_ratio.jsonl', ref_file_ids)

    def test_057__st_view_slicer(self):
        """go_to_definition('_st_view_slicer'): 1 defs, used via assigns in symbolic_tensor/tensor_util/register_tensor_ops.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_st_view_slicer')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '_st_view_slicer')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/register_tensor_ops.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_st_view_slicer')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/tensor_util/register_tensor_ops.jsonl', ref_file_ids)

    def test_058_step(self):
        """go_to_definition('step'): 1 defs, used via attr_name in symbolic_tensor/optimizer/st_sgd.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'step')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'step')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/optimizer/st_sgd.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'step')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/optimizer/st_sgd.jsonl', ref_file_ids)

    def test_059_zero_first(self):
        """go_to_definition('zero_first'): 1 defs, used via keyword_arg in symbolic_tensor/module/with_dense_view.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'zero_first')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'zero_first')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/module/with_dense_view.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'zero_first')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/module/with_dense_view.jsonl', ref_file_ids)

    def test_060_forward(self):
        """go_to_definition('forward'): 13 defs, used via attr_name in symbolic_tensor/function/get_edit_distance_ratio.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'forward')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 13)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'forward')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'forward')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/get_edit_distance_ratio.jsonl', ref_file_ids)

    def test_061_backward(self):
        """go_to_definition('backward'): 11 defs, used via attr_name in symbolic_tensor/function/get_edit_distance_ratio.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'backward')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 11)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'backward')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'backward')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/get_edit_distance_ratio.jsonl', ref_file_ids)

    def test_062__scalar_slice_indices(self):
        """go_to_definition('_scalar_slice_indices'): 5 defs, used via calls in symbolic_tensor/function/coding_agent.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_scalar_slice_indices')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 5)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '_scalar_slice_indices')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_scalar_slice_indices')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/coding_agent.jsonl', ref_file_ids)

    def test_063_default_prompt_for_grad_exp_value(self):
        """go_to_definition('default_prompt_for_grad_exp_value'): 1 defs, used via imports in symbolic_tensor/function/st_moe.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'default_prompt_for_grad_exp_value')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'default_prompt_for_grad_exp_value')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/st_moe_backward.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'default_prompt_for_grad_exp_value')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/st_moe.jsonl', ref_file_ids)

    def test_064_assign_view(self):
        """go_to_definition('assign_view'): 1 defs, used via imports in symbolic_tensor/function/slice_attention_forward.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'assign_view')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'assign_view')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/assign_view.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'assign_view')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/slice_attention_forward.jsonl', ref_file_ids)

    def test_065_copy_impl(self):
        """go_to_definition('copy_impl'): 1 defs, used via calls in symbolic_tensor/function/st_copy.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'copy_impl')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'copy_impl')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/st_copy.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'copy_impl')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/st_copy.jsonl', ref_file_ids)

    def test_066_SimpleMerger(self):
        """go_to_definition('SimpleMerger'): 3 defs, used via call_pos_arg in symbolic_tensor/function/merge.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'SimpleMerger')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 3)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'SimpleMerger')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/st_attention.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'SimpleMerger')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/merge.jsonl', ref_file_ids)

    def test_067_patch_tensor(self):
        """go_to_definition('patch_tensor'): 1 defs, used via imports in symbolic_tensor/optimizer/st_sgd.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'patch_tensor')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'patch_tensor')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/patch_tensor.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'patch_tensor')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/optimizer/st_sgd.jsonl', ref_file_ids)

    def test_068_st_patch(self):
        """go_to_definition('st_patch'): 1 defs, used via attr_name in symbolic_tensor/tensor_util/patch_tensor.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'st_patch')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'st_patch')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/register_tensor_ops.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'st_patch')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/tensor_util/patch_tensor.jsonl', ref_file_ids)

    def test_069_read_storage(self):
        """go_to_definition('read_storage'): 23 defs, used via calls in symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'read_storage')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 23)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'read_storage')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'test/test_attention_vs_traditional.jsonl', 'test/test_gain_st_sgd.jsonl', 'test/test_st_attention_followed_by_st_moe.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'read_storage')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', ref_file_ids)

    def test_070_forward(self):
        """go_to_definition('forward'): 13 defs, used via attr_name in symbolic_tensor/function/get_edit_distance_ratio.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'forward')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 13)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'forward')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'forward')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/get_edit_distance_ratio.jsonl', ref_file_ids)

    def test_071__write_storage(self):
        """go_to_definition('_write_storage'): 4 defs, used via calls in symbolic_tensor/function/merge_forward.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_write_storage')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 4)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '_write_storage')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_stack.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_write_storage')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/merge_forward.jsonl', ref_file_ids)

    def test_072_GetEditDistanceRatio(self):
        """go_to_definition('GetEditDistanceRatio'): 1 defs, used via attr_value in symbolic_tensor/function/get_edit_distance_ratio.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'GetEditDistanceRatio')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'GetEditDistanceRatio')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/get_edit_distance_ratio.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'GetEditDistanceRatio')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/get_edit_distance_ratio.jsonl', ref_file_ids)

    def test_073_run_test(self):
        """go_to_definition('run_test'): 44 defs, used via calls in symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'run_test')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 44)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'run_test')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_causal_attention_mask.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/select_qkv_indexes.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/dump_tensor.jsonl', 'symbolic_tensor/tensor_util/dump_view.jsonl', 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl', 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl', 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', 'symbolic_tensor/tensor_util/pack_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl', 'test/test_st_attention_followed_by_st_moe.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'run_test')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', ref_file_ids)

    def test_074_read_storage(self):
        """go_to_definition('read_storage'): 23 defs, used via calls in symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'read_storage')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 23)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'read_storage')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'test/test_attention_vs_traditional.jsonl', 'test/test_gain_st_sgd.jsonl', 'test/test_st_attention_followed_by_st_moe.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'read_storage')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', ref_file_ids)

    def test_075_read_storage(self):
        """go_to_definition('read_storage'): 23 defs, used via calls in symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'read_storage')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 23)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'read_storage')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'test/test_attention_vs_traditional.jsonl', 'test/test_gain_st_sgd.jsonl', 'test/test_st_attention_followed_by_st_moe.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'read_storage')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', ref_file_ids)

    def test_076__get_storage_path(self):
        """go_to_definition('_get_storage_path'): 12 defs, used via calls in fs_util/get_nested_list_file_pathes.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_get_storage_path')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 12)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '_get_storage_path')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['fs_util/get_nested_list_file_pathes.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_get_storage_path')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('fs_util/get_nested_list_file_pathes.jsonl', ref_file_ids)

    def test_077_st_stack_forward(self):
        """go_to_definition('st_stack_forward'): 1 defs, used via calls in symbolic_tensor/function/st_stack.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'st_stack_forward')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'st_stack_forward')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/st_stack.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'st_stack_forward')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/st_stack.jsonl', ref_file_ids)

    def test_078__flatten(self):
        """go_to_definition('_flatten'): 1 defs, used via calls in symbolic_tensor/tensor_util/make_tensor.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_flatten')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '_flatten')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/make_tensor.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_flatten')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/tensor_util/make_tensor.jsonl', ref_file_ids)

    def test_079_run_test(self):
        """go_to_definition('run_test'): 44 defs, used via calls in symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'run_test')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 44)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'run_test')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_causal_attention_mask.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/select_qkv_indexes.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/dump_tensor.jsonl', 'symbolic_tensor/tensor_util/dump_view.jsonl', 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl', 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl', 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', 'symbolic_tensor/tensor_util/pack_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl', 'test/test_st_attention_followed_by_st_moe.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'run_test')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', ref_file_ids)

    def test_080__dense_to_sparse_impl(self):
        """go_to_definition('_dense_to_sparse_impl'): 1 defs, used via imports in symbolic_tensor/function/with_dense_view.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_dense_to_sparse_impl')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '_dense_to_sparse_impl')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/dense_to_sparse.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_dense_to_sparse_impl')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/with_dense_view.jsonl', ref_file_ids)

    def test_081_forward(self):
        """go_to_definition('forward'): 13 defs, used via attr_name in symbolic_tensor/function/get_edit_distance_ratio.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'forward')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 13)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'forward')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'forward')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/get_edit_distance_ratio.jsonl', ref_file_ids)

    def test_082__map_nested(self):
        """go_to_definition('_map_nested'): 1 defs, used via calls in symbolic_tensor/tensor_util/todo_tensor_like.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_map_nested')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '_map_nested')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/todo_tensor_like.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_map_nested')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/tensor_util/todo_tensor_like.jsonl', ref_file_ids)

    def test_083_st_moe_backward_grad_input(self):
        """go_to_definition('st_moe_backward_grad_input'): 1 defs, used via calls in symbolic_tensor/function/st_moe_backward.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'st_moe_backward_grad_input')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'st_moe_backward_grad_input')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/st_moe_backward.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'st_moe_backward_grad_input')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/st_moe_backward.jsonl', ref_file_ids)

    def test_084__write_storage(self):
        """go_to_definition('_write_storage'): 4 defs, used via calls in symbolic_tensor/function/merge_forward.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_write_storage')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 4)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '_write_storage')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_stack.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_write_storage')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/merge_forward.jsonl', ref_file_ids)

    def test_085_run_test(self):
        """go_to_definition('run_test'): 44 defs, used via calls in symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'run_test')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 44)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'run_test')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_causal_attention_mask.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/select_qkv_indexes.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/dump_tensor.jsonl', 'symbolic_tensor/tensor_util/dump_view.jsonl', 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl', 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl', 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', 'symbolic_tensor/tensor_util/pack_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl', 'test/test_st_attention_followed_by_st_moe.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'run_test')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', ref_file_ids)

    def test_086_convert_nested_list_coordinates_to_pairs_coordinates(self):
        """go_to_definition('convert_nested_list_coordinates_to_pairs_coordinates'): 1 defs, used via imports in symbolic_tensor/function/st_moe_backward.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'convert_nested_list_coordinates_to_pairs_coordinates')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'convert_nested_list_coordinates_to_pairs_coordinates')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['sparse_util/convert_nested_list_coordinates_to_pairs_coordinates.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'convert_nested_list_coordinates_to_pairs_coordinates')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/st_moe_backward.jsonl', ref_file_ids)

    def test_087__str_to_digit_list(self):
        """go_to_definition('_str_to_digit_list'): 4 defs, used via calls in symbolic_tensor/tensor_util/dump_tensor.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_str_to_digit_list')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 4)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '_str_to_digit_list')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/dump_tensor.jsonl', 'symbolic_tensor/tensor_util/dump_view.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_str_to_digit_list')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/tensor_util/dump_tensor.jsonl', ref_file_ids)

    def test_088__scalar_slice_indices(self):
        """go_to_definition('_scalar_slice_indices'): 5 defs, used via calls in symbolic_tensor/function/coding_agent.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_scalar_slice_indices')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 5)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '_scalar_slice_indices')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_scalar_slice_indices')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/coding_agent.jsonl', ref_file_ids)

    def test_089__normalize_slice_element(self):
        """go_to_definition('_normalize_slice_element'): 2 defs, used via imports in symbolic_tensor/function/slice_tensor.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_normalize_slice_element')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 2)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '_normalize_slice_element')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_normalize_slice_element')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/slice_tensor.jsonl', ref_file_ids)

    def test_090_run_test(self):
        """go_to_definition('run_test'): 44 defs, used via calls in symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'run_test')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 44)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'run_test')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_causal_attention_mask.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/select_qkv_indexes.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/dump_tensor.jsonl', 'symbolic_tensor/tensor_util/dump_view.jsonl', 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl', 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl', 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', 'symbolic_tensor/tensor_util/pack_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl', 'test/test_st_attention_followed_by_st_moe.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'run_test')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', ref_file_ids)

    def test_091_read_out(self):
        """go_to_definition('read_out'): 7 defs, used via calls in symbolic_tensor/function/merge_backward.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'read_out')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 7)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'read_out')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_stack.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'read_out')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/merge_backward.jsonl', ref_file_ids)

    def test_092_read_storage(self):
        """go_to_definition('read_storage'): 23 defs, used via calls in symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'read_storage')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 23)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'read_storage')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'test/test_attention_vs_traditional.jsonl', 'test/test_gain_st_sgd.jsonl', 'test/test_st_attention_followed_by_st_moe.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'read_storage')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', ref_file_ids)

    def test_093__get_storage_path(self):
        """go_to_definition('_get_storage_path'): 12 defs, used via calls in fs_util/get_nested_list_file_pathes.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_get_storage_path')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 12)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '_get_storage_path')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['fs_util/get_nested_list_file_pathes.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_get_storage_path')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('fs_util/get_nested_list_file_pathes.jsonl', ref_file_ids)

    def test_094_default_prompt_for_fork_grad_input(self):
        """go_to_definition('default_prompt_for_fork_grad_input'): 1 defs, used via bool_op_operand in symbolic_tensor/function/fork_tensor.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'default_prompt_for_fork_grad_input')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'default_prompt_for_fork_grad_input')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/fork_tensor.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'default_prompt_for_fork_grad_input')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/fork_tensor.jsonl', ref_file_ids)

    def test_095_unpack(self):
        """go_to_definition('unpack'): 4 defs, used via call_pos_arg in fs_util/text_merger.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'unpack')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 4)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'unpack')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['fs_util/text_merger.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/st_attention.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'unpack')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('fs_util/text_merger.jsonl', ref_file_ids)

    def test_096_forward(self):
        """go_to_definition('forward'): 13 defs, used via attr_name in symbolic_tensor/function/get_edit_distance_ratio.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'forward')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 13)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'forward')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'forward')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/get_edit_distance_ratio.jsonl', ref_file_ids)

    def test_097__unflatten(self):
        """go_to_definition('_unflatten'): 2 defs, used via calls in symbolic_tensor/function/get_edit_distance_ratio.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_unflatten')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 2)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, '_unflatten')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_unflatten')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/get_edit_distance_ratio.jsonl', ref_file_ids)

    def test_098_select_qkv_indexes(self):
        """go_to_definition('select_qkv_indexes'): 1 defs, used via calls in symbolic_tensor/function/select_qkv_indexes.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'select_qkv_indexes')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'select_qkv_indexes')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/select_qkv_indexes.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'select_qkv_indexes')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/select_qkv_indexes.jsonl', ref_file_ids)

    def test_099_backward(self):
        """go_to_definition('backward'): 11 defs, used via attr_name in symbolic_tensor/function/get_edit_distance_ratio.jsonl"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'backward')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 11)
        for d in defs:
            self.assertEqual(d.relation_tag, "defines")
            self.assertEqual(d.member_tag, 'backward')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'backward')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/get_edit_distance_ratio.jsonl', ref_file_ids)


if __name__ == "__main__":
    unittest.main()
