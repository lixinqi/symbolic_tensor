"""
test_dynamic_scope_find_all_references: 100 concrete test cases.
Each case: symbol_name -> expected reference file_ids, expected count.
Round-trip: go_to_definition confirms the symbol exists.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ast_tag_db import AstTagDB
from relation_tag_classification import DYNAMIC_RELATION_TAGS
from tag_actions.dynamic_scope_find_all_references import dynamic_scope_find_all_references
from tag_actions.dynamic_scope_go_to_definition import dynamic_scope_go_to_definition


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


class TestDynamicScopeFindAllReferences(unittest.TestCase):

    def test_000__flat_index_from_coordinates(self):
        """find_all_references('_flat_index_from_coordinates'): 1 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_flat_index_from_coordinates')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r.member_tag, '_flat_index_from_coordinates')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/tensor_util/dump_view.jsonl'])
        defs = dynamic_scope_go_to_definition(db, '_flat_index_from_coordinates')
        self.assertGreaterEqual(len(defs), 1)

    def test_001_assign_view(self):
        """find_all_references('assign_view'): 9 refs in 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'assign_view')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 9)
        for r in refs:
            self.assertEqual(r.member_tag, 'assign_view')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'assign_view')
        self.assertGreaterEqual(len(defs), 1)

    def test_002_SparseToDense(self):
        """find_all_references('SparseToDense'): 1 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'SparseToDense')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r.member_tag, 'SparseToDense')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'SparseToDense')
        self.assertGreaterEqual(len(defs), 1)

    def test_003_read_out(self):
        """find_all_references('read_out'): 131 refs in 7 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'read_out')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 131)
        for r in refs:
            self.assertEqual(r.member_tag, 'read_out')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_stack.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'read_out')
        self.assertGreaterEqual(len(defs), 1)

    def test_004_Optional(self):
        """find_all_references('Optional'): 72 refs in 24 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'Optional')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 72)
        for r in refs:
            self.assertEqual(r.member_tag, 'Optional')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['llm_client/coding_agent_query.jsonl', 'llm_client/coding_agent_task_handler.jsonl', 'llm_client/raw_llm_query.jsonl', 'llm_client/raw_llm_task_handler.jsonl', 'llm_client/task_handler.jsonl', 'symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_query_tensor.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/select_qkv_indexes.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'Optional')
        self.assertGreaterEqual(len(defs), 1)

    def test_005_slice_backward(self):
        """find_all_references('slice_backward'): 7 refs in 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'slice_backward')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 7)
        for r in refs:
            self.assertEqual(r.member_tag, 'slice_backward')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'slice_backward')
        self.assertGreaterEqual(len(defs), 1)

    def test_006_merge_forward(self):
        """find_all_references('merge_forward'): 17 refs in 3 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'merge_forward')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 17)
        for r in refs:
            self.assertEqual(r.member_tag, 'merge_forward')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'merge_forward')
        self.assertGreaterEqual(len(defs), 1)

    def test_007___init__(self):
        """find_all_references('__init__'): 3 refs in 3 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '__init__')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 3)
        for r in refs:
            self.assertEqual(r.member_tag, '__init__')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl'])
        defs = dynamic_scope_go_to_definition(db, '__init__')
        self.assertGreaterEqual(len(defs), 1)

    def test_008_merge_backward(self):
        """find_all_references('merge_backward'): 8 refs in 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'merge_backward')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 8)
        for r in refs:
            self.assertEqual(r.member_tag, 'merge_backward')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'merge_backward')
        self.assertGreaterEqual(len(defs), 1)

    def test_009__PLAIN(self):
        """find_all_references('_PLAIN'): 6 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_PLAIN')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 6)
        for r in refs:
            self.assertEqual(r.member_tag, '_PLAIN')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/st_moe_backward.jsonl'])
        defs = dynamic_scope_go_to_definition(db, '_PLAIN')
        self.assertGreaterEqual(len(defs), 1)

    def test_010_seedir(self):
        """find_all_references('seedir'): 1 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'seedir')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r.member_tag, 'seedir')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['fs_util/pack_dir.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'seedir')
        self.assertGreaterEqual(len(defs), 1)

    def test_011_st_patch(self):
        """find_all_references('st_patch'): 3 refs in 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'st_patch')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 3)
        for r in refs:
            self.assertEqual(r.member_tag, 'st_patch')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'st_patch')
        self.assertGreaterEqual(len(defs), 1)

    def test_012_threading(self):
        """find_all_references('threading'): 1 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'threading')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r.member_tag, 'threading')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/symbolic_grad_registry.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'threading')
        self.assertGreaterEqual(len(defs), 1)

    def test_013_empty_tensor_like(self):
        """find_all_references('empty_tensor_like'): 5 refs in 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'empty_tensor_like')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 5)
        for r in refs:
            self.assertEqual(r.member_tag, 'empty_tensor_like')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'empty_tensor_like')
        self.assertGreaterEqual(len(defs), 1)

    def test_014_patch_tensor(self):
        """find_all_references('patch_tensor'): 8 refs in 4 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'patch_tensor')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 8)
        for r in refs:
            self.assertEqual(r.member_tag, 'patch_tensor')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'patch_tensor')
        self.assertGreaterEqual(len(defs), 1)

    def test_015__assert_is_dumped_tensor_dir(self):
        """find_all_references('_assert_is_dumped_tensor_dir'): 1 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_assert_is_dumped_tensor_dir')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r.member_tag, '_assert_is_dumped_tensor_dir')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/tensor_util/load_tensor.jsonl'])
        defs = dynamic_scope_go_to_definition(db, '_assert_is_dumped_tensor_dir')
        self.assertGreaterEqual(len(defs), 1)

    def test_016_fork_tensor_forward(self):
        """find_all_references('fork_tensor_forward'): 6 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'fork_tensor_forward')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 6)
        for r in refs:
            self.assertEqual(r.member_tag, 'fork_tensor_forward')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/fork_tensor.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'fork_tensor_forward')
        self.assertGreaterEqual(len(defs), 1)

    def test_017_defaultdict(self):
        """find_all_references('defaultdict'): 1 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'defaultdict')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r.member_tag, 'defaultdict')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['sparse_util/transpose_pairs_coordinates.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'defaultdict')
        self.assertGreaterEqual(len(defs), 1)

    def test_018_with_dense_view(self):
        """find_all_references('with_dense_view'): 6 refs in 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'with_dense_view')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 6)
        for r in refs:
            self.assertEqual(r.member_tag, 'with_dense_view')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'with_dense_view')
        self.assertGreaterEqual(len(defs), 1)

    def test_019__check_file_path(self):
        """find_all_references('_check_file_path'): 1 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_check_file_path')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r.member_tag, '_check_file_path')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/tensor_util/todo_tensor_like.jsonl'])
        defs = dynamic_scope_go_to_definition(db, '_check_file_path')
        self.assertGreaterEqual(len(defs), 1)

    def test_020__expand(self):
        """find_all_references('_expand'): 2 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_expand')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r.member_tag, '_expand')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/tensor_util/empty_tensor_like.jsonl'])
        defs = dynamic_scope_go_to_definition(db, '_expand')
        self.assertGreaterEqual(len(defs), 1)

    def test_021_dump_tensor(self):
        """find_all_references('dump_tensor'): 7 refs in 3 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'dump_tensor')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 7)
        for r in refs:
            self.assertEqual(r.member_tag, 'dump_tensor')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/tensor_util/dump_tensor.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'dump_tensor')
        self.assertGreaterEqual(len(defs), 1)

    def test_022_forward(self):
        """find_all_references('forward'): 2 refs in 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'forward')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r.member_tag, 'forward')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/st_copy.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'forward')
        self.assertGreaterEqual(len(defs), 1)

    def test_023_pack_tensor(self):
        """find_all_references('pack_tensor'): 5 refs in 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'pack_tensor')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 5)
        for r in refs:
            self.assertEqual(r.member_tag, 'pack_tensor')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/tensor_util/pack_tensor.jsonl', 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'pack_tensor')
        self.assertGreaterEqual(len(defs), 1)

    def test_024_subprocess(self):
        """find_all_references('subprocess'): 13 refs in 13 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'subprocess')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 13)
        for r in refs:
            self.assertEqual(r.member_tag, 'subprocess')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/get_query_tensor.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'subprocess')
        self.assertGreaterEqual(len(defs), 1)

    def test_025_parameters(self):
        """find_all_references('parameters'): 2 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'parameters')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r.member_tag, 'parameters')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/module/st_moe.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'parameters')
        self.assertGreaterEqual(len(defs), 1)

    def test_026__save_st_attrs(self):
        """find_all_references('_save_st_attrs'): 2 refs in 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_save_st_attrs')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r.member_tag, '_save_st_attrs')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl'])
        defs = dynamic_scope_go_to_definition(db, '_save_st_attrs')
        self.assertGreaterEqual(len(defs), 1)

    def test_027_SoleFileBatchDataLoader(self):
        """find_all_references('SoleFileBatchDataLoader'): 6 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'SoleFileBatchDataLoader')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 6)
        for r in refs:
            self.assertEqual(r.member_tag, 'SoleFileBatchDataLoader')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'SoleFileBatchDataLoader')
        self.assertGreaterEqual(len(defs), 1)

    def test_028_Path(self):
        """find_all_references('Path'): 19 refs in 8 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'Path')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 19)
        for r in refs:
            self.assertEqual(r.member_tag, 'Path')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['fs_util/get_nested_list_file_pathes.jsonl', 'symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'Path')
        self.assertGreaterEqual(len(defs), 1)

    def test_029_zero_grad(self):
        """find_all_references('zero_grad'): 3 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'zero_grad')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 3)
        for r in refs:
            self.assertEqual(r.member_tag, 'zero_grad')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/optimizer/st_sgd.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'zero_grad')
        self.assertGreaterEqual(len(defs), 1)

    def test_030__unzip_to_tensor_list(self):
        """find_all_references('_unzip_to_tensor_list'): 1 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_unzip_to_tensor_list')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r.member_tag, '_unzip_to_tensor_list')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/select_qkv_indexes.jsonl'])
        defs = dynamic_scope_go_to_definition(db, '_unzip_to_tensor_list')
        self.assertGreaterEqual(len(defs), 1)

    def test_031__read_storage(self):
        """find_all_references('_read_storage'): 15 refs in 6 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_read_storage')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 15)
        for r in refs:
            self.assertEqual(r.member_tag, '_read_storage')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_stack.jsonl'])
        defs = dynamic_scope_go_to_definition(db, '_read_storage')
        self.assertGreaterEqual(len(defs), 1)

    def test_032__coords_to_flat(self):
        """find_all_references('_coords_to_flat'): 6 refs in 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_coords_to_flat')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 6)
        for r in refs:
            self.assertEqual(r.member_tag, '_coords_to_flat')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_stack.jsonl'])
        defs = dynamic_scope_go_to_definition(db, '_coords_to_flat')
        self.assertGreaterEqual(len(defs), 1)

    def test_033_AsyncOpenAI(self):
        """find_all_references('AsyncOpenAI'): 1 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'AsyncOpenAI')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r.member_tag, 'AsyncOpenAI')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['llm_client/raw_llm_query.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'AsyncOpenAI')
        self.assertGreaterEqual(len(defs), 1)

    def test_034_st_stack_forward(self):
        """find_all_references('st_stack_forward'): 12 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'st_stack_forward')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 12)
        for r in refs:
            self.assertEqual(r.member_tag, 'st_stack_forward')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/st_stack.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'st_stack_forward')
        self.assertGreaterEqual(len(defs), 1)

    def test_035__read_flat_data(self):
        """find_all_references('_read_flat_data'): 1 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_read_flat_data')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r.member_tag, '_read_flat_data')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/tensor_util/load_tensor.jsonl'])
        defs = dynamic_scope_go_to_definition(db, '_read_flat_data')
        self.assertGreaterEqual(len(defs), 1)

    def test_036_StSGD(self):
        """find_all_references('StSGD'): 8 refs in 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'StSGD')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 8)
        for r in refs:
            self.assertEqual(r.member_tag, 'StSGD')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'StSGD')
        self.assertGreaterEqual(len(defs), 1)

    def test_037_SliceTensor(self):
        """find_all_references('SliceTensor'): 2 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'SliceTensor')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r.member_tag, 'SliceTensor')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/slice_tensor.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'SliceTensor')
        self.assertGreaterEqual(len(defs), 1)

    def test_038_Union(self):
        """find_all_references('Union'): 20 refs in 12 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'Union')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 20)
        for r in refs:
            self.assertEqual(r.member_tag, 'Union')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['fs_util/get_nested_list_file_pathes.jsonl', 'llm_client/agent_task.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'Union')
        self.assertGreaterEqual(len(defs), 1)

    def test_039_default_prompt_for_grad_input_frame(self):
        """find_all_references('default_prompt_for_grad_input_frame'): 1 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'default_prompt_for_grad_input_frame')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r.member_tag, 'default_prompt_for_grad_input_frame')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/st_moe_backward.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'default_prompt_for_grad_input_frame')
        self.assertGreaterEqual(len(defs), 1)

    def test_040_get_query_tensor(self):
        """find_all_references('get_query_tensor'): 4 refs in 3 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'get_query_tensor')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 4)
        for r in refs:
            self.assertEqual(r.member_tag, 'get_query_tensor')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/get_query_tensor.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'get_query_tensor')
        self.assertGreaterEqual(len(defs), 1)

    def test_041__replace_last_tensor_with_full_slice(self):
        """find_all_references('_replace_last_tensor_with_full_slice'): 2 refs in 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_replace_last_tensor_with_full_slice')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r.member_tag, '_replace_last_tensor_with_full_slice')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl'])
        defs = dynamic_scope_go_to_definition(db, '_replace_last_tensor_with_full_slice')
        self.assertGreaterEqual(len(defs), 1)

    def test_042_default_prompt_for_query(self):
        """find_all_references('default_prompt_for_query'): 1 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'default_prompt_for_query')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r.member_tag, 'default_prompt_for_query')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/get_query_tensor.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'default_prompt_for_query')
        self.assertGreaterEqual(len(defs), 1)

    def test_043_SimpleMerger(self):
        """find_all_references('SimpleMerger'): 3 refs in 3 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'SimpleMerger')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 3)
        for r in refs:
            self.assertEqual(r.member_tag, 'SimpleMerger')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/st_attention.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'SimpleMerger')
        self.assertGreaterEqual(len(defs), 1)

    def test_044_slice_attention_forward(self):
        """find_all_references('slice_attention_forward'): 16 refs in 3 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'slice_attention_forward')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 16)
        for r in refs:
            self.assertEqual(r.member_tag, 'slice_attention_forward')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'slice_attention_forward')
        self.assertGreaterEqual(len(defs), 1)

    def test_045_uuid(self):
        """find_all_references('uuid'): 2 refs in 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'uuid')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r.member_tag, 'uuid')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/tensor_util/make_none_tensor.jsonl', 'symbolic_tensor/tensor_util/none_tensor_like.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'uuid')
        self.assertGreaterEqual(len(defs), 1)

    def test_046__get_storage_path(self):
        """find_all_references('_get_storage_path'): 22 refs in 12 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_get_storage_path')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 22)
        for r in refs:
            self.assertEqual(r.member_tag, '_get_storage_path')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['fs_util/get_nested_list_file_pathes.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        defs = dynamic_scope_go_to_definition(db, '_get_storage_path')
        self.assertGreaterEqual(len(defs), 1)

    def test_047__flat_index_to_coords(self):
        """find_all_references('_flat_index_to_coords'): 1 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_flat_index_to_coords')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r.member_tag, '_flat_index_to_coords')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/tensor_util/make_tensor.jsonl'])
        defs = dynamic_scope_go_to_definition(db, '_flat_index_to_coords')
        self.assertGreaterEqual(len(defs), 1)

    def test_048_ClaudeAgentOptions(self):
        """find_all_references('ClaudeAgentOptions'): 1 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'ClaudeAgentOptions')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r.member_tag, 'ClaudeAgentOptions')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['llm_client/coding_agent_query.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'ClaudeAgentOptions')
        self.assertGreaterEqual(len(defs), 1)

    def test_049_symbolic_grad_registry(self):
        """find_all_references('symbolic_grad_registry'): 20 refs in 10 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'symbolic_grad_registry')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 20)
        for r in refs:
            self.assertEqual(r.member_tag, 'symbolic_grad_registry')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'symbolic_grad_registry')
        self.assertGreaterEqual(len(defs), 1)

    def test_050__scalar_slice_indices(self):
        """find_all_references('_scalar_slice_indices'): 5 refs in 5 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_scalar_slice_indices')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 5)
        for r in refs:
            self.assertEqual(r.member_tag, '_scalar_slice_indices')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl'])
        defs = dynamic_scope_go_to_definition(db, '_scalar_slice_indices')
        self.assertGreaterEqual(len(defs), 1)

    def test_051_st_moe_backward_grad_experience(self):
        """find_all_references('st_moe_backward_grad_experience'): 1 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'st_moe_backward_grad_experience')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r.member_tag, 'st_moe_backward_grad_experience')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/st_moe_backward.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'st_moe_backward_grad_experience')
        self.assertGreaterEqual(len(defs), 1)

    def test_052_get_causal_attention_mask(self):
        """find_all_references('get_causal_attention_mask'): 8 refs in 3 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'get_causal_attention_mask')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 8)
        for r in refs:
            self.assertEqual(r.member_tag, 'get_causal_attention_mask')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/get_causal_attention_mask.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'get_causal_attention_mask')
        self.assertGreaterEqual(len(defs), 1)

    def test_053__flatten_nested(self):
        """find_all_references('_flatten_nested'): 4 refs in 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_flatten_nested')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 4)
        for r in refs:
            self.assertEqual(r.member_tag, '_flatten_nested')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['llm_client/coding_agent_task_handler.jsonl', 'llm_client/raw_llm_task_handler.jsonl'])
        defs = dynamic_scope_go_to_definition(db, '_flatten_nested')
        self.assertGreaterEqual(len(defs), 1)

    def test_054_fork_tensor_backward(self):
        """find_all_references('fork_tensor_backward'): 3 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'fork_tensor_backward')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 3)
        for r in refs:
            self.assertEqual(r.member_tag, 'fork_tensor_backward')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/fork_tensor.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'fork_tensor_backward')
        self.assertGreaterEqual(len(defs), 1)

    def test_055_query(self):
        """find_all_references('query'): 1 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'query')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r.member_tag, 'query')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['llm_client/coding_agent_query.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'query')
        self.assertGreaterEqual(len(defs), 1)

    def test_056_st_assign(self):
        """find_all_references('st_assign'): 2 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'st_assign')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r.member_tag, 'st_assign')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/tensor_util/register_tensor_ops.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'st_assign')
        self.assertGreaterEqual(len(defs), 1)

    def test_057__get_shape(self):
        """find_all_references('_get_shape'): 2 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_get_shape')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r.member_tag, '_get_shape')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/tensor_util/make_tensor.jsonl'])
        defs = dynamic_scope_go_to_definition(db, '_get_shape')
        self.assertGreaterEqual(len(defs), 1)

    def test_058_fork_tensor(self):
        """find_all_references('fork_tensor'): 2 refs in 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'fork_tensor')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r.member_tag, 'fork_tensor')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'fork_tensor')
        self.assertGreaterEqual(len(defs), 1)

    def test_059_Levenshtein(self):
        """find_all_references('Levenshtein'): 1 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'Levenshtein')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r.member_tag, 'Levenshtein')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/get_edit_distance_ratio.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'Levenshtein')
        self.assertGreaterEqual(len(defs), 1)

    def test_060__sparse_to_dense_impl(self):
        """find_all_references('_sparse_to_dense_impl'): 3 refs in 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_sparse_to_dense_impl')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 3)
        for r in refs:
            self.assertEqual(r.member_tag, '_sparse_to_dense_impl')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        defs = dynamic_scope_go_to_definition(db, '_sparse_to_dense_impl')
        self.assertGreaterEqual(len(defs), 1)

    def test_061__filter_last_coordinate_eq_zero(self):
        """find_all_references('_filter_last_coordinate_eq_zero'): 1 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_filter_last_coordinate_eq_zero')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r.member_tag, '_filter_last_coordinate_eq_zero')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/select_qkv_indexes.jsonl'])
        defs = dynamic_scope_go_to_definition(db, '_filter_last_coordinate_eq_zero')
        self.assertGreaterEqual(len(defs), 1)

    def test_062__flatten_nested_paths(self):
        """find_all_references('_flatten_nested_paths'): 3 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_flatten_nested_paths')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 3)
        for r in refs:
            self.assertEqual(r.member_tag, '_flatten_nested_paths')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/optimizer/st_sgd.jsonl'])
        defs = dynamic_scope_go_to_definition(db, '_flatten_nested_paths')
        self.assertGreaterEqual(len(defs), 1)

    def test_063_storage_path(self):
        """find_all_references('storage_path'): 7 refs in 4 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'storage_path')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 7)
        for r in refs:
            self.assertEqual(r.member_tag, 'storage_path')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/get_query_tensor.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'storage_path')
        self.assertGreaterEqual(len(defs), 1)

    def test_064__is_leaf(self):
        """find_all_references('_is_leaf'): 4 refs in 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_is_leaf')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 4)
        for r in refs:
            self.assertEqual(r.member_tag, '_is_leaf')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['sparse_util/convert_nested_list_coordinates_to_pairs_coordinates.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl'])
        defs = dynamic_scope_go_to_definition(db, '_is_leaf')
        self.assertGreaterEqual(len(defs), 1)

    def test_065_get_last_step_stats(self):
        """find_all_references('get_last_step_stats'): 3 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'get_last_step_stats')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 3)
        for r in refs:
            self.assertEqual(r.member_tag, 'get_last_step_stats')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/optimizer/st_sgd.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'get_last_step_stats')
        self.assertGreaterEqual(len(defs), 1)

    def test_066_Callable(self):
        """find_all_references('Callable'): 33 refs in 14 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'Callable')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 33)
        for r in refs:
            self.assertEqual(r.member_tag, 'Callable')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_query_tensor.jsonl', 'symbolic_tensor/function/select_qkv_indexes.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'Callable')
        self.assertGreaterEqual(len(defs), 1)

    def test_067_convert_nested_list_coordinates_to_pairs_coordinates(self):
        """find_all_references('convert_nested_list_coordinates_to_pairs_coordinates'): 2 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'convert_nested_list_coordinates_to_pairs_coordinates')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r.member_tag, 'convert_nested_list_coordinates_to_pairs_coordinates')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/st_moe_backward.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'convert_nested_list_coordinates_to_pairs_coordinates')
        self.assertGreaterEqual(len(defs), 1)

    def test_068__st_view_slicer(self):
        """find_all_references('_st_view_slicer'): 1 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_st_view_slicer')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r.member_tag, '_st_view_slicer')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/tensor_util/register_tensor_ops.jsonl'])
        defs = dynamic_scope_go_to_definition(db, '_st_view_slicer')
        self.assertGreaterEqual(len(defs), 1)

    def test_069__assert_consistent_shape(self):
        """find_all_references('_assert_consistent_shape'): 2 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_assert_consistent_shape')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r.member_tag, '_assert_consistent_shape')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/tensor_util/make_tensor.jsonl'])
        defs = dynamic_scope_go_to_definition(db, '_assert_consistent_shape')
        self.assertGreaterEqual(len(defs), 1)

    def test_070_time(self):
        """find_all_references('time'): 4 refs in 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'time')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 4)
        for r in refs:
            self.assertEqual(r.member_tag, 'time')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['llm_client/task_handler.jsonl', 'test/test_transform_method_time_comparison.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'time')
        self.assertGreaterEqual(len(defs), 1)

    def test_071__select_random_indexes_with_none_experience_indexes(self):
        """find_all_references('_select_random_indexes_with_none_experience_indexes'): 2 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_select_random_indexes_with_none_experience_indexes')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r.member_tag, '_select_random_indexes_with_none_experience_indexes')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/st_moe_backward.jsonl'])
        defs = dynamic_scope_go_to_definition(db, '_select_random_indexes_with_none_experience_indexes')
        self.assertGreaterEqual(len(defs), 1)

    def test_072__grep_by_file_content_hint(self):
        """find_all_references('_grep_by_file_content_hint'): 1 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_grep_by_file_content_hint')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r.member_tag, '_grep_by_file_content_hint')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['llm_client/raw_llm_task_handler.jsonl'])
        defs = dynamic_scope_go_to_definition(db, '_grep_by_file_content_hint')
        self.assertGreaterEqual(len(defs), 1)

    def test_073_st_pack(self):
        """find_all_references('st_pack'): 2 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'st_pack')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r.member_tag, 'st_pack')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/tensor_util/register_tensor_ops.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'st_pack')
        self.assertGreaterEqual(len(defs), 1)

    def test_074__restore_st_attrs(self):
        """find_all_references('_restore_st_attrs'): 2 refs in 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_restore_st_attrs')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r.member_tag, '_restore_st_attrs')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl'])
        defs = dynamic_scope_go_to_definition(db, '_restore_st_attrs')
        self.assertGreaterEqual(len(defs), 1)

    def test_075_slice_attention_backward(self):
        """find_all_references('slice_attention_backward'): 5 refs in 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'slice_attention_backward')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 5)
        for r in refs:
            self.assertEqual(r.member_tag, 'slice_attention_backward')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'slice_attention_backward')
        self.assertGreaterEqual(len(defs), 1)

    def test_076__get_coordinates(self):
        """find_all_references('_get_coordinates'): 1 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_get_coordinates')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r.member_tag, '_get_coordinates')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/tensor_util/dump_view.jsonl'])
        defs = dynamic_scope_go_to_definition(db, '_get_coordinates')
        self.assertGreaterEqual(len(defs), 1)

    def test_077_Type(self):
        """find_all_references('Type'): 4 refs in 4 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'Type')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 4)
        for r in refs:
            self.assertEqual(r.member_tag, 'Type')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/st_attention.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'Type')
        self.assertGreaterEqual(len(defs), 1)

    def test_078_slice_tensor(self):
        """find_all_references('slice_tensor'): 15 refs in 9 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'slice_tensor')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 15)
        for r in refs:
            self.assertEqual(r.member_tag, 'slice_tensor')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_query_tensor.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'slice_tensor')
        self.assertGreaterEqual(len(defs), 1)

    def test_079_itertools(self):
        """find_all_references('itertools'): 21 refs in 18 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'itertools')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 21)
        for r in refs:
            self.assertEqual(r.member_tag, 'itertools')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['fs_util/get_nested_list_file_pathes.jsonl', 'symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_query_tensor.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dump_view.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'itertools')
        self.assertGreaterEqual(len(defs), 1)

    def test_080_Any(self):
        """find_all_references('Any'): 9 refs in 5 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'Any')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 9)
        for r in refs:
            self.assertEqual(r.member_tag, 'Any')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['sparse_util/convert_nested_list_coordinates_to_pairs_coordinates.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/module/st_moe.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'Any')
        self.assertGreaterEqual(len(defs), 1)

    def test_081_st_assign_view(self):
        """find_all_references('st_assign_view'): 2 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'st_assign_view')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r.member_tag, 'st_assign_view')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/tensor_util/register_tensor_ops.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'st_assign_view')
        self.assertGreaterEqual(len(defs), 1)

    def test_082_frame_to_str(self):
        """find_all_references('frame_to_str'): 1 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'frame_to_str')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r.member_tag, 'frame_to_str')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['fs_util/text_merger.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'frame_to_str')
        self.assertGreaterEqual(len(defs), 1)

    def test_083_tempfile(self):
        """find_all_references('tempfile'): 271 refs in 44 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'tempfile')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 271)
        for r in refs:
            self.assertEqual(r.member_tag, 'tempfile')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/get_query_tensor.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/select_qkv_indexes.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/dump_tensor.jsonl', 'symbolic_tensor/tensor_util/dump_view.jsonl', 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl', 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl', 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', 'symbolic_tensor/tensor_util/pack_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl', 'test/test_transform_method_time_comparison.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'tempfile')
        self.assertGreaterEqual(len(defs), 1)

    def test_084_assign_tensor(self):
        """find_all_references('assign_tensor'): 19 refs in 10 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'assign_tensor')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 19)
        for r in refs:
            self.assertEqual(r.member_tag, 'assign_tensor')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'assign_tensor')
        self.assertGreaterEqual(len(defs), 1)

    def test_085_default_prompt_for_grad_exp_value(self):
        """find_all_references('default_prompt_for_grad_exp_value'): 1 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'default_prompt_for_grad_exp_value')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r.member_tag, 'default_prompt_for_grad_exp_value')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/st_moe_backward.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'default_prompt_for_grad_exp_value')
        self.assertGreaterEqual(len(defs), 1)

    def test_086_dense_to_sparse(self):
        """find_all_references('dense_to_sparse'): 12 refs in 3 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'dense_to_sparse')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 12)
        for r in refs:
            self.assertEqual(r.member_tag, 'dense_to_sparse')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'dense_to_sparse')
        self.assertGreaterEqual(len(defs), 1)

    def test_087__extract_coordinates(self):
        """find_all_references('_extract_coordinates'): 1 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_extract_coordinates')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r.member_tag, '_extract_coordinates')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/select_qkv_indexes.jsonl'])
        defs = dynamic_scope_go_to_definition(db, '_extract_coordinates')
        self.assertGreaterEqual(len(defs), 1)

    def test_088_make_tensor(self):
        """find_all_references('make_tensor'): 316 refs in 43 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'make_tensor')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 316)
        for r in refs:
            self.assertEqual(r.member_tag, 'make_tensor')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/get_query_tensor.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/select_qkv_indexes.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/dump_tensor.jsonl', 'symbolic_tensor/tensor_util/dump_view.jsonl', 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl', 'symbolic_tensor/tensor_util/pack_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl', 'test/test_st_attention_followed_by_st_moe.jsonl', 'test/test_transform_method_time_comparison.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'make_tensor')
        self.assertGreaterEqual(len(defs), 1)

    def test_089_default_prompt_for_output(self):
        """find_all_references('default_prompt_for_output'): 2 refs in 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'default_prompt_for_output')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r.member_tag, 'default_prompt_for_output')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'default_prompt_for_output')
        self.assertGreaterEqual(len(defs), 1)

    def test_090__build_nested(self):
        """find_all_references('_build_nested'): 6 refs in 3 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_build_nested')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 6)
        for r in refs:
            self.assertEqual(r.member_tag, '_build_nested')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['fs_util/get_nested_list_file_pathes.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl'])
        defs = dynamic_scope_go_to_definition(db, '_build_nested')
        self.assertGreaterEqual(len(defs), 1)

    def test_091__get_store(self):
        """find_all_references('_get_store'): 3 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_get_store')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 3)
        for r in refs:
            self.assertEqual(r.member_tag, '_get_store')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/symbolic_grad_registry.jsonl'])
        defs = dynamic_scope_go_to_definition(db, '_get_store')
        self.assertGreaterEqual(len(defs), 1)

    def test_092_st_file_paths(self):
        """find_all_references('st_file_paths'): 3 refs in 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'st_file_paths')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 3)
        for r in refs:
            self.assertEqual(r.member_tag, 'st_file_paths')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/tensor_util/register_tensor_ops.jsonl', 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'st_file_paths')
        self.assertGreaterEqual(len(defs), 1)

    def test_093_exact_match(self):
        """find_all_references('exact_match'): 1 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'exact_match')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r.member_tag, 'exact_match')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/select_qkv_indexes.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'exact_match')
        self.assertGreaterEqual(len(defs), 1)

    def test_094__read_file_content(self):
        """find_all_references('_read_file_content'): 1 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_read_file_content')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 1)
        for r in refs:
            self.assertEqual(r.member_tag, '_read_file_content')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/st_moe_forward.jsonl'])
        defs = dynamic_scope_go_to_definition(db, '_read_file_content')
        self.assertGreaterEqual(len(defs), 1)

    def test_095_sparse_to_dense(self):
        """find_all_references('sparse_to_dense'): 9 refs in 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'sparse_to_dense')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 9)
        for r in refs:
            self.assertEqual(r.member_tag, 'sparse_to_dense')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'sparse_to_dense')
        self.assertGreaterEqual(len(defs), 1)

    def test_096_SliceView(self):
        """find_all_references('SliceView'): 2 refs in 1 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'SliceView')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r.member_tag, 'SliceView')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/slice_view.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'SliceView')
        self.assertGreaterEqual(len(defs), 1)

    def test_097_load_tensor(self):
        """find_all_references('load_tensor'): 5 refs in 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'load_tensor')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 5)
        for r in refs:
            self.assertEqual(r.member_tag, 'load_tensor')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'load_tensor')
        self.assertGreaterEqual(len(defs), 1)

    def test_098__dense_to_sparse_impl(self):
        """find_all_references('_dense_to_sparse_impl'): 2 refs in 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, '_dense_to_sparse_impl')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r.member_tag, '_dense_to_sparse_impl')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl'])
        defs = dynamic_scope_go_to_definition(db, '_dense_to_sparse_impl')
        self.assertGreaterEqual(len(defs), 1)

    def test_099_WithDenseView(self):
        """find_all_references('WithDenseView'): 4 refs in 2 files"""
        db = _db()
        refs = dynamic_scope_find_all_references(db, 'WithDenseView')
        self.assertIsInstance(refs, list)
        self.assertEqual(len(refs), 4)
        for r in refs:
            self.assertEqual(r.member_tag, 'WithDenseView')
        ref_file_ids = sorted(set(r.file_id for r in refs))
        self.assertEqual(ref_file_ids, ['symbolic_tensor/module/with_dense_view.jsonl', 'test/test_st_attention_followed_by_st_moe.jsonl'])
        defs = dynamic_scope_go_to_definition(db, 'WithDenseView')
        self.assertGreaterEqual(len(defs), 1)


if __name__ == '__main__':
    unittest.main()
