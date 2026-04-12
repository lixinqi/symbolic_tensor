"""
test_dynamic_scope_go_to_definition: 100 concrete test cases.
Each case: symbol_name -> expected definition file_ids, expected count.
Round-trip: find_all_references includes the usage site.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ast_tag_db import AstTagDB
from tag_actions.dynamic_scope_go_to_definition import dynamic_scope_go_to_definition, DEFINITION_RELATION_TAGS
from tag_actions.dynamic_scope_find_all_references import dynamic_scope_find_all_references


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


class TestDynamicScopeGoToDefinition(unittest.TestCase):

    def test_000_slice_attention(self):
        """go_to_definition('slice_attention'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'slice_attention')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'slice_attention')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/st_attention.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'slice_attention')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/slice_attention.jsonl', ref_file_ids)

    def test_001__merge_and_shuffle_and_select_prefix_topk(self):
        """go_to_definition('_merge_and_shuffle_and_select_prefix_topk'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_merge_and_shuffle_and_select_prefix_topk')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_merge_and_shuffle_and_select_prefix_topk')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/st_moe_backward.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_merge_and_shuffle_and_select_prefix_topk')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/st_moe_backward.jsonl', ref_file_ids)

    def test_002_Union(self):
        """go_to_definition('Union'): 13 defs in 13 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'Union')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 13)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'Union')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['fs_util/get_nested_list_file_pathes.jsonl', 'llm_client/agent_task.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'Union')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('fs_util/get_nested_list_file_pathes.jsonl', ref_file_ids)

    def test_003__flatten_nested_indexes(self):
        """go_to_definition('_flatten_nested_indexes'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_flatten_nested_indexes')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_flatten_nested_indexes')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/st_moe_backward.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_flatten_nested_indexes')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/st_moe_backward.jsonl', ref_file_ids)

    def test_004_Path(self):
        """go_to_definition('Path'): 7 defs in 6 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'Path')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 7)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'Path')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'Path')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('fs_util/get_nested_list_file_pathes.jsonl', ref_file_ids)

    def test_005__do_one(self):
        """go_to_definition('_do_one'): 2 defs in 2 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_do_one')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 2)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_do_one')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['llm_client/coding_agent_task_handler.jsonl', 'llm_client/raw_llm_task_handler.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_do_one')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('llm_client/coding_agent_task_handler.jsonl', ref_file_ids)

    def test_006___init__(self):
        """go_to_definition('__init__'): 5 defs in 5 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '__init__')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 5)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '__init__')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl'])
        refs = dynamic_scope_find_all_references(db, '__init__')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/module/st_moe.jsonl', ref_file_ids)

    def test_007_zero_grad(self):
        """go_to_definition('zero_grad'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'zero_grad')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'zero_grad')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/optimizer/st_sgd.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'zero_grad')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/optimizer/st_sgd.jsonl', ref_file_ids)

    def test_008_frame_texts(self):
        """go_to_definition('frame_texts'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'frame_texts')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'frame_texts')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['test/test_attention_vs_traditional.jsonl'])

    def test_009__extract_coordinates(self):
        """go_to_definition('_extract_coordinates'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_extract_coordinates')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_extract_coordinates')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/select_qkv_indexes.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_extract_coordinates')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/select_qkv_indexes.jsonl', ref_file_ids)

    def test_010_make_none_tensor(self):
        """go_to_definition('make_none_tensor'): 11 defs in 10 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'make_none_tensor')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 11)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'make_none_tensor')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl', 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'make_none_tensor')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/merge_forward.jsonl', ref_file_ids)

    def test_011_dump_tensor(self):
        """go_to_definition('dump_tensor'): 3 defs in 3 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'dump_tensor')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 3)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'dump_tensor')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/tensor_util/dump_tensor.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'dump_tensor')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/st_copy.jsonl', ref_file_ids)

    def test_012_group_random_select(self):
        """go_to_definition('group_random_select'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'group_random_select')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'group_random_select')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['sparse_util/group_random_select.jsonl'])

    def test_013__get_nonzero_points(self):
        """go_to_definition('_get_nonzero_points'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_get_nonzero_points')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_get_nonzero_points')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/optimizer/st_sgd.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_get_nonzero_points')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/optimizer/st_sgd.jsonl', ref_file_ids)

    def test_014__write_storage(self):
        """go_to_definition('_write_storage'): 4 defs in 4 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_write_storage')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 4)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_write_storage')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_stack.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_write_storage')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/merge_forward.jsonl', ref_file_ids)

    def test_015_default_prompt_for_fork_grad_input(self):
        """go_to_definition('default_prompt_for_fork_grad_input'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'default_prompt_for_fork_grad_input')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'default_prompt_for_fork_grad_input')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/fork_tensor.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'default_prompt_for_fork_grad_input')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/fork_tensor.jsonl', ref_file_ids)

    def test_016_exact_match(self):
        """go_to_definition('exact_match'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'exact_match')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'exact_match')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/select_qkv_indexes.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'exact_match')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/select_qkv_indexes.jsonl', ref_file_ids)

    def test_017_StMoeModule(self):
        """go_to_definition('StMoeModule'): 2 defs in 2 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'StMoeModule')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 2)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'StMoeModule')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/module/st_moe.jsonl', 'test/test_st_attention_followed_by_st_moe.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'StMoeModule')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/module/st_moe.jsonl', ref_file_ids)

    def test_018__reset_grad_text_to_todo(self):
        """go_to_definition('_reset_grad_text_to_todo'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_reset_grad_text_to_todo')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_reset_grad_text_to_todo')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/optimizer/st_sgd.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_reset_grad_text_to_todo')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/optimizer/st_sgd.jsonl', ref_file_ids)

    def test_019___call__(self):
        """go_to_definition('__call__'): 3 defs in 3 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '__call__')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 3)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '__call__')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['llm_client/coding_agent_task_handler.jsonl', 'llm_client/raw_llm_task_handler.jsonl', 'llm_client/task_handler.jsonl'])

    def test_020_merge(self):
        """go_to_definition('merge'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'merge')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'merge')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/st_attention.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'merge')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/merge.jsonl', ref_file_ids)

    def test_021_st_moe_backward_grad_experience(self):
        """go_to_definition('st_moe_backward_grad_experience'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'st_moe_backward_grad_experience')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'st_moe_backward_grad_experience')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/st_moe_backward.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'st_moe_backward_grad_experience')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/st_moe_backward.jsonl', ref_file_ids)

    def test_022_parameters(self):
        """go_to_definition('parameters'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'parameters')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'parameters')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/module/st_moe.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'parameters')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/module/st_moe.jsonl', ref_file_ids)

    def test_023_torch_nn(self):
        """go_to_definition('torch.nn'): 2 defs in 2 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'torch.nn')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 2)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'torch.nn')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl'])

    def test_024_Copy(self):
        """go_to_definition('Copy'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'Copy')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'Copy')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/st_copy.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'Copy')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/st_copy.jsonl', ref_file_ids)

    def test_025_read_storage(self):
        """go_to_definition('read_storage'): 23 defs in 23 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'read_storage')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 23)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'read_storage')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'test/test_attention_vs_traditional.jsonl', 'test/test_gain_st_sgd.jsonl', 'test/test_st_attention_followed_by_st_moe.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'read_storage')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', ref_file_ids)

    def test_026_WithDenseView(self):
        """go_to_definition('WithDenseView'): 2 defs in 2 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'WithDenseView')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 2)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'WithDenseView')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/module/with_dense_view.jsonl', 'test/test_st_attention_followed_by_st_moe.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'WithDenseView')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/module/with_dense_view.jsonl', ref_file_ids)

    def test_027_st_patched(self):
        """go_to_definition('st_patched'): 4 defs in 4 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'st_patched')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 4)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'st_patched')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'st_patched')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/merge_backward.jsonl', ref_file_ids)

    def test_028__coords_to_flat(self):
        """go_to_definition('_coords_to_flat'): 2 defs in 2 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_coords_to_flat')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 2)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_coords_to_flat')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_stack.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_coords_to_flat')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/slice_view.jsonl', ref_file_ids)

    def test_029_Optional(self):
        """go_to_definition('Optional'): 25 defs in 25 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'Optional')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 25)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'Optional')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['llm_client/coding_agent_query.jsonl', 'llm_client/coding_agent_task_handler.jsonl', 'llm_client/raw_llm_query.jsonl', 'llm_client/raw_llm_task_handler.jsonl', 'llm_client/task_handler.jsonl', 'symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_query_tensor.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/select_qkv_indexes.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl', 'test/test_st_attention_followed_by_st_moe.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'Optional')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('llm_client/coding_agent_query.jsonl', ref_file_ids)

    def test_030__MERGED(self):
        """go_to_definition('_MERGED'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_MERGED')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_MERGED')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['test/test_st_attention_followed_by_st_moe.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_MERGED')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/st_moe_backward.jsonl', ref_file_ids)

    def test_031__str_to_digit_list(self):
        """go_to_definition('_str_to_digit_list'): 4 defs in 4 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_str_to_digit_list')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 4)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_str_to_digit_list')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/dump_tensor.jsonl', 'symbolic_tensor/tensor_util/dump_view.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_str_to_digit_list')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/tensor_util/dump_tensor.jsonl', ref_file_ids)

    def test_032__get_storage_path(self):
        """go_to_definition('_get_storage_path'): 13 defs in 12 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_get_storage_path')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 13)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_get_storage_path')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['fs_util/get_nested_list_file_pathes.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_get_storage_path')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('fs_util/get_nested_list_file_pathes.jsonl', ref_file_ids)

    def test_033_transpose_pairs_coordinates(self):
        """go_to_definition('transpose_pairs_coordinates'): 2 defs in 2 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'transpose_pairs_coordinates')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 2)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'transpose_pairs_coordinates')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['sparse_util/transpose_pairs_coordinates.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'transpose_pairs_coordinates')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/st_moe_backward.jsonl', ref_file_ids)

    def test_034_get_frames(self):
        """go_to_definition('get_frames'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'get_frames')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'get_frames')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['test/test_attention_vs_traditional.jsonl'])

    def test_035_get_nested_list_file_pathes(self):
        """go_to_definition('get_nested_list_file_pathes'): 3 defs in 3 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'get_nested_list_file_pathes')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 3)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'get_nested_list_file_pathes')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['fs_util/get_nested_list_file_pathes.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'get_nested_list_file_pathes')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/optimizer/st_sgd.jsonl', ref_file_ids)

    def test_036__grep_by_file_content_hint(self):
        """go_to_definition('_grep_by_file_content_hint'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_grep_by_file_content_hint')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_grep_by_file_content_hint')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['llm_client/raw_llm_task_handler.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_grep_by_file_content_hint')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('llm_client/raw_llm_task_handler.jsonl', ref_file_ids)

    def test_037__flat_index_from_coordinates(self):
        """go_to_definition('_flat_index_from_coordinates'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_flat_index_from_coordinates')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_flat_index_from_coordinates')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/dump_view.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_flat_index_from_coordinates')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/tensor_util/dump_view.jsonl', ref_file_ids)

    def test_038__build_nested(self):
        """go_to_definition('_build_nested'): 3 defs in 3 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_build_nested')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 3)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_build_nested')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['fs_util/get_nested_list_file_pathes.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_build_nested')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('fs_util/get_nested_list_file_pathes.jsonl', ref_file_ids)

    def test_039_pack(self):
        """go_to_definition('pack'): 4 defs in 4 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'pack')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 4)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'pack')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['fs_util/text_merger.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/st_attention.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'pack')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('fs_util/text_merger.jsonl', ref_file_ids)

    def test_040_coding_agent(self):
        """go_to_definition('coding_agent'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'coding_agent')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'coding_agent')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/coding_agent.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'coding_agent')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/coding_agent.jsonl', ref_file_ids)

    def test_041_assign_view(self):
        """go_to_definition('assign_view'): 3 defs in 3 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'assign_view')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 3)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'assign_view')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'assign_view')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/tensor_util/assign_view.jsonl', ref_file_ids)

    def test_042_st_get_diff(self):
        """go_to_definition('st_get_diff'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'st_get_diff')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'st_get_diff')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/register_tensor_ops.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'st_get_diff')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/tensor_util/patch_tensor.jsonl', ref_file_ids)

    def test_043_symbolic_grad_registry(self):
        """go_to_definition('symbolic_grad_registry'): 10 defs in 10 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'symbolic_grad_registry')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 10)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'symbolic_grad_registry')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'symbolic_grad_registry')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/fork_tensor.jsonl', ref_file_ids)

    def test_044_slice_attention_backward(self):
        """go_to_definition('slice_attention_backward'): 2 defs in 2 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'slice_attention_backward')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 2)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'slice_attention_backward')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'slice_attention_backward')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/slice_attention.jsonl', ref_file_ids)

    def test_045_get_causal_attention_mask(self):
        """go_to_definition('get_causal_attention_mask'): 5 defs in 5 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'get_causal_attention_mask')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 5)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'get_causal_attention_mask')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/get_causal_attention_mask.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'test/test_attention_vs_traditional.jsonl', 'test/test_st_attention_followed_by_st_moe.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'get_causal_attention_mask')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/get_causal_attention_mask.jsonl', ref_file_ids)

    def test_046__pad_random_indexes_to_topk_with_none_experience_indexes(self):
        """go_to_definition('_pad_random_indexes_to_topk_with_none_experience_indexes'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_pad_random_indexes_to_topk_with_none_experience_indexes')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_pad_random_indexes_to_topk_with_none_experience_indexes')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/st_moe_backward.jsonl'])

    def test_047__resolve_grad_output(self):
        """go_to_definition('_resolve_grad_output'): 2 defs in 2 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_resolve_grad_output')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 2)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_resolve_grad_output')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_resolve_grad_output')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/slice_tensor.jsonl', ref_file_ids)

    def test_048_GetEditDistanceRatio(self):
        """go_to_definition('GetEditDistanceRatio'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'GetEditDistanceRatio')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'GetEditDistanceRatio')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/get_edit_distance_ratio.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'GetEditDistanceRatio')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/get_edit_distance_ratio.jsonl', ref_file_ids)

    def test_049__unflatten(self):
        """go_to_definition('_unflatten'): 2 defs in 2 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_unflatten')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 2)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_unflatten')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_unflatten')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/get_edit_distance_ratio.jsonl', ref_file_ids)

    def test_050_experience_symbolic_tensor_tensor_util_register_tensor_ops(self):
        """go_to_definition('experience.symbolic_tensor.tensor_util.register_tensor_ops'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'experience.symbolic_tensor.tensor_util.register_tensor_ops')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'experience.symbolic_tensor.tensor_util.register_tensor_ops')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/patch_tensor.jsonl'])

    def test_051_pack_dir(self):
        """go_to_definition('pack_dir'): 3 defs in 3 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'pack_dir')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 3)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'pack_dir')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['fs_util/pack_dir.jsonl', 'llm_client/raw_llm_task_handler.jsonl', 'symbolic_tensor/tensor_util/pack_tensor.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'pack_dir')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('llm_client/raw_llm_task_handler.jsonl', ref_file_ids)

    def test_052__read_storage(self):
        """go_to_definition('_read_storage'): 7 defs in 7 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_read_storage')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 7)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_read_storage')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'test/test_st_attention_followed_by_st_moe.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_read_storage')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/get_edit_distance_ratio.jsonl', ref_file_ids)

    def test_053_step(self):
        """go_to_definition('step'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'step')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'step')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/optimizer/st_sgd.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'step')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/optimizer/st_sgd.jsonl', ref_file_ids)

    def test_054_torch(self):
        """go_to_definition('torch'): 53 defs in 53 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'torch')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 53)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'torch')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['fs_util/get_nested_list_file_pathes.jsonl', 'sparse_util/convert_nested_list_coordinates_to_pairs_coordinates.jsonl', 'sparse_util/group_random_select.jsonl', 'sparse_util/transpose_pairs_coordinates.jsonl', 'symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl', 'symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_causal_attention_mask.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/get_query_tensor.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/select_qkv_indexes.jsonl', 'symbolic_tensor/function/slice_and_concat_attention_forward.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/assign_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/dump_tensor.jsonl', 'symbolic_tensor/tensor_util/dump_view.jsonl', 'symbolic_tensor/tensor_util/empty_tensor_like.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl', 'symbolic_tensor/tensor_util/make_none_tensor.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl', 'symbolic_tensor/tensor_util/none_tensor_like.jsonl', 'symbolic_tensor/tensor_util/pack_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl', 'symbolic_tensor/tensor_util/todo_tensor_like.jsonl', 'test/test_attention_vs_traditional.jsonl', 'test/test_gain_st_sgd.jsonl', 'test/test_st_attention_followed_by_st_moe.jsonl', 'test/test_transform_method_time_comparison.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'torch')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('fs_util/get_nested_list_file_pathes.jsonl', ref_file_ids)

    def test_055_TextMerger(self):
        """go_to_definition('TextMerger'): 9 defs in 8 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'TextMerger')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 9)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'TextMerger')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['fs_util/text_merger.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/merge_forward.jsonl', 'symbolic_tensor/function/st_attention.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'test/test_attention_vs_traditional.jsonl', 'test/test_st_attention_followed_by_st_moe.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'TextMerger')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/merge.jsonl', ref_file_ids)

    def test_056_time(self):
        """go_to_definition('time'): 2 defs in 2 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'time')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 2)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'time')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['llm_client/task_handler.jsonl', 'test/test_transform_method_time_comparison.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'time')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('llm_client/task_handler.jsonl', ref_file_ids)

    def test_057_slice_backward(self):
        """go_to_definition('slice_backward'): 2 defs in 2 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'slice_backward')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 2)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'slice_backward')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'slice_backward')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/slice_tensor.jsonl', ref_file_ids)

    def test_058_custom_prompt(self):
        """go_to_definition('custom_prompt'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'custom_prompt')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'custom_prompt')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/coding_agent.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'custom_prompt')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/coding_agent.jsonl', ref_file_ids)

    def test_059_default_prompt_for_grad_input_frame(self):
        """go_to_definition('default_prompt_for_grad_input_frame'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'default_prompt_for_grad_input_frame')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'default_prompt_for_grad_input_frame')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/st_moe_backward.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'default_prompt_for_grad_input_frame')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/st_moe_backward.jsonl', ref_file_ids)

    def test_060_run_benchmark(self):
        """go_to_definition('run_benchmark'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'run_benchmark')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'run_benchmark')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['test/test_transform_method_time_comparison.jsonl'])

    def test_061__save_st_attrs(self):
        """go_to_definition('_save_st_attrs'): 2 defs in 2 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_save_st_attrs')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 2)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_save_st_attrs')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_save_st_attrs')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/slice_tensor.jsonl', ref_file_ids)

    def test_062__ensure_newline(self):
        """go_to_definition('_ensure_newline'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_ensure_newline')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_ensure_newline')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/get_diff_tensor.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_ensure_newline')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/tensor_util/get_diff_tensor.jsonl', ref_file_ids)

    def test_063_Tuple(self):
        """go_to_definition('Tuple'): 13 defs in 13 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'Tuple')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 13)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'Tuple')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['fs_util/text_merger.jsonl', 'sparse_util/convert_nested_list_coordinates_to_pairs_coordinates.jsonl', 'sparse_util/transpose_pairs_coordinates.jsonl', 'symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/select_qkv_indexes.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl', 'symbolic_tensor/tensor_util/dump_view.jsonl', 'symbolic_tensor/tensor_util/slice_view.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'Tuple')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('fs_util/text_merger.jsonl', ref_file_ids)

    def test_064_slice_tensor(self):
        """go_to_definition('slice_tensor'): 10 defs in 9 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'slice_tensor')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 10)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'slice_tensor')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_query_tensor.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl', 'symbolic_tensor/tensor_util/slice_tensor.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'slice_tensor')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/coding_agent.jsonl', ref_file_ids)

    def test_065_SparseToDense(self):
        """go_to_definition('SparseToDense'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'SparseToDense')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'SparseToDense')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'SparseToDense')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/tensor_util/sparse_to_dense.jsonl', ref_file_ids)

    def test_066_read_output(self):
        """go_to_definition('read_output'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'read_output')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'read_output')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/coding_agent.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'read_output')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/coding_agent.jsonl', ref_file_ids)

    def test_067_select_qkv_indexes(self):
        """go_to_definition('select_qkv_indexes'): 2 defs in 2 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'select_qkv_indexes')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 2)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'select_qkv_indexes')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/select_qkv_indexes.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'select_qkv_indexes')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/select_qkv_indexes.jsonl', ref_file_ids)

    def test_068_Any(self):
        """go_to_definition('Any'): 8 defs in 8 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'Any')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 8)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'Any')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['fs_util/get_nested_list_file_pathes.jsonl', 'sparse_util/convert_nested_list_coordinates_to_pairs_coordinates.jsonl', 'sparse_util/transpose_pairs_coordinates.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/module/st_moe.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'Any')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('sparse_util/convert_nested_list_coordinates_to_pairs_coordinates.jsonl', ref_file_ids)

    def test_069__assert_consistent_shape(self):
        """go_to_definition('_assert_consistent_shape'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_assert_consistent_shape')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_assert_consistent_shape')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/make_tensor.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_assert_consistent_shape')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/tensor_util/make_tensor.jsonl', ref_file_ids)

    def test_070__sparse_to_dense_impl(self):
        """go_to_definition('_sparse_to_dense_impl'): 2 defs in 2 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_sparse_to_dense_impl')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 2)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_sparse_to_dense_impl')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_sparse_to_dense_impl')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/with_dense_view.jsonl', ref_file_ids)

    def test_071__StSlicer(self):
        """go_to_definition('_StSlicer'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_StSlicer')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_StSlicer')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/register_tensor_ops.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_StSlicer')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/tensor_util/register_tensor_ops.jsonl', ref_file_ids)

    def test_072_TaskHandler(self):
        """go_to_definition('TaskHandler'): 7 defs in 7 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'TaskHandler')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 7)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'TaskHandler')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['llm_client/task_handler.jsonl', 'symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_query_tensor.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'TaskHandler')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/coding_agent.jsonl', ref_file_ids)

    def test_073_get_diff_tensor(self):
        """go_to_definition('get_diff_tensor'): 12 defs in 12 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'get_diff_tensor')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 12)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'get_diff_tensor')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/tensor_util/get_diff_tensor.jsonl', 'symbolic_tensor/tensor_util/patch_tensor.jsonl', 'symbolic_tensor/tensor_util/register_tensor_ops.jsonl', 'symbolic_tensor/tensor_util/st_patched.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'get_diff_tensor')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/fork_tensor.jsonl', ref_file_ids)

    def test_074__st_value_slicer(self):
        """go_to_definition('_st_value_slicer'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_st_value_slicer')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_st_value_slicer')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/register_tensor_ops.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_st_value_slicer')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/tensor_util/register_tensor_ops.jsonl', ref_file_ids)

    def test_075_raw_llm_query(self):
        """go_to_definition('raw_llm_query'): 2 defs in 2 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'raw_llm_query')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 2)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'raw_llm_query')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['llm_client/raw_llm_query.jsonl', 'llm_client/raw_llm_task_handler.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'raw_llm_query')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('llm_client/raw_llm_task_handler.jsonl', ref_file_ids)

    def test_076__flatten(self):
        """go_to_definition('_flatten'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_flatten')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_flatten')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/make_tensor.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_flatten')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/tensor_util/make_tensor.jsonl', ref_file_ids)

    def test_077_uuid(self):
        """go_to_definition('uuid'): 2 defs in 2 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'uuid')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 2)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'uuid')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/make_none_tensor.jsonl', 'symbolic_tensor/tensor_util/none_tensor_like.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'uuid')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/tensor_util/make_none_tensor.jsonl', ref_file_ids)

    def test_078__read_flat_data(self):
        """go_to_definition('_read_flat_data'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_read_flat_data')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_read_flat_data')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/load_tensor.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_read_flat_data')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/tensor_util/load_tensor.jsonl', ref_file_ids)

    def test_079_build_model(self):
        """go_to_definition('build_model'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'build_model')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'build_model')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['test/test_st_attention_followed_by_st_moe.jsonl'])

    def test_080__run_all(self):
        """go_to_definition('_run_all'): 2 defs in 2 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_run_all')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 2)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_run_all')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['llm_client/coding_agent_task_handler.jsonl', 'llm_client/raw_llm_task_handler.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_run_all')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('llm_client/coding_agent_task_handler.jsonl', ref_file_ids)

    def test_081__flatten_nested_paths(self):
        """go_to_definition('_flatten_nested_paths'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_flatten_nested_paths')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_flatten_nested_paths')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/optimizer/st_sgd.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_flatten_nested_paths')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/optimizer/st_sgd.jsonl', ref_file_ids)

    def test_082__get_store(self):
        """go_to_definition('_get_store'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_get_store')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_get_store')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/symbolic_grad_registry.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_get_store')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/symbolic_grad_registry.jsonl', ref_file_ids)

    def test_083_get_query_tensor(self):
        """go_to_definition('get_query_tensor'): 3 defs in 3 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'get_query_tensor')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 3)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'get_query_tensor')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/get_query_tensor.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/optimizer/st_sgd.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'get_query_tensor')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/get_query_tensor.jsonl', ref_file_ids)

    def test_084___iter__(self):
        """go_to_definition('__iter__'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '__iter__')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '__iter__')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/data_loader/sole_file_batch_data_loader.jsonl'])

    def test_085__dense_to_sparse_impl(self):
        """go_to_definition('_dense_to_sparse_impl'): 2 defs in 2 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_dense_to_sparse_impl')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 2)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_dense_to_sparse_impl')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/tensor_util/dense_to_sparse.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_dense_to_sparse_impl')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/with_dense_view.jsonl', ref_file_ids)

    def test_086_StMoe(self):
        """go_to_definition('StMoe'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'StMoe')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'StMoe')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/st_moe.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'StMoe')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/st_moe.jsonl', ref_file_ids)

    def test_087_forward(self):
        """go_to_definition('forward'): 13 defs in 13 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'forward')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 13)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'forward')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/get_edit_distance_ratio.jsonl', 'symbolic_tensor/function/merge.jsonl', 'symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_tensor.jsonl', 'symbolic_tensor/function/slice_view.jsonl', 'symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_stack.jsonl', 'symbolic_tensor/function/with_dense_view.jsonl', 'symbolic_tensor/module/st_moe.jsonl', 'symbolic_tensor/module/with_dense_view.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'forward')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/get_edit_distance_ratio.jsonl', ref_file_ids)

    def test_088__unzip_to_tensor_list(self):
        """go_to_definition('_unzip_to_tensor_list'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_unzip_to_tensor_list')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_unzip_to_tensor_list')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/select_qkv_indexes.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_unzip_to_tensor_list')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/select_qkv_indexes.jsonl', ref_file_ids)

    def test_089__expand(self):
        """go_to_definition('_expand'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_expand')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_expand')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/empty_tensor_like.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_expand')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/tensor_util/empty_tensor_like.jsonl', ref_file_ids)

    def test_090_peek(self):
        """go_to_definition('peek'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'peek')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'peek')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/symbolic_grad_registry.jsonl'])

    def test_091_load_tensor(self):
        """go_to_definition('load_tensor'): 2 defs in 2 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'load_tensor')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 2)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'load_tensor')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/st_copy.jsonl', 'symbolic_tensor/tensor_util/load_tensor.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'load_tensor')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/st_copy.jsonl', ref_file_ids)

    def test_092_ClaudeAgentOptions(self):
        """go_to_definition('ClaudeAgentOptions'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'ClaudeAgentOptions')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'ClaudeAgentOptions')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['llm_client/coding_agent_query.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'ClaudeAgentOptions')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('llm_client/coding_agent_query.jsonl', ref_file_ids)

    def test_093_default_prompt_for_grad_exp_value(self):
        """go_to_definition('default_prompt_for_grad_exp_value'): 2 defs in 2 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'default_prompt_for_grad_exp_value')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 2)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'default_prompt_for_grad_exp_value')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/st_moe.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'default_prompt_for_grad_exp_value')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/st_moe_backward.jsonl', ref_file_ids)

    def test_094__st_view_slicer(self):
        """go_to_definition('_st_view_slicer'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_st_view_slicer')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_st_view_slicer')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/register_tensor_ops.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_st_view_slicer')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/tensor_util/register_tensor_ops.jsonl', ref_file_ids)

    def test_095_threading(self):
        """go_to_definition('threading'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'threading')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'threading')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/symbolic_grad_registry.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'threading')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/symbolic_grad_registry.jsonl', ref_file_ids)

    def test_096__get_shape(self):
        """go_to_definition('_get_shape'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, '_get_shape')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, '_get_shape')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/tensor_util/make_tensor.jsonl'])
        refs = dynamic_scope_find_all_references(db, '_get_shape')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/tensor_util/make_tensor.jsonl', ref_file_ids)

    def test_097_shutil(self):
        """go_to_definition('shutil'): 10 defs in 10 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'shutil')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 10)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'shutil')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/coding_agent.jsonl', 'symbolic_tensor/function/fork_tensor.jsonl', 'symbolic_tensor/function/merge_backward.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/st_moe_backward.jsonl', 'symbolic_tensor/function/st_moe_forward.jsonl', 'symbolic_tensor/tensor_util/assign_tensor.jsonl', 'symbolic_tensor/tensor_util/dump_tensor.jsonl', 'symbolic_tensor/tensor_util/make_tensor.jsonl', 'symbolic_tensor/tensor_util/sparse_to_dense.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'shutil')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/coding_agent.jsonl', ref_file_ids)

    def test_098_slice_attention_forward(self):
        """go_to_definition('slice_attention_forward'): 3 defs in 3 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'slice_attention_forward')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 3)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'slice_attention_forward')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['symbolic_tensor/function/slice_attention.jsonl', 'symbolic_tensor/function/slice_attention_backward.jsonl', 'symbolic_tensor/function/slice_attention_forward.jsonl'])
        refs = dynamic_scope_find_all_references(db, 'slice_attention_forward')
        ref_file_ids = set(r.file_id for r in refs)
        self.assertIn('symbolic_tensor/function/slice_attention.jsonl', ref_file_ids)

    def test_099_run_pipeline(self):
        """go_to_definition('run_pipeline'): 1 defs in 1 files"""
        db = _db()
        defs = dynamic_scope_go_to_definition(db, 'run_pipeline')
        self.assertIsInstance(defs, list)
        self.assertEqual(len(defs), 1)
        for d in defs:
            self.assertIn(d.relation_tag, DEFINITION_RELATION_TAGS)
            self.assertEqual(d.member_tag, 'run_pipeline')
        pairs = [(d.file_id, d.owner_tag) for d in defs]
        self.assertEqual(len(pairs), len(set(pairs)))
        def_file_ids = sorted(set(d.file_id for d in defs))
        self.assertEqual(def_file_ids, ['test/test_st_attention_followed_by_st_moe.jsonl'])


if __name__ == '__main__':
    unittest.main()
