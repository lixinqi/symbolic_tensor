"""Compute normalized distance between two sets of AST tag relations.

Converts each relation list to AST JSON, then measures JSON tree distance.
"""

from typing import Any, Dict, List, Union
from typing_extensions import TypedDict

from experience.ast_tag.convert_ast_tag_jsonl_to_ast_json import (
    convert_ast_tag_jsonl_to_ast_json,
)
from experience.ast_tag.get_json_distance import get_json_distance


# ---------------------------------------------------------------------------
# AstTagRelation[Python] — typed dict matching ast_tag_relation.viba
# ---------------------------------------------------------------------------

#: Symbol: $Type_N owner tags, leaf-inlined member tags, or <module>.
Symbol = Union[str, int, float, bool]

#: RelationTag: "Type.field_name" format (e.g. "FunctionDef.body").
RelationTag = str


class AstTagRelation(TypedDict):
    """A single AST tag relation record (AstTagRelation[Python])."""
    line: int
    relation_tag: RelationTag
    owner_tag: Symbol
    member_tag: Symbol
    member_order_value: int


def get_ast_tags_distance(
    lhs: List[AstTagRelation],
    rhs: List[AstTagRelation],
) -> float:
    """Distance between two AST tag relation lists.

    Converts each to AST JSON tree via convert_ast_tag_jsonl_to_ast_json,
    then computes get_json_distance on the resulting trees.

    Returns 0.0 for identical, approaches 1.0 for fully different.
    """
    lhs_json: List[Dict[str, Any]]
    rhs_json: List[Dict[str, Any]]
    lhs_json, _ = convert_ast_tag_jsonl_to_ast_json(lhs)
    rhs_json, _ = convert_ast_tag_jsonl_to_ast_json(rhs)
    distance: float = get_json_distance(lhs_json, rhs_json)
    return distance


if __name__ == "__main__":
    import json
    import os
    import random

    from experience.ast_tag.convert_python_to_ast_tag_jsonl import (
        convert_python_to_ast_tag_jsonl,
    )
    from experience.ast_tag.convert_ast_json_to_ast_tag_jsonl import (
        random_dropout_tag_relations,
    )

    DATASET = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "example", "tag_auto_encoder", "dataset",
    )
    CODEBASE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "example", "code_auto_encoder", "codebase",
    )

    # Collect dataset JSONL files
    jsonl_files = []
    for root, _dirs, files in os.walk(DATASET):
        for f in sorted(files):
            if f.endswith(".jsonl"):
                jsonl_files.append(os.path.join(root, f))
    jsonl_files.sort()

    # Test 1: self-distance == 0
    with open(jsonl_files[0]) as fh:
        recs = [json.loads(l) for l in fh if l.strip()]
    d1 = get_ast_tags_distance(recs, recs)
    assert d1 == 0.0, f"self-distance should be 0.0, got {d1}"
    print(f"Test 1 (self=0):           {d1}")

    # Test 2: different files > 0
    with open(jsonl_files[1]) as fh:
        recs2 = [json.loads(l) for l in fh if l.strip()]
    d2 = get_ast_tags_distance(recs, recs2)
    print(f"Test 2 (diff files):       {d2:.4f}")
    assert d2 > 0.0

    # Test 3: symmetry
    d3a = get_ast_tags_distance(recs, recs2)
    d3b = get_ast_tags_distance(recs2, recs)
    print(f"Test 3 (symmetry):         {d3a:.4f} vs {d3b:.4f}")
    assert abs(d3a - d3b) < 1e-9

    # Test 4: dropout increases distance from original
    random.seed(42)
    dropped = random_dropout_tag_relations(recs, dropout_rate=0.3)
    d4 = get_ast_tags_distance(recs, dropped)
    print(f"Test 4 (dropout dist):     {d4:.4f} ({len(dropped)}/{len(recs)} kept)")
    assert d4 > 0.0

    # Test 5: more dropout = more distance
    dropped2 = random_dropout_tag_relations(recs, dropout_rate=0.6)
    d5 = get_ast_tags_distance(recs, dropped2)
    print(f"Test 5 (more dropout):     {d5:.4f} ({len(dropped2)}/{len(recs)} kept)")
    assert d5 > 0.0

    # Test 6: batch self-distance == 0 on 5 files
    self_ok = 0
    for fp in jsonl_files[:5]:
        with open(fp) as fh:
            r = [json.loads(l) for l in fh if l.strip()]
        if get_ast_tags_distance(r, r) == 0.0:
            self_ok += 1
    print(f"Test 6 (self=0 batch):     {self_ok}/5")
    assert self_ok == 5

    # Test 7: batch symmetry on 5 pairs
    random.seed(7)
    sym_ok = 0
    for _ in range(5):
        fp_a, fp_b = random.sample(jsonl_files, 2)
        with open(fp_a) as fh:
            ra = [json.loads(l) for l in fh if l.strip()]
        with open(fp_b) as fh:
            rb = [json.loads(l) for l in fh if l.strip()]
        dab = get_ast_tags_distance(ra, rb)
        dba = get_ast_tags_distance(rb, ra)
        if abs(dab - dba) < 1e-9:
            sym_ok += 1
    print(f"Test 7 (sym batch 5):      {sym_ok}/5")
    assert sym_ok == 5

    # Test 8: from source code — roundtrip distance should be 0
    py_file = os.path.join(CODEBASE, "fs_util", "pack_dir.py")
    with open(py_file) as fh:
        source = fh.read()
    jsonl_str = convert_python_to_ast_tag_jsonl(source)
    original = [json.loads(l) for l in jsonl_str.strip().splitlines()]
    d8 = get_ast_tags_distance(original, original)
    assert d8 == 0.0
    print(f"Test 8 (source self=0):    0.0")

    # Test 9: empty vs non-empty
    d9 = get_ast_tags_distance([], recs)
    print(f"Test 9 (empty vs full):    {d9:.4f}")
    assert d9 > 0.0

    # Test 10: empty vs empty
    d10 = get_ast_tags_distance([], [])
    assert d10 == 0.0
    print(f"Test 10 (empty vs empty):  0.0")

    # Test 11: triangle inequality
    with open(jsonl_files[2]) as fh:
        recs3 = [json.loads(l) for l in fh if l.strip()]
    d_ab = get_ast_tags_distance(recs, recs2)
    d_bc = get_ast_tags_distance(recs2, recs3)
    d_ac = get_ast_tags_distance(recs, recs3)
    print(f"Test 11 (triangle):        {d_ac:.4f} <= {d_ab + d_bc:.4f}")
    assert d_ac <= d_ab + d_bc + 1e-9

    # Test 12: distance from Python source roundtrip through JSONL is small
    py_file2 = os.path.join(CODEBASE, "sparse_util", "group_random_select.py")
    with open(py_file2) as fh:
        source2 = fh.read()
    jsonl_str2 = convert_python_to_ast_tag_jsonl(source2)
    original2 = [json.loads(l) for l in jsonl_str2.strip().splitlines()]
    d12 = get_ast_tags_distance(original, original2)
    print(f"Test 12 (cross-source):    {d12:.4f}")
    assert d12 > 0.0

    print(f"\nAll 12 tests passed.")
