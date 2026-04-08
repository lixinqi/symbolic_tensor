"""Convert AST JSON tree to AstTagRelation JSONL.

Thin wrapper: wraps body list in a Module node and runs the generic extractor.
"""

import json
from typing import Any, Dict, List, Optional

from experience.ast_tag.convert_python_to_ast_tag_jsonl import (
    _TempVarGen, _extract, _LEAF_TYPES,
)


def _annotate_symbols(node: Any, path: str, path_map: Dict[str, str]):
    """Walk JSON tree and inject _symbol from path_map into each node."""
    if not isinstance(node, dict) or '_type' not in node:
        return
    t = node.get('_type', '')
    if t in ('Name', 'Constant') or t in _LEAF_TYPES:
        return
    sym = path_map.get(path)
    if sym:
        node['_symbol'] = sym
    for field_name, value in node.items():
        if field_name.startswith('_'):
            continue
        if isinstance(value, list):
            for i, item in enumerate(value):
                _annotate_symbols(item, f"{path}/{field_name}/{i}", path_map)
        elif isinstance(value, dict):
            _annotate_symbols(value, f"{path}/{field_name}", path_map)


def convert_ast_json_to_ast_tag_jsonl(
    json_nodes: List[Dict],
    inner_path_to_tmp_var_name: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    module_node = {"_type": "Module", "body": json_nodes}
    if inner_path_to_tmp_var_name:
        _annotate_symbols(module_node, "", inner_path_to_tmp_var_name)
    rels: List[Dict] = []
    tmp_gen = _TempVarGen()
    _extract(module_node, rels, tmp_gen)
    return rels


def random_dropout_tag_relations(
    records: List[Dict], dropout_rate: float = 0.3,
) -> List[Dict]:
    """Randomly drop relations, then prune to the subgraph reachable from <module>.

    After random dropout, only keep relations whose owner_tag is transitively
    reachable from <module> via surviving member_tag→owner_tag links.
    Then remove any relation pointing to a $Type_N member that has no owner.
    """
    import random
    kept = [r for r in records if random.random() >= dropout_rate]
    # Build reachability from <module>
    children = {}
    for r in kept:
        children.setdefault(r["owner_tag"], set()).add(r["member_tag"])
    reachable = set()
    stack = ["<module>"]
    while stack:
        sym = stack.pop()
        if sym in reachable:
            continue
        reachable.add(sym)
        for child in children.get(sym, ()):
            if child.startswith("$"):
                stack.append(child)
    # Keep only relations whose owner is reachable
    kept = [r for r in kept if r["owner_tag"] in reachable]
    # Drop relations pointing to unreachable $Type_N members
    owners = {r["owner_tag"] for r in kept}
    owners.add("<module>")
    kept = [r for r in kept if not r["member_tag"].startswith("$")
            or r["member_tag"] in owners]
    return kept


if __name__ == "__main__":
    import os
    import random
    from experience.ast_tag.convert_python_to_ast_tag_jsonl import (
        convert_python_to_ast_tag_jsonl,
    )
    from experience.ast_tag.convert_ast_tag_jsonl_to_ast_json import (
        convert_ast_tag_jsonl_to_ast_json,
    )

    CODEBASE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "example", "code_auto_encoder", "codebase",
    )
    py_files = []
    for root, _dirs, files in os.walk(CODEBASE):
        for f in sorted(files):
            if f.endswith(".py") and f != "__init__.py":
                py_files.append(os.path.join(root, f))
    py_files.sort()

    # --- exact roundtrip tests (existing) ---
    passed = 0
    for fp in py_files:
        rel = os.path.relpath(fp, CODEBASE)
        with open(fp) as fh:
            source = fh.read()
        jsonl_str = convert_python_to_ast_tag_jsonl(source)
        original = [json.loads(l) for l in jsonl_str.strip().splitlines()]

        json_nodes, path_map = convert_ast_tag_jsonl_to_ast_json(original)
        roundtrip = convert_ast_json_to_ast_tag_jsonl(json_nodes, path_map)

        orig_set = {(r["relation_tag"], r["owner_tag"], r["member_tag"])
                     for r in original}
        rt_set = {(r["relation_tag"], r["owner_tag"], r["member_tag"])
                   for r in roundtrip}
        if orig_set == rt_set:
            passed += 1
        else:
            print(f"DIFF {rel}: -{len(orig_set - rt_set)} +{len(rt_set - orig_set)}")
    print(f"Roundtrip: {passed}/{len(py_files)}")

    # --- robustness tests: dropout + roundtrip ---
    DATASET = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "example", "tag_auto_encoder", "dataset",
    )
    jsonl_files = []
    for root, _dirs, files in os.walk(DATASET):
        for f in sorted(files):
            if f.endswith(".jsonl"):
                jsonl_files.append(os.path.join(root, f))
    jsonl_files.sort()

    random.seed(42)
    robust_passed = 0
    robust_total = 25
    sample_files = random.sample(jsonl_files, min(robust_total, len(jsonl_files)))
    for fp in sample_files:
        rel = os.path.relpath(fp, DATASET)
        with open(fp) as fh:
            records = [json.loads(l) for l in fh if l.strip()]
        x = random_dropout_tag_relations(records)
        json_nodes, path_map = convert_ast_tag_jsonl_to_ast_json(x)
        rt = convert_ast_json_to_ast_tag_jsonl(json_nodes, path_map)
        x_set = {(r["relation_tag"], r["owner_tag"], r["member_tag"]) for r in x}
        rt_set = {(r["relation_tag"], r["owner_tag"], r["member_tag"]) for r in rt}
        if x_set == rt_set:
            robust_passed += 1
            print(f"ROBUST OK   {rel} ({len(x)}/{len(records)} relations kept)")
        else:
            print(f"ROBUST DIFF {rel}: -{len(x_set - rt_set)} +{len(rt_set - x_set)}")
    print(f"\nRobust roundtrip: {robust_passed}/{robust_total}")
