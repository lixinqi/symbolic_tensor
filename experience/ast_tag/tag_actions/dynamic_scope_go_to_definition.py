"""
dynamic_scope_go_to_definition :=
    # Go to Definition: where is this symbol defined?
    # searches ALL files for definition relations (FunctionDef.name, ClassDef.name)
    list[$definition_site CodePosition]
    <- $ast_tag_db AstTagDB
    <- $symbol_name str
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tag_actions.code_position import CodePosition
from ast_tag_db import AstTagDB


DEFINITION_RELATION_TAGS = frozenset({
    "FunctionDef__name", "AsyncFunctionDef__name", "ClassDef__name",
    "alias__name",  # imported symbols
})


def dynamic_scope_go_to_definition(
    ast_tag_db: AstTagDB, symbol_name: str
) -> list[CodePosition]:
    """Go to Definition: where is this symbol defined?

    Searches ALL files for definition relations (FunctionDef.name, ClassDef.name, alias.name).
    """
    results: list[CodePosition] = []
    seen: set[tuple[str, str]] = set()  # (file_id, owner_tag) dedup

    # Direct definition lookup across all files
    placeholders = ",".join("?" for _ in DEFINITION_RELATION_TAGS)
    cursor = ast_tag_db._conn.execute(
        f"""
        SELECT file_id, line, owner_tag, relation_tag, member_tag, member_order_value
        FROM relations
        WHERE relation_tag IN ({placeholders}) AND member_tag = ?
        """,
        (*DEFINITION_RELATION_TAGS, symbol_name),
    )
    for row in cursor.fetchall():
        key = (row[0], row[2])  # (file_id, owner_tag)
        if key not in seen:
            seen.add(key)
            results.append(CodePosition(*row))

    return results


if __name__ == "__main__":
    from ast_tag_db import load_jsonl_dataset_into_ast_tag_db

    dataset_dir = os.path.join(os.path.dirname(__file__), "..", "test_dataset")
    db = load_jsonl_dataset_into_ast_tag_db(dataset_dir)
    # pick some defined symbols
    rows = db.execute_raw_sql_query(
        "SELECT DISTINCT member_tag FROM relations WHERE relation_tag = 'FunctionDef__name' LIMIT 10"
    )
    for (sym,) in rows:
        defs = dynamic_scope_go_to_definition(db, sym)
        print(f"go_to_definition({sym!r}): {len(defs)} results")
        for d in defs[:3]:
            print(f"  {d.file_id}:{d.line} {d.owner_tag} {d.relation_tag} {d.member_tag}")
    print("dynamic_scope_go_to_definition: OK")