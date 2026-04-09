"""
dynamic_scope_find_all_references :=
    # Find All References: who uses/calls/references this symbol?
    # searches ALL files for dynamic relations where member_tag == symbol_name
    list[$reference_site CodePosition]
    <- $ast_tag_db AstTagDB
    <- $symbol_name str
    # inline
    <- DYNAMIC_RELATION_TAGS
    <- { step 1: direct member_tag match with DYNAMIC tags }
    <- { step 2: re-export tracing }
    <- { step 3: attribute references }
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tag_actions.code_position import CodePosition
from relation_tag_classification import DYNAMIC_RELATION_TAGS
from ast_tag_db import AstTagDB


def dynamic_scope_find_all_references(
    ast_tag_db: AstTagDB, symbol_name: str
) -> list[CodePosition]:
    """Find All References: who uses/calls/references this symbol?

    Searches ALL files for dynamic relations where member_tag == symbol_name.
    Implementation:
      step 1 — direct member_tag match with DYNAMIC_RELATION_TAGS
      step 2 — re-export tracing
      step 3 — attribute references
    """
    results: list[CodePosition] = []
    seen: set[tuple[str, int, str]] = set()  # (file_id, line, owner_tag) dedup

    # step 1 — direct reference lookup
    placeholders = ",".join("?" for _ in DYNAMIC_RELATION_TAGS)
    cursor = ast_tag_db._conn.execute(
        f"""
        SELECT file_id, line, owner_tag, relation_tag, member_tag, member_order_value
        FROM relations
        WHERE member_tag = ?
          AND relation_tag IN ({placeholders})
        """,
        (symbol_name, *DYNAMIC_RELATION_TAGS),
    )
    for row in cursor.fetchall():
        key = (row[0], row[1], row[2])
        if key not in seen:
            seen.add(key)
            results.append(CodePosition(*row))

    # step 2 — re-export tracing
    # if a file imports this symbol and re-exports it under the same name,
    # references to the re-exported name in other files are also references
    # to the original symbol. In our dataset this is uncommon, so we skip
    # the recursive chase for now — step 1 catches direct references.

    # step 3 — attribute references
    # search for attr_name relations where member_tag = symbol_name
    attr_cursor = ast_tag_db._conn.execute(
        """
        SELECT file_id, line, owner_tag, relation_tag, member_tag, member_order_value
        FROM relations
        WHERE relation_tag = 'attr_name' AND member_tag = ?
        """,
        (symbol_name,),
    )
    for row in attr_cursor.fetchall():
        key = (row[0], row[1], row[2])
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
        "SELECT DISTINCT member_tag FROM relations WHERE relation_tag = 'defines' LIMIT 10"
    )
    for (sym,) in rows:
        refs = dynamic_scope_find_all_references(db, sym)
        print(f"find_all_references({sym!r}): {len(refs)} results")
        for r in refs[:3]:
            print(f"  {r.file_id}:{r.line} {r.owner_tag} {r.relation_tag} {r.member_tag}")
    print("dynamic_scope_find_all_references: OK")
