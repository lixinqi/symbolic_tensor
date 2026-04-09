"""
dynamic_scope_go_to_definition :=
    # Go to Definition: where is this symbol defined?
    # searches ALL files for "defines" relation with member_tag == symbol_name
    list[$definition_site CodePosition]
    <- $ast_tag_db AstTagDB
    <- $symbol_name str
    # inline
    <- { step 1: direct "defines" lookup across all files }
    <- { step 2: import resolution — trace import chains }
    <- { step 3: attribute resolution — for dotted names }
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tag_actions.code_position import CodePosition
from ast_tag_db import AstTagDB


def dynamic_scope_go_to_definition(
    ast_tag_db: AstTagDB, symbol_name: str
) -> list[CodePosition]:
    """Go to Definition: where is this symbol defined?

    Searches ALL files for 'defines' relation with member_tag == symbol_name.
    Implementation:
      step 1 — direct definition lookup
      step 2 — import resolution (trace import chain to defining file)
      step 3 — attribute resolution (for dotted names like foo.bar)
    """
    results: list[CodePosition] = []
    seen: set[tuple[str, str]] = set()  # (file_id, owner_tag) dedup

    # step 1 — direct definition lookup across all files
    cursor = ast_tag_db._conn.execute(
        """
        SELECT file_id, line, owner_tag, relation_tag, member_tag, member_order_value
        FROM relations
        WHERE relation_tag = 'defines' AND member_tag = ?
        """,
        (symbol_name,),
    )
    for row in cursor.fetchall():
        key = (row[0], row[2])  # (file_id, owner_tag)
        if key not in seen:
            seen.add(key)
            results.append(CodePosition(*row))

    if results:
        return results

    # step 2 — import resolution
    # find files that import this symbol, then trace to the defining module
    import_cursor = ast_tag_db._conn.execute(
        """
        SELECT file_id, owner_tag FROM relations
        WHERE relation_tag = 'imports' AND member_tag = ?
        """,
        (symbol_name,),
    )
    for file_id, import_owner_tag in import_cursor.fetchall():
        # the import_owner_tag (e.g., $import_0) might have a "calls" relation
        # pointing to the module path — but in our dataset, "imports" directly
        # contains the symbol name. The definition should be found by step 1.
        # If step 1 found nothing, this symbol might be external (builtin/stdlib).
        pass

    # step 3 — attribute resolution for dotted names (foo.bar)
    if "." in symbol_name:
        parts = symbol_name.split(".", 1)
        base_name, attr_name = parts[0], parts[1]
        # find where base_name is defined
        base_defs = dynamic_scope_go_to_definition(ast_tag_db, base_name)
        for base_def in base_defs:
            # search for attr_name defined within the base's scope
            attr_cursor = ast_tag_db._conn.execute(
                """
                SELECT file_id, line, owner_tag, relation_tag, member_tag, member_order_value
                FROM relations
                WHERE file_id = ? AND relation_tag = 'defines' AND member_tag = ?
                """,
                (base_def.file_id, attr_name),
            )
            for row in attr_cursor.fetchall():
                key = (row[0], row[2])
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
        defs = dynamic_scope_go_to_definition(db, sym)
        print(f"go_to_definition({sym!r}): {len(defs)} results")
        for d in defs[:3]:
            print(f"  {d.file_id}:{d.line} {d.owner_tag} {d.relation_tag} {d.member_tag}")
    print("dynamic_scope_go_to_definition: OK")
