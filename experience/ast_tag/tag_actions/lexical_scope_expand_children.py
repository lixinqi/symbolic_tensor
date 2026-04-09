"""
lexical_scope_expand_children :=
    # Expand Children: what symbols are lexically contained inside this node?
    list[$child CodePosition]
    <- $ast_tag_db AstTagDB
    <- $file_id str
    <- $owner_tag str
    # inline
    <- LEXICAL_RELATION_TAGS
    <- { SELECT ... WHERE file_id = ? AND owner_tag = ? AND relation_tag IN LEXICAL ORDER BY member_order_value }
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tag_actions.code_position import CodePosition
from relation_tag_classification import LEXICAL_RELATION_TAGS
from ast_tag_db import AstTagDB


def lexical_scope_expand_children(
    ast_tag_db: AstTagDB, file_id: str, owner_tag: str
) -> list[CodePosition]:
    """Expand Children: what symbols are lexically contained inside this node?"""
    placeholders = ",".join("?" for _ in LEXICAL_RELATION_TAGS)
    cursor = ast_tag_db._conn.execute(
        f"""
        SELECT file_id, line, owner_tag, relation_tag, member_tag, member_order_value
        FROM relations
        WHERE file_id = ? AND owner_tag = ?
          AND relation_tag IN ({placeholders})
        ORDER BY member_order_value
        """,
        (file_id, owner_tag, *LEXICAL_RELATION_TAGS),
    )
    return [CodePosition(*row) for row in cursor.fetchall()]


if __name__ == "__main__":
    from ast_tag_db import load_jsonl_dataset_into_ast_tag_db

    dataset_dir = os.path.join(os.path.dirname(__file__), "..", "test_dataset")
    db = load_jsonl_dataset_into_ast_tag_db(dataset_dir)
    file_ids = db.get_all_loaded_file_ids()
    # test on a few owners
    for fid in file_ids[:3]:
        rows = db.execute_raw_sql_query(
            "SELECT DISTINCT owner_tag FROM relations WHERE file_id = ? LIMIT 5",
            (fid,),
        )
        for (owner_tag,) in rows:
            children = lexical_scope_expand_children(db, fid, owner_tag)
            print(f"{fid} / {owner_tag}: {len(children)} lexical children")
            for c in children[:3]:
                print(f"  {c.relation_tag} -> {c.member_tag} (order={c.member_order_value})")
    print("lexical_scope_expand_children: OK")
