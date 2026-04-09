"""
lexical_scope_go_to_parent :=
    # Go to Parent: which scope lexically contains this node?
    # every AST node has exactly one lexical parent — returns a single position
    CodePosition
    <- $ast_tag_db AstTagDB
    <- $file_id str
    <- $member_tag str
    # inline
    <- LEXICAL_RELATION_TAGS
    <- { SELECT ... WHERE file_id = ? AND member_tag = ? AND relation_tag IN LEXICAL LIMIT 1 }
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tag_actions.code_position import CodePosition
from relation_tag_classification import LEXICAL_RELATION_TAGS
from ast_tag_db import AstTagDB


def lexical_scope_go_to_parent(
    ast_tag_db: AstTagDB, file_id: str, member_tag: str
) -> CodePosition:
    """Go to Parent: which scope lexically contains this node? Returns exactly one CodePosition."""
    placeholders = ",".join("?" for _ in LEXICAL_RELATION_TAGS)
    cursor = ast_tag_db._conn.execute(
        f"""
        SELECT file_id, line, owner_tag, relation_tag, member_tag, member_order_value
        FROM relations
        WHERE file_id = ? AND member_tag = ?
          AND relation_tag IN ({placeholders})
        LIMIT 1
        """,
        (file_id, member_tag, *LEXICAL_RELATION_TAGS),
    )
    row = cursor.fetchone()
    if row is None:
        raise ValueError(
            f"No lexical parent found for member_tag={member_tag!r} in file_id={file_id!r}"
        )
    return CodePosition(*row)


if __name__ == "__main__":
    from ast_tag_db import load_jsonl_dataset_into_ast_tag_db

    dataset_dir = os.path.join(os.path.dirname(__file__), "..", "test_dataset")
    db = load_jsonl_dataset_into_ast_tag_db(dataset_dir)
    file_ids = db.get_all_loaded_file_ids()
    # test on a few members
    for fid in file_ids[:3]:
        rows = db.execute_raw_sql_query(
            "SELECT DISTINCT member_tag FROM relations WHERE file_id = ? LIMIT 5",
            (fid,),
        )
        for (member_tag,) in rows:
            try:
                parent = lexical_scope_go_to_parent(db, fid, member_tag)
                print(f"{fid} / {member_tag} -> parent: {parent.owner_tag} ({parent.relation_tag})")
            except ValueError as e:
                print(f"{fid} / {member_tag} -> {e}")
    print("lexical_scope_go_to_parent: OK")
