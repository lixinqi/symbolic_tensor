"""
lexical_scope_go_to_parent :=
    CodePosition
    <- $ast_tag_db AstTagDB
    <- $file_id str
    <- $member_tag str
    # dispatch on ast_tag_db type
"""

from experience.ast_tag.tag_actions.code_position import CodePosition
from experience.ast_tag.ast_tag_db import AstTagDB
from experience.ast_tag.tag_actions.sqlite_lexical_scope_go_to_parent import (
    sqlite_lexical_scope_go_to_parent,
)


def _db_type_name(db: AstTagDB) -> str:
    return type(db).__name__


def lexical_scope_go_to_parent(
    ast_tag_db: AstTagDB, file_id: str, member_tag: str
) -> CodePosition:
    """Go to Parent: which scope lexically contains this node?

    Dispatches to the appropriate implementation based on ast_tag_db type.
    """
    name = _db_type_name(ast_tag_db)
    if name == "AstTagSqliteDB":
        return sqlite_lexical_scope_go_to_parent(ast_tag_db, file_id, member_tag)
    if name == "AstTagPostgresAgeDB":
        from experience.ast_tag.tag_actions.postgres_age_lexical_scope_go_to_parent import (
            postgres_age_lexical_scope_go_to_parent,
        )
        return postgres_age_lexical_scope_go_to_parent(ast_tag_db, file_id, member_tag)
    raise NotImplementedError(
        f"lexical_scope_go_to_parent not implemented for {name}"
    )


if __name__ == "__main__":
    import os
    from experience.ast_tag.ast_tag_db import load_jsonl_dataset_into_ast_tag_db

    dataset_dir = os.path.join(os.path.dirname(__file__), "..", "test_dataset")
    db = load_jsonl_dataset_into_ast_tag_db(dataset_dir)
    file_ids = db.get_all_loaded_file_ids()
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
