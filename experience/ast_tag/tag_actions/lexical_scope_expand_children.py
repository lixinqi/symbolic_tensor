"""
lexical_scope_expand_children :=
    list[$child CodePosition]
    <- $ast_tag_db AstTagDB
    <- $file_id str
    <- $owner_tag str
    # dispatch on ast_tag_db type
"""

from experience.ast_tag.tag_actions.code_position import CodePosition
from experience.ast_tag.ast_tag_db import AstTagDB
from experience.ast_tag.tag_actions.sqlite_lexical_scope_expand_children import (
    sqlite_lexical_scope_expand_children,
)


def _db_type_name(db: AstTagDB) -> str:
    return type(db).__name__


def lexical_scope_expand_children(
    ast_tag_db: AstTagDB, file_id: str, owner_tag: str
) -> list[CodePosition]:
    """Expand Children: what symbols are lexically contained inside this node?

    Dispatches to the appropriate implementation based on ast_tag_db type.
    """
    name = _db_type_name(ast_tag_db)
    if name == "AstTagSqliteDB":
        return sqlite_lexical_scope_expand_children(ast_tag_db, file_id, owner_tag)
    if name == "AstTagPostgresAgeDB":
        from experience.ast_tag.tag_actions.postgres_age_lexical_scope_expand_children import (
            postgres_age_lexical_scope_expand_children,
        )
        return postgres_age_lexical_scope_expand_children(ast_tag_db, file_id, owner_tag)
    raise NotImplementedError(
        f"lexical_scope_expand_children not implemented for {name}"
    )


if __name__ == "__main__":
    import os
    from experience.ast_tag.ast_tag_db import load_jsonl_dataset_into_ast_tag_db

    dataset_dir = os.path.join(os.path.dirname(__file__), "..", "test_dataset")
    db = load_jsonl_dataset_into_ast_tag_db(dataset_dir)
    file_ids = db.get_all_loaded_file_ids()
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
