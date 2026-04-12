"""
dynamic_scope_find_all_references :=
    list[$reference_site CodePosition]
    <- $ast_tag_db AstTagDB
    <- $symbol_name str
    # dispatch on ast_tag_db type
"""

from experience.ast_tag.tag_actions.code_position import CodePosition
from experience.ast_tag.ast_tag_db import AstTagDB
from experience.ast_tag.tag_actions.sqlite_dynamic_scope_find_all_references import (
    sqlite_dynamic_scope_find_all_references,
)


def _db_type_name(db: AstTagDB) -> str:
    return type(db).__name__


def dynamic_scope_find_all_references(
    ast_tag_db: AstTagDB, symbol_name: str
) -> list[CodePosition]:
    """Find All References: who uses/calls/references this symbol?

    Dispatches to the appropriate implementation based on ast_tag_db type.
    """
    name = _db_type_name(ast_tag_db)
    if name == "AstTagSqliteDB":
        return sqlite_dynamic_scope_find_all_references(ast_tag_db, symbol_name)
    if name == "AstTagPostgresAgeDB":
        from experience.ast_tag.tag_actions.postgres_age_dynamic_scope_find_all_references import (
            postgres_age_dynamic_scope_find_all_references,
        )
        return postgres_age_dynamic_scope_find_all_references(ast_tag_db, symbol_name)
    raise NotImplementedError(
        f"dynamic_scope_find_all_references not implemented for {name}"
    )


if __name__ == "__main__":
    import os
    from experience.ast_tag.ast_tag_db import load_jsonl_dataset_into_ast_tag_db

    dataset_dir = os.path.join(os.path.dirname(__file__), "..", "test_dataset")
    db = load_jsonl_dataset_into_ast_tag_db(dataset_dir)
    rows = db.execute_raw_sql_query(
        "SELECT DISTINCT member_tag FROM relations WHERE relation_tag = 'FunctionDef__name' LIMIT 10"
    )
    for (sym,) in rows:
        refs = dynamic_scope_find_all_references(db, sym)
        print(f"find_all_references({sym!r}): {len(refs)} results")
        for r in refs[:3]:
            print(f"  {r.file_id}:{r.line} {r.owner_tag} {r.relation_tag} {r.member_tag}")
    print("dynamic_scope_find_all_references: OK")
