"""
ast_tag module — AST tag relation database utilities.
"""

from .ast_tag_db import AstTagDB
from .ast_tag_sqlite_db import (
    AstTagSqliteDB,
    create_ast_tag_sqlite_db_schema,
    load_jsonl_dataset_into_ast_tag_sqlite_db,
)

# Backward compatibility alias
load_jsonl_dataset_into_ast_tag_db = load_jsonl_dataset_into_ast_tag_sqlite_db

__all__ = [
    "AstTagDB",
    "AstTagSqliteDB",
    "create_ast_tag_sqlite_db_schema",
    "load_jsonl_dataset_into_ast_tag_sqlite_db",
    "load_jsonl_dataset_into_ast_tag_db",  # backward compat
]