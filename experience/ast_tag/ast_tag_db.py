"""
AstTagDB — Abstract Interface

A database abstraction for querying AST tag relations.
Implementations may use SQLite, PostgreSQL, in-memory structures, etc.
"""

from abc import ABC, abstractmethod
from typing import Any


class AstTagDB(ABC):
    """Abstract interface for AST tag relation databases."""

    # ═══════════════════════════════════════════════════════════
    # Core query methods (must be implemented by subclasses)
    # ═══════════════════════════════════════════════════════════

    @abstractmethod
    def get_all_loaded_file_ids(self) -> list[str]:
        """
        Enumerate all loaded source files, sorted alphabetically.
        Returns empty list if no files loaded.
        """
        ...

    @abstractmethod
    def count_file_relation_records(self, file_id: str) -> int:
        """
        Total relation rows in a file — proxy for file complexity.
        Returns 0 if file_id not found.
        """
        ...

    @abstractmethod
    def get_nearby_symbols_around_line_range(
        self,
        file_id: str,
        line_range_start: int,
        line_range_end: int,
        context_margin: int = 5,
    ) -> list[str]:
        """
        Collect symbols from lines surrounding a line range (excluding the range itself).
        Used to seed the agent's initial queries around a masked region.
        """
        ...

    @abstractmethod
    def execute_raw_query(
        self, query_statement: str, query_params: tuple = ()
    ) -> list[tuple] | dict | Any:
        """
        Raw query escape hatch for ad-hoc analysis.
        Return type depends on implementation.
        """
        ...

    # ═══════════════════════════════════════════════════════════
    # Lifecycle methods (optional, implementation-specific)
    # ═══════════════════════════════════════════════════════════

    def close(self) -> None:
        """Release resources (connections, file handles, etc.). Safe to call multiple times."""
        pass  # Default: no-op

    def __enter__(self) -> "AstTagDB":
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager support — ensures close() is called."""
        self.close()


# Backward compatibility: re-export sqlite implementation
from experience.ast_tag.ast_tag_sqlite_db import (
    AstTagSqliteDB,
    create_ast_tag_sqlite_db_schema,
    load_jsonl_dataset_into_ast_tag_sqlite_db,
)

# Re-export AGE implementation (lazy — psycopg2 may not be installed)
try:
    from experience.ast_tag.ast_tag_postgres_age_db import (
        AstTagPostgresAgeDB,
        classify_node_label,
        create_ast_tag_age_graph,
        load_jsonl_dataset_into_ast_tag_age_db,
    )
except ImportError:
    pass

# Alias for backward compatibility
load_jsonl_dataset_into_ast_tag_db = load_jsonl_dataset_into_ast_tag_sqlite_db

__all__ = [
    "AstTagDB",
    "AstTagSqliteDB",
    "create_ast_tag_sqlite_db_schema",
    "load_jsonl_dataset_into_ast_tag_sqlite_db",
    "AstTagPostgresAgeDB",
    "classify_node_label",
    "create_ast_tag_age_graph",
    "load_jsonl_dataset_into_ast_tag_age_db",
    "load_jsonl_dataset_into_ast_tag_db",
]