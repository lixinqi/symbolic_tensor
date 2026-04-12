"""
AstTagSqliteDB — SQLite Implementation of AstTagDB
"""

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from experience.ast_tag.ast_tag_db import AstTagDB


class AstTagSqliteDB(AstTagDB):
    """SQLite implementation of AstTagDB interface."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """
        Wrap an existing sqlite3.Connection — assumes schema already exists and is populated.
        Use load_jsonl_dataset_into_ast_tag_sqlite_db() to create a populated instance.
        """
        self._conn = conn

    # ═══════════════════════════════════════════════════════════
    # AstTagDB interface implementation
    # ═══════════════════════════════════════════════════════════

    def get_all_loaded_file_ids(self) -> list[str]:
        """Enumerate all loaded JSONL files, sorted alphabetically."""
        cursor = self._conn.execute(
            "SELECT DISTINCT file_id FROM relations ORDER BY file_id"
        )
        return [row[0] for row in cursor.fetchall()]

    def count_file_relation_records(self, file_id: str) -> int:
        """Total relation rows in a file — proxy for file complexity."""
        cursor = self._conn.execute(
            "SELECT COUNT(*) FROM relations WHERE file_id = ?", (file_id,)
        )
        return cursor.fetchone()[0]

    def get_nearby_symbols_around_line_range(
        self,
        file_id: str,
        line_range_start: int,
        line_range_end: int,
        context_margin: int = 5,
    ) -> list[str]:
        """Collect symbols from lines surrounding a line range (excluding the range itself)."""
        cursor = self._conn.execute(
            """
            SELECT DISTINCT owner_tag, member_tag FROM relations
            WHERE file_id = ?
              AND line BETWEEN ? AND ?
              AND line NOT BETWEEN ? AND ?
            """,
            (
                file_id,
                line_range_start - context_margin,
                line_range_end + context_margin,
                line_range_start,
                line_range_end,
            ),
        )
        symbols: set[str] = set()
        for owner_tag, member_tag in cursor.fetchall():
            symbols.add(owner_tag)
            symbols.add(member_tag)
        return sorted(symbols)

    def execute_raw_query(
        self, sql_statement: str, query_params: tuple = ()
    ) -> list[tuple]:
        """Raw SQL escape hatch for ad-hoc analysis."""
        cursor = self._conn.execute(sql_statement, query_params)
        return cursor.fetchall()

    # ═══════════════════════════════════════════════════════════
    # Lifecycle methods
    # ═══════════════════════════════════════════════════════════

    def close(self) -> None:
        """Close the underlying sqlite3 connection."""
        self._conn.close()


# ═══════════════════════════════════════════════════════════
# Schema management
# ═══════════════════════════════════════════════════════════

def create_ast_tag_sqlite_db_schema(conn: sqlite3.Connection) -> None:
    """
    Create the relations table and indexes on a bare sqlite3 connection.
    Idempotent: uses CREATE TABLE IF NOT EXISTS.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS relations (
            file_id TEXT,
            line INTEGER,
            relation_tag TEXT,
            owner_tag TEXT,
            member_tag TEXT,
            member_order_value INTEGER
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_owner ON relations(file_id, owner_tag, relation_tag)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_member ON relations(file_id, member_tag, relation_tag)"
    )
    conn.commit()


# ═══════════════════════════════════════════════════════════
# Factory function
# ═══════════════════════════════════════════════════════════

def load_jsonl_dataset_into_ast_tag_sqlite_db(
    dataset_dir: str, db_path: str = ":memory:"
) -> AstTagSqliteDB:
    """
    Walk dataset_dir, load all .jsonl files, INSERT as AstTagRecord rows.
    Returns a ready-to-query AstTagSqliteDB wrapping an in-memory or file-backed DB.
    """
    conn = sqlite3.connect(db_path)
    create_ast_tag_sqlite_db_schema(conn)

    dataset_path = Path(dataset_dir)
    total_records = 0
    for jsonl_file in sorted(dataset_path.rglob("*.jsonl")):
        file_id = str(jsonl_file.relative_to(dataset_path))
        rows = []
        with open(jsonl_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                rows.append((
                    file_id,
                    record["line"],
                    record["relation_tag"],
                    record["owner_tag"],
                    record["member_tag"],
                    record["member_order_value"],
                ))
        conn.executemany(
            "INSERT INTO relations (file_id, line, relation_tag, owner_tag, member_tag, member_order_value) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        total_records += len(rows)

    conn.commit()
    print(f"Loaded {total_records} records from {dataset_dir}")
    return AstTagSqliteDB(conn)


if __name__ == "__main__":
    import sys

    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "test_dataset"
    )
    db = load_jsonl_dataset_into_ast_tag_sqlite_db(dataset_dir)
    file_ids = db.get_all_loaded_file_ids()
    print(f"\n{len(file_ids)} files loaded:")
    for fid in file_ids:
        count = db.count_file_relation_records(fid)
        print(f"  {fid}: {count} records")
    total = sum(db.count_file_relation_records(fid) for fid in file_ids)
    print(f"\nTotal: {total} records across {len(file_ids)} files")
    print("ast_tag_sqlite_db: OK")