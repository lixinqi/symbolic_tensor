"""
AstTagDB := WrapperOf[sqlite3.Connection]

AstTagDB.__init__ :=
    void
    <- $self AstTagDB
    <- $conn sqlite3.Connection
    # inline
    <- (void <- { store conn reference } <- $conn)

create_ast_tag_db_schema :=
    void
    <- $conn sqlite3.Connection
    # inline
    <- { CREATE TABLE IF NOT EXISTS relations ... }

load_jsonl_dataset_into_ast_tag_db :=
    AstTagDB
    <- $dataset_dir str
    <- $db_path str # default ":memory:"
    # inline

AstTagDB.get_all_loaded_file_ids :=
    list[$file_id str]
    <- $self AstTagDB

AstTagDB.count_file_relation_records :=
    int
    <- $self AstTagDB
    <- $file_id str

AstTagDB.get_nearby_symbols_around_line_range :=
    list[$nearby_symbol str]
    <- $self AstTagDB
    <- $file_id str
    <- $line_range_start int
    <- $line_range_end int
    <- $context_margin int # default 5

AstTagDB.execute_raw_sql_query :=
    list[tuple]
    <- $self AstTagDB
    <- $sql_statement str
    <- $query_params tuple # default ()
"""

import json
import os
import sqlite3
from pathlib import Path
from typing import Optional


class AstTagDB:
    """Wrapper around sqlite3.Connection for AST tag relation queries."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

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

    def execute_raw_sql_query(
        self, sql_statement: str, query_params: tuple = ()
    ) -> list[tuple]:
        """Raw SQL escape hatch for ad-hoc analysis."""
        cursor = self._conn.execute(sql_statement, query_params)
        return cursor.fetchall()


def create_ast_tag_db_schema(conn: sqlite3.Connection) -> None:
    """Create the relations table and indexes. Idempotent."""
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


def load_jsonl_dataset_into_ast_tag_db(
    dataset_dir: str, db_path: str = ":memory:"
) -> AstTagDB:
    """Walk dataset_dir, load all .jsonl files, INSERT as rows. Returns ready-to-query AstTagDB."""
    conn = sqlite3.connect(db_path)
    create_ast_tag_db_schema(conn)

    dataset_path = Path(dataset_dir)
    total_records = 0
    for jsonl_file in sorted(dataset_path.rglob("*.jsonl")):
        # file_id = relative path from dataset_dir, e.g. "fs_util/get_nested_list_file_pathes.jsonl"
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
    return AstTagDB(conn)


if __name__ == "__main__":
    import sys

    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "test_dataset"
    )
    db = load_jsonl_dataset_into_ast_tag_db(dataset_dir)
    file_ids = db.get_all_loaded_file_ids()
    print(f"\n{len(file_ids)} files loaded:")
    for fid in file_ids:
        count = db.count_file_relation_records(fid)
        print(f"  {fid}: {count} records")
    total = sum(db.count_file_relation_records(fid) for fid in file_ids)
    print(f"\nTotal: {total} records across {len(file_ids)} files")
    print("ast_tag_db: OK")
