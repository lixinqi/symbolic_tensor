"""
AstTagPostgresAgeDB — PostgreSQL + Apache AGE Implementation of AstTagDB

Uses Apache AGE graph extension for openCypher queries.
Each relation_tag maps to an edge label in the graph.

Graph name: ast_tag (default)

Node labels: root, inner, leaf_symbol, leaf_literal
Edge labels: one per relation_tag (e.g. FunctionDef__body, Call__func)
"""

import json
import os
import re
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras

from experience.ast_tag.ast_tag_db import AstTagDB
from experience.ast_tag.relation_tag_classification import (
    LEXICAL_RELATION_TAGS,
    DYNAMIC_RELATION_TAGS,
)


# ═══════════════════════════════════════════════════════════
# Node label classification
# ═══════════════════════════════════════════════════════════

_LITERAL_KEYWORDS = frozenset({"True", "False", "None", "..."})
_LITERAL_PREFIX_RE = re.compile(
    r"""^(?:
        [+-]?\d+(?:\.\d*)?          # int or float: 42, 3.14, -1
        | [+-]?\.\d+                # float: .5
        | '.*'                      # single-quoted repr
        | ".*"                      # double-quoted repr
        | b'.*'                     # bytes repr
        | b".*"                     # bytes repr
    )$""",
    re.VERBOSE | re.DOTALL,
)


def classify_node_label(symbol: str) -> str:
    """Classify a symbol string into its AGE node label."""
    if symbol == "<module>":
        return "root"
    if symbol.startswith("$"):
        return "inner"
    if symbol in _LITERAL_KEYWORDS:
        return "leaf_literal"
    if _LITERAL_PREFIX_RE.match(symbol):
        return "leaf_literal"
    # Try numeric parse as fallback
    try:
        float(symbol)
        return "leaf_literal"
    except (ValueError, TypeError):
        pass
    return "leaf_symbol"


def _is_file_scoped(label: str) -> bool:
    """root and inner nodes are file-scoped; leaves are global."""
    return label in ("root", "inner")


# ═══════════════════════════════════════════════════════════
# AGE Cypher helpers
# ═══════════════════════════════════════════════════════════

def _age_setup(conn: "psycopg2.extensions.connection") -> None:
    """Load AGE extension and set search_path for a connection."""
    with conn.cursor() as cur:
        cur.execute("LOAD 'age'")
        cur.execute("SET search_path = ag_catalog, \"$user\", public")
    conn.commit()


def _cypher_query(
    conn: "psycopg2.extensions.connection",
    graph_name: str,
    cypher: str,
    params: tuple = (),
    columns: list[str] | None = None,
) -> list[tuple]:
    """Execute a Cypher query via ag_catalog.cypher() and return rows as tuples."""
    if columns is None:
        columns = ["v agtype"]
    col_def = ", ".join(columns)
    sql = f"SELECT * FROM cypher('{graph_name}', $${cypher}$$) as t({col_def})"
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    except Exception:
        conn.rollback()
        raise


def _cypher_exec(
    conn: "psycopg2.extensions.connection",
    graph_name: str,
    cypher: str,
) -> None:
    """Execute a Cypher statement with no result set."""
    sql = f"SELECT * FROM cypher('{graph_name}', $${cypher}$$) as t(v agtype)"
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def _escape_cypher_string(s: str) -> str:
    """Escape a string for safe embedding in Cypher literals."""
    return s.replace("\\", "\\\\").replace("'", "\\'")


# ═══════════════════════════════════════════════════════════
# AstTagPostgresAgeDB class
# ═══════════════════════════════════════════════════════════

class AstTagPostgresAgeDB(AstTagDB):
    """PostgreSQL + Apache AGE implementation of AstTagDB interface."""

    def __init__(
        self, conn: "psycopg2.extensions.connection", graph_name: str = "ast_tag"
    ) -> None:
        """
        Wrap an existing psycopg2 connection with AGE extension loaded.
        Assumes graph already exists and is populated.
        Use load_jsonl_dataset_into_ast_tag_age_db() to create a populated instance.
        """
        self._conn = conn
        self._graph_name = graph_name

    @property
    def graph_name(self) -> str:
        return self._graph_name

    # ═══════════════════════════════════════════════════════════
    # AstTagDB interface implementation
    # ═══════════════════════════════════════════════════════════

    def get_all_loaded_file_ids(self) -> list[str]:
        """Enumerate all loaded source files, sorted alphabetically."""
        rows = _cypher_query(
            self._conn,
            self._graph_name,
            "MATCH (r:root) RETURN DISTINCT r.file_id ORDER BY r.file_id",
            columns=["file_id agtype"],
        )
        return [_agtype_to_str(row[0]) for row in rows]

    def count_file_relation_records(self, file_id: str) -> int:
        """Total edges in a file — proxy for file complexity."""
        esc_fid = _escape_cypher_string(file_id)
        rows = _cypher_query(
            self._conn,
            self._graph_name,
            f"MATCH ()-[r]->() WHERE r.file_id = '{esc_fid}' RETURN count(r)",
            columns=["cnt agtype"],
        )
        return _agtype_to_int(rows[0][0]) if rows else 0

    def get_nearby_symbols_around_line_range(
        self,
        file_id: str,
        line_range_start: int,
        line_range_end: int,
        context_margin: int = 5,
    ) -> list[str]:
        """Collect symbols from edges near a line range (excluding the range itself)."""
        esc_fid = _escape_cypher_string(file_id)
        lo = line_range_start - context_margin
        hi = line_range_end + context_margin
        rows = _cypher_query(
            self._conn,
            self._graph_name,
            f"""
            MATCH (owner)-[r]->(member)
            WHERE r.file_id = '{esc_fid}'
              AND r.line >= {lo} AND r.line <= {hi}
              AND NOT (r.line >= {line_range_start} AND r.line <= {line_range_end})
            RETURN DISTINCT owner.symbol, member.symbol
            """,
            columns=["owner_sym agtype", "member_sym agtype"],
        )
        symbols: set[str] = set()
        for owner_sym, member_sym in rows:
            symbols.add(_agtype_to_str(owner_sym))
            symbols.add(_agtype_to_str(member_sym))
        return sorted(symbols)

    def execute_raw_query(
        self, query_statement: str, query_params: tuple = ()
    ) -> list[tuple]:
        """Raw Cypher query escape hatch for ad-hoc analysis."""
        with self._conn.cursor() as cur:
            cur.execute(query_statement, query_params)
            return cur.fetchall()

    # ═══════════════════════════════════════════════════════════
    # Lifecycle
    # ═══════════════════════════════════════════════════════════

    def close(self) -> None:
        """Close the underlying psycopg2 connection."""
        self._conn.close()


# ═══════════════════════════════════════════════════════════
# agtype parsing helpers
# ═══════════════════════════════════════════════════════════

def _agtype_to_str(val: Any) -> str:
    """Extract a Python str from an agtype value."""
    if isinstance(val, str):
        # AGE returns agtype as strings like '"value"' — strip quotes
        if val.startswith('"') and val.endswith('"'):
            return val[1:-1]
        return val
    return str(val)


def _agtype_to_int(val: Any) -> int:
    """Extract a Python int from an agtype value."""
    if isinstance(val, int):
        return val
    return int(str(val))


# ═══════════════════════════════════════════════════════════
# Schema management
# ═══════════════════════════════════════════════════════════

def create_ast_tag_age_graph(
    conn: "psycopg2.extensions.connection", graph_name: str = "ast_tag"
) -> None:
    """
    Create the AGE graph and vertex/edge labels.
    Idempotent: checks existence before creating.
    """
    _age_setup(conn)

    # Create graph if not exists
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM ag_catalog.ag_graph WHERE name = %s",
            (graph_name,),
        )
        if cur.fetchone()[0] == 0:
            cur.execute("SELECT create_graph(%s)", (graph_name,))
    conn.commit()

    # Create vertex labels
    for vlabel in ("root", "inner", "leaf_symbol", "leaf_literal"):
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM ag_catalog.ag_label "
                "WHERE name = %s AND graph = (SELECT graphid FROM ag_catalog.ag_graph WHERE name = %s)",
                (vlabel, graph_name),
            )
            if cur.fetchone()[0] == 0:
                cur.execute(f"SELECT create_vlabel('{graph_name}', '{vlabel}')")
        conn.commit()

    # Create edge labels — one per relation_tag
    all_relation_tags = LEXICAL_RELATION_TAGS | DYNAMIC_RELATION_TAGS
    for elabel in sorted(all_relation_tags):
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM ag_catalog.ag_label "
                "WHERE name = %s AND graph = (SELECT graphid FROM ag_catalog.ag_graph WHERE name = %s)",
                (elabel, graph_name),
            )
            if cur.fetchone()[0] == 0:
                cur.execute(f"SELECT create_elabel('{graph_name}', '{elabel}')")
        conn.commit()


# ═══════════════════════════════════════════════════════════
# Factory function
# ═══════════════════════════════════════════════════════════

def load_jsonl_dataset_into_ast_tag_age_db(
    dataset_dir: str,
    conn_params: str = "dbname=ast_tag",
    graph_name: str = "ast_tag",
) -> AstTagPostgresAgeDB:
    """
    Walk dataset_dir, load all .jsonl files into the AGE graph.
    Returns a ready-to-query AstTagPostgresAgeDB.
    """
    conn = psycopg2.connect(conn_params)
    conn.autocommit = False

    # Drop existing graph to avoid duplicate edges on re-load
    _age_setup(conn)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM ag_catalog.ag_graph WHERE name = %s",
            (graph_name,),
        )
        if cur.fetchone()[0] > 0:
            cur.execute("SELECT drop_graph(%s, true)", (graph_name,))
    conn.commit()
    # Reconnect — drop_graph(cascade) can invalidate the connection
    conn.close()
    conn = psycopg2.connect(conn_params)
    conn.autocommit = False

    create_ast_tag_age_graph(conn, graph_name)
    _age_setup(conn)

    dataset_path = Path(dataset_dir)
    total_records = 0

    for jsonl_file in sorted(dataset_path.rglob("*.jsonl")):
        file_id = str(jsonl_file.relative_to(dataset_path))
        esc_fid = _escape_cypher_string(file_id)

        with open(jsonl_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                owner_tag = str(record["owner_tag"])
                member_tag = str(record["member_tag"])
                relation_tag = record["relation_tag"]
                line_no = record["line"]
                member_order_value = record["member_order_value"]

                owner_label = classify_node_label(owner_tag)
                member_label = classify_node_label(member_tag)
                esc_owner = _escape_cypher_string(owner_tag)
                esc_member = _escape_cypher_string(member_tag)

                # Build MERGE for owner node
                if _is_file_scoped(owner_label):
                    owner_merge = (
                        f"MERGE (owner:{owner_label} "
                        f"{{symbol: '{esc_owner}', file_id: '{esc_fid}'}})"
                    )
                else:
                    owner_merge = (
                        f"MERGE (owner:{owner_label} {{symbol: '{esc_owner}'}})"
                    )

                # Build MERGE for member node
                if _is_file_scoped(member_label):
                    member_merge = (
                        f"MERGE (member:{member_label} "
                        f"{{symbol: '{esc_member}', file_id: '{esc_fid}'}})"
                    )
                else:
                    member_merge = (
                        f"MERGE (member:{member_label} {{symbol: '{esc_member}'}})"
                    )

                cypher_stmt = (
                    f"{owner_merge} "
                    f"{member_merge} "
                    f"CREATE (owner)-[:{relation_tag} {{"
                    f"file_id: '{esc_fid}', "
                    f"line: {line_no}, "
                    f"member_order_value: {member_order_value}"
                    f"}}]->(member)"
                )

                sql = (
                    f"SELECT * FROM cypher('{graph_name}', $${cypher_stmt}$$) "
                    f"as t(v agtype)"
                )
                with conn.cursor() as cur:
                    cur.execute(sql)
                total_records += 1

        conn.commit()

    print(f"Loaded {total_records} records into graph '{graph_name}' from {dataset_dir}")
    return AstTagPostgresAgeDB(conn, graph_name)


if __name__ == "__main__":
    import sys

    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "test_dataset"
    )
    conn_params = sys.argv[2] if len(sys.argv) > 2 else "dbname=ast_tag"
    db = load_jsonl_dataset_into_ast_tag_age_db(dataset_dir, conn_params)
    file_ids = db.get_all_loaded_file_ids()
    print(f"\n{len(file_ids)} files loaded:")
    for fid in file_ids:
        count = db.count_file_relation_records(fid)
        print(f"  {fid}: {count} records")
    total = sum(db.count_file_relation_records(fid) for fid in file_ids)
    print(f"\nTotal: {total} records across {len(file_ids)} files")
    print("ast_tag_postgres_age_db: OK")
