"""
postgres_age_dynamic_scope_go_to_definition :=
    list[$definition_site CodePosition]
    <- $ast_tag_db AstTagPostgresAgeDB
    <- $symbol_name str
    # inline — PostgreSQL AGE graph implementation
"""

from experience.ast_tag.tag_actions.code_position import CodePosition
from experience.ast_tag.ast_tag_postgres_age_db import (
    AstTagPostgresAgeDB,
    _cypher_query,
    _escape_cypher_string,
    _agtype_to_str,
    _agtype_to_int,
)


DEFINITION_EDGE_LABELS = frozenset({
    "FunctionDef__name",
    "AsyncFunctionDef__name",
    "ClassDef__name",
    "alias__name",
})


_RESULT_COLUMNS = [
    "file_id agtype",
    "line agtype",
    "owner_symbol agtype",
    "rel_type agtype",
    "member_symbol agtype",
    "member_order_value agtype",
]


def postgres_age_dynamic_scope_go_to_definition(
    ast_tag_db: AstTagPostgresAgeDB, symbol_name: str
) -> list[CodePosition]:
    """Go to Definition: where is this symbol defined?

    PostgreSQL AGE implementation. Traverses incoming definition edges
    to the global leaf node for the symbol.
    """
    results: list[CodePosition] = []
    seen: set[tuple[str, str]] = set()  # (file_id, owner_tag) dedup
    esc_sym = _escape_cypher_string(symbol_name)
    graph = ast_tag_db.graph_name

    # Edge label filter using WHERE type(r) IN [...] (AGE doesn't support pipe syntax)
    tag_list = ", ".join(f"'{t}'" for t in sorted(DEFINITION_EDGE_LABELS))

    rows = _cypher_query(
        ast_tag_db._conn,
        graph,
        f"""
        MATCH (owner)-[r]->(member {{symbol: '{esc_sym}'}})
        WHERE type(r) IN [{tag_list}]
        RETURN r.file_id, r.line, owner.symbol, type(r), member.symbol, r.member_order_value
        """,
        columns=_RESULT_COLUMNS,
    )
    for row in rows:
        fid = _agtype_to_str(row[0])
        ln = _agtype_to_int(row[1])
        owner_sym = _agtype_to_str(row[2])
        rel_type = _agtype_to_str(row[3])
        member_sym = _agtype_to_str(row[4])
        mov = _agtype_to_int(row[5])
        key = (fid, owner_sym)
        if key not in seen:
            seen.add(key)
            results.append(CodePosition(fid, ln, owner_sym, rel_type, member_sym, mov))

    return results
