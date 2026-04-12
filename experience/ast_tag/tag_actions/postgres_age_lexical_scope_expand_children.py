"""
postgres_age_lexical_scope_expand_children :=
    list[$child CodePosition]
    <- $ast_tag_db AstTagPostgresAgeDB
    <- $file_id str
    <- $owner_tag str
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
from experience.ast_tag.relation_tag_classification import LEXICAL_RELATION_TAGS


_RESULT_COLUMNS = [
    "file_id agtype",
    "line agtype",
    "owner_symbol agtype",
    "rel_type agtype",
    "child_symbol agtype",
    "member_order_value agtype",
]


def postgres_age_lexical_scope_expand_children(
    ast_tag_db: AstTagPostgresAgeDB, file_id: str, owner_tag: str
) -> list[CodePosition]:
    """Expand Children: what symbols are lexically contained inside this node?

    PostgreSQL AGE implementation. Single hop from owner node,
    filtered by lexical edge labels.
    """
    esc_fid = _escape_cypher_string(file_id)
    esc_owner = _escape_cypher_string(owner_tag)
    graph = ast_tag_db.graph_name

    tag_list = ", ".join(f"'{t}'" for t in sorted(LEXICAL_RELATION_TAGS))

    rows = _cypher_query(
        ast_tag_db._conn,
        graph,
        f"""
        MATCH (owner {{symbol: '{esc_owner}', file_id: '{esc_fid}'}})-[r]->(child)
        WHERE type(r) IN [{tag_list}]
          AND r.file_id = '{esc_fid}'
        RETURN r.file_id, r.line, owner.symbol, type(r), child.symbol, r.member_order_value
        ORDER BY r.member_order_value, type(r)
        """,
        columns=_RESULT_COLUMNS,
    )
    return [
        CodePosition(
            _agtype_to_str(row[0]),
            _agtype_to_int(row[1]),
            _agtype_to_str(row[2]),
            _agtype_to_str(row[3]),
            _agtype_to_str(row[4]),
            _agtype_to_int(row[5]),
        )
        for row in rows
    ]
