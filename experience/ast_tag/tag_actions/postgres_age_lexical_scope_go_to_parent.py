"""
postgres_age_lexical_scope_go_to_parent :=
    CodePosition
    <- $ast_tag_db AstTagPostgresAgeDB
    <- $file_id str
    <- $member_tag str
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
    "parent_symbol agtype",
    "rel_type agtype",
    "member_symbol agtype",
    "member_order_value agtype",
]


def postgres_age_lexical_scope_go_to_parent(
    ast_tag_db: AstTagPostgresAgeDB, file_id: str, member_tag: str
) -> CodePosition:
    """Go to Parent: which scope lexically contains this node?

    PostgreSQL AGE implementation. Single reverse hop, filtered by
    lexical edge labels and r.file_id to handle both file-scoped
    and global member nodes.
    """
    esc_fid = _escape_cypher_string(file_id)
    esc_member = _escape_cypher_string(member_tag)
    graph = ast_tag_db.graph_name

    tag_list = ", ".join(f"'{t}'" for t in sorted(LEXICAL_RELATION_TAGS))

    rows = _cypher_query(
        ast_tag_db._conn,
        graph,
        f"""
        MATCH (parent)-[r]->(member {{symbol: '{esc_member}'}})
        WHERE type(r) IN [{tag_list}]
          AND r.file_id = '{esc_fid}'
        RETURN r.file_id, r.line, parent.symbol, type(r), member.symbol, r.member_order_value
        LIMIT 1
        """,
        columns=_RESULT_COLUMNS,
    )
    if not rows:
        raise ValueError(
            f"No lexical parent found for member_tag={member_tag!r} in file_id={file_id!r}"
        )
    row = rows[0]
    return CodePosition(
        _agtype_to_str(row[0]),
        _agtype_to_int(row[1]),
        _agtype_to_str(row[2]),
        _agtype_to_str(row[3]),
        _agtype_to_str(row[4]),
        _agtype_to_int(row[5]),
    )
