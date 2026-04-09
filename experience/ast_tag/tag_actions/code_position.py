"""
CodePosition :=
    # a single row from the relations table, representing one edge in the AST graph
    Object
    * $file_id str
    * $line int
    * $owner_tag str
    * $relation_tag str
    * $member_tag str
    * $member_order_value int
"""

from typing import NamedTuple


class CodePosition(NamedTuple):
    """A single row from the relations table, representing one edge in the AST graph."""
    file_id: str
    line: int
    owner_tag: str
    relation_tag: str
    member_tag: str
    member_order_value: int


if __name__ == "__main__":
    cp = CodePosition(
        file_id="test.jsonl",
        line=5,
        owner_tag="$functiondef_0",
        relation_tag="param",
        member_tag="root_dir",
        member_order_value=0,
    )
    print(cp)
    assert cp.file_id == "test.jsonl"
    assert cp.line == 5
    print("CodePosition: OK")
