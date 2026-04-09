"""
AstTagRecord :=
    DatabaseRecord
    * $file_id str
    * $line int
    * $relation_tag str
    * $owner_tag str
    * $member_tag str
    * $member_order_value int
"""

from typing import NamedTuple


class AstTagRecord(NamedTuple):
    file_id: str
    line: int
    relation_tag: str
    owner_tag: str
    member_tag: str
    member_order_value: int


if __name__ == "__main__":
    r = AstTagRecord(
        file_id="test.jsonl",
        line=1,
        relation_tag="defines",
        owner_tag="$functiondef_0",
        member_tag="pack_dir",
        member_order_value=0,
    )
    print(r)
    assert r.file_id == "test.jsonl"
    assert r.line == 1
    print("AstTagRecord: OK")
