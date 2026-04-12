#!/usr/bin/env bash
set -euo pipefail

# Run all tag_actions tests against SQLite backend (default)
export AST_TAG_DB_BACKEND=sqlite

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== tag_actions tests: SQLite backend ==="
echo

python -m experience.ast_tag.tag_actions.test_dynamic_scope_find_all_references
python -m experience.ast_tag.tag_actions.test_dynamic_scope_go_to_definition
python -m experience.ast_tag.tag_actions.test_lexical_scope_expand_children
python -m experience.ast_tag.tag_actions.test_lexical_scope_go_to_parent

echo
echo "=== All SQLite tag_actions tests passed ==="
