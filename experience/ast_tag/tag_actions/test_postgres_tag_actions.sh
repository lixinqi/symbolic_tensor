#!/usr/bin/env bash
set -euo pipefail

# Run all tag_actions tests against PostgreSQL + Apache AGE backend
#
# Environment variables:
#   AST_TAG_PG_CONN   — psycopg2 connection string (default: "dbname=ast_tag")
#   AST_TAG_PG_GRAPH  — AGE graph name (default: "ast_tag")
#
# Prerequisites:
#   - PostgreSQL with Apache AGE extension installed
#   - pip install psycopg2-binary
#   - Database exists: createdb ast_tag
#
# Example:
#   AST_TAG_PG_CONN="host=localhost dbname=ast_tag user=postgres" ./test_postgres_tag_actions.sh

export AST_TAG_DB_BACKEND=postgres_age
export AST_TAG_PG_CONN="${AST_TAG_PG_CONN:-dbname=ast_tag}"
export AST_TAG_PG_GRAPH="${AST_TAG_PG_GRAPH:-ast_tag}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== tag_actions tests: PostgreSQL + AGE backend ==="
echo "  connection: $AST_TAG_PG_CONN"
echo "  graph:      $AST_TAG_PG_GRAPH"
echo

python -m experience.ast_tag.tag_actions.test_dynamic_scope_find_all_references
python -m experience.ast_tag.tag_actions.test_dynamic_scope_go_to_definition
python -m experience.ast_tag.tag_actions.test_lexical_scope_expand_children
python -m experience.ast_tag.tag_actions.test_lexical_scope_go_to_parent

echo
echo "=== All PostgreSQL + AGE tag_actions tests passed ==="
