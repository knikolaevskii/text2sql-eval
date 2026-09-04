"""
Generates tiny WikiSQL- and Defog-shaped datasets under tests/fixtures/.

Neither of those benchmarks auto-downloads, so without real data on disk their
code paths can't be exercised at all. These fixtures are a few rows each and
exist to prove the plumbing works end to end:

    python source/code/run_eval.py --dataset wikisql --backend openrouter \
        --model gpt-4o-mini --dataset-folder tests/fixtures

They mirror the layouts the dataset classes expect:

  wikisql/                          schema_source="metadata_file"
    test.json                         rows: question / query / db_id
    test_wiki_sql_metadata.json       {"table_metadata": {db_id: [columns]}}
    database/test.db                  one SQLite file holding every table

  defog/                            schema_source="schema_path"
    questions_gen.json                rows: question / query / db_id
    database/<db_id>/<db_id>.json     {"table_metadata": {...}} per db_id
    database/<db_id>/<db_id>.sqlite   one database per db_id

Run from the repository root:
    python tests/make_fixtures.py
"""
import json
import sqlite3
from pathlib import Path

ROOT = Path("tests/fixtures")

# Mirrors the conventions of the real conversion pipeline (see
# previous_work/approach1/WikiSQL/conversion_test/):
#   - table id "1-10015132-16" becomes "table_1_10015132_16"
#   - physical columns are positional (col0, col1, ...); the human-readable
#     header survives only as column_description, which is what actually tells
#     a model what col2 means. Omitting it makes the task unanswerable.
#   - gold queries are "SELECT colN AS result FROM ... WHERE colM = '...'"
#     with string literals lowercased, and questions are lowercased too
WIKISQL_TABLE = "table_1_10015132_16"
WIKISQL_HEADERS = [("City", "text"), ("Country", "text"), ("Population", "real")]
WIKISQL_COLUMNS = [
    {"data_type": dtype, "column_name": f"col{i}", "column_description": header}
    for i, (header, dtype) in enumerate(WIKISQL_HEADERS)
]
WIKISQL_ROWS = [
    ("berlin", "germany", 3600000),
    ("paris", "france", 2100000),
    ("madrid", "spain", 3200000),
]
WIKISQL_QUESTIONS = [
    {
        "question": "what is the country for berlin",
        "query": f"SELECT col1 AS result FROM {WIKISQL_TABLE} WHERE col0 = 'berlin';",
        "db_id": WIKISQL_TABLE,
    },
    {
        "question": "what is the population of madrid",
        "query": f"SELECT col2 AS result FROM {WIKISQL_TABLE} WHERE col0 = 'madrid';",
        "db_id": WIKISQL_TABLE,
    },
]

DEFOG_DB = "fixture_shop"
DEFOG_SCHEMA = {
    "products": [
        {"column_name": "id", "data_type": "integer"},
        {"column_name": "name", "data_type": "text"},
        {"column_name": "price", "data_type": "integer"},
    ],
    "orders": [
        {"column_name": "id", "data_type": "integer"},
        {"column_name": "product_id", "data_type": "integer"},
        {"column_name": "quantity", "data_type": "integer"},
    ],
}
# Defog gold answers are sets of acceptable queries, not single strings. Two
# conventions from questions_gen_*.csv are reproduced here because the
# evaluator has dedicated handling for both (see get_all_minimal_queries):
#   - several full queries separated by semicolons, any of which counts
#   - {col_a,col_b} marking interchangeable column selections, expanded into
#     every subset. Defog writes these without a space after the comma.
DEFOG_QUESTIONS = [
    {
        "question": "What is the name of the most expensive product?",
        "query": "SELECT name FROM products ORDER BY price DESC LIMIT 1;",
        "db_id": DEFOG_DB,
        "query_category": "order_by",
    },
    {
        "question": "How many orders are there in total?",
        "query": "SELECT COUNT(*) FROM orders;",
        "db_id": DEFOG_DB,
        "query_category": "group_by",
    },
    {
        # semicolon-separated alternatives: either projection is acceptable
        "question": "Which product has the most orders?",
        "query": (
            "SELECT p.name FROM products p JOIN orders o ON p.id = o.product_id "
            "GROUP BY p.name ORDER BY COUNT(*) DESC LIMIT 1;"
            "SELECT p.id FROM products p JOIN orders o ON p.id = o.product_id "
            "GROUP BY p.id ORDER BY COUNT(*) DESC LIMIT 1;"
        ),
        "db_id": DEFOG_DB,
        "query_category": "table_join",
    },
    {
        # brace notation: name, id, or both may be selected
        "question": "List the products that cost more than 100.",
        "query": "SELECT {name,id} FROM products WHERE price > 100;",
        "db_id": DEFOG_DB,
        "query_category": "instruct",
    },
]


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def build_wikisql() -> None:
    base = ROOT / "wikisql"
    write_json(base / "test.json", WIKISQL_QUESTIONS)
    write_json(
        base / "test_wiki_sql_metadata.json",
        {"table_metadata": {WIKISQL_TABLE: WIKISQL_COLUMNS}, "glossary": ""},
    )

    db_path = base / "database" / "test.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(f"CREATE TABLE {WIKISQL_TABLE} (col0 text, col1 text, col2 real)")
    conn.executemany(f"INSERT INTO {WIKISQL_TABLE} VALUES (?, ?, ?)", WIKISQL_ROWS)
    conn.commit()
    conn.close()
    print(f"wrote {base}")


def build_defog() -> None:
    base = ROOT / "defog"
    write_json(base / "questions_gen.json", DEFOG_QUESTIONS)

    db_dir = base / "database" / DEFOG_DB
    db_dir.mkdir(parents=True, exist_ok=True)
    write_json(db_dir / f"{DEFOG_DB}.json", {"table_metadata": DEFOG_SCHEMA, "glossary": ""})

    db_path = db_dir / f"{DEFOG_DB}.sqlite"
    db_path.unlink(missing_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE products (id INTEGER, name TEXT, price INTEGER)")
    conn.executemany(
        "INSERT INTO products VALUES (?, ?, ?)",
        [(1, "desk", 300), (2, "chair", 120), (3, "lamp", 45)],
    )
    conn.execute("CREATE TABLE orders (id INTEGER, product_id INTEGER, quantity INTEGER)")
    conn.executemany(
        "INSERT INTO orders VALUES (?, ?, ?)",
        [(1, 1, 2), (2, 2, 5), (3, 1, 1), (4, 3, 7)],
    )
    conn.commit()
    conn.close()
    print(f"wrote {base}")


if __name__ == "__main__":
    build_wikisql()
    build_defog()
