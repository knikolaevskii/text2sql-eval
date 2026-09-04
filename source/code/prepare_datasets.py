"""
Converts the WikiSQL and Defog benchmarks into the layout premsql expects.

Bird and Spider download themselves on first use. These two do not: they are
distributed in their own formats and need converting, which is why
`--dataset wikisql` / `--dataset defog` otherwise fail with "dataset not
found".

Run from the repository root.

WikiSQL
-------
Clone the benchmark and unpack its data, then convert:

    git clone https://github.com/salesforce/WikiSQL /tmp/WikiSQL
    tar -xvjf /tmp/WikiSQL/data.tar.bz2 -C /tmp/WikiSQL

    python source/code/prepare_datasets.py wikisql --source /tmp/WikiSQL

WikiSQL does not ship SQL strings. Each question carries a structured form
({sel, agg, conds}) which is rendered into SQL here, following the same rules
the benchmark's own evaluator uses:

  - table id "1-10015132-16" becomes table name "table_1_10015132_16"
  - columns are positional (col0, col1, ...); the human-readable header is
    preserved only as column_description in the metadata, which is the sole
    signal a model has for what a column means
  - string literals in the gold query are lowercased, and so are questions.
    This is a property of the benchmark, not a stylistic choice: predictions
    that differ only by case would otherwise be scored wrong, which is what
    the WikiSQL generator's lowercasing postprocessing compensates for.
  - questions containing characters outside a conservative safe set are
    skipped, and the count is reported

Defog
-----
Clone both repositories, then convert:

    git clone https://github.com/defog-ai/defog-data /tmp/defog-data
    git clone https://github.com/defog-ai/sql-eval /tmp/sql-eval

    python source/code/prepare_datasets.py defog \
        --defog-data /tmp/defog-data --sql-eval /tmp/sql-eval

Defog is distributed as Postgres dumps, and evaluation runs against a live
Postgres server. Load the databases first:

    cd /tmp/defog-data && ./setup.sh

That creates 11 databases using DBUSER/DBPASSWORD/DBHOST/DBPORT (defaulting
to postgres/postgres/localhost/5432); put matching values in .env as
POSTGRES_* so the executor can reach them. Then:

    python source/code/run_eval.py --dataset defog --backend openrouter \
        --model gpt-4o-mini --executor postgres --num-rows 10

Gold answers here are sets, not single queries: several full queries separated
by semicolons, and {col_a,col_b} marking interchangeable column selections.
Both are preserved verbatim — the evaluator expands them.
"""
import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

OUT_ROOT = Path("source/datasets")

# WikiSQL's own operator tables (lib/query.py). Indices in the structured form
# refer into these.
AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
COND_OPS = ["=", ">", "<", "OP"]

SAFE_QUESTION = re.compile(r"^[A-Za-z0-9 .,!?'\"()\[\]{}:;\-]+$")
NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+")


def table_name_for(table_id: str) -> str:
    return table_id if table_id.startswith("table") else f"table_{table_id.replace('-', '_')}"


def to_number(value):
    """
    WikiSQL compares against numeric columns using values lifted from question
    text, so "1,234" and "12.5 km" both appear. Strip separators, then fall
    back to the first number in the string.
    """
    if isinstance(value, (int, float)):
        return value
    text = str(value)
    for candidate in (text, text.replace(",", "")):
        try:
            return float(candidate)
        except ValueError:
            pass
    found = NUMBER_RE.findall(text)
    return float(found[0]) if found else value


def sql_literal(value) -> str:
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def build_query(table: str, column_types: list[str], sql_spec: dict) -> str:
    select = f"col{sql_spec['sel']}"
    agg = AGG_OPS[sql_spec["agg"]]
    if agg:
        select = f"{agg}({select})"

    conditions = []
    for col_index, op_index, value in sql_spec.get("conds", []):
        if isinstance(value, str):
            value = value.lower()
        if column_types[col_index] == "real":
            value = to_number(value)
        conditions.append(f"col{col_index} {COND_OPS[op_index]} {sql_literal(value)}")

    where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    return f"SELECT {select} AS result FROM {table}{where};"


def prepare_wikisql(source: Path, split: str) -> None:
    data = source / "data"
    questions_file = data / f"{split}.jsonl"
    tables_file = data / f"{split}.tables.jsonl"
    db_file = data / f"{split}.db"

    missing = [p for p in (questions_file, tables_file, db_file) if not p.exists()]
    if missing:
        sys.exit(
            "Missing WikiSQL data: "
            + ", ".join(str(p) for p in missing)
            + "\nUnpack it first:  tar -xvjf data.tar.bz2 -C <source>"
        )

    # tables.jsonl carries both the headers and the column types, so the
    # schema can be built without opening the SQLite file.
    metadata, types_by_table = {}, {}
    with open(tables_file) as handle:
        for line in handle:
            entry = json.loads(line)
            table = table_name_for(entry["id"])
            metadata[table] = [
                {
                    "data_type": dtype,
                    "column_name": f"col{index}",
                    "column_description": header,
                }
                for index, (header, dtype) in enumerate(zip(entry["header"], entry["types"]))
            ]
            types_by_table[table] = entry["types"]

    rows, skipped = [], 0
    with open(questions_file) as handle:
        for line in handle:
            entry = json.loads(line)
            question = entry["question"].lower()
            if not SAFE_QUESTION.match(question):
                skipped += 1
                continue
            table = table_name_for(entry["table_id"])
            if table not in types_by_table:
                skipped += 1
                continue
            rows.append(
                {
                    "db_id": table,
                    "query": build_query(table, types_by_table[table], entry["sql"]),
                    "question": question,
                }
            )

    out = OUT_ROOT / "wikisql"
    (out / "database").mkdir(parents=True, exist_ok=True)
    (out / "test.json").write_text(json.dumps(rows, indent=2))
    (out / "test_wiki_sql_metadata.json").write_text(
        json.dumps({"table_metadata": metadata, "glossary": ""}, indent=2)
    )
    shutil.copy(db_file, out / "database" / "test.db")

    print(f"wikisql -> {out}")
    print(f"  {len(rows)} questions across {len(metadata)} tables")
    print(f"  {skipped} skipped (unsafe characters or missing table)")
    print(f"  copied {db_file.name} ({db_file.stat().st_size / 1e6:.1f} MB)")


def prepare_defog(defog_data: Path, sql_eval: Path, questions_csv: str) -> None:
    schema_root = defog_data / "defog_data"
    questions_file = sql_eval / "data" / questions_csv
    if not schema_root.is_dir():
        sys.exit(f"Not found: {schema_root}")
    if not questions_file.exists():
        sys.exit(f"Not found: {questions_file}")

    out = OUT_ROOT / "defog"
    (out / "database").mkdir(parents=True, exist_ok=True)

    copied = []
    for schema_file in sorted(schema_root.glob("*/*.json")):
        db_id = schema_file.parent.name
        target = out / "database" / db_id / f"{db_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(schema_file, target)
        copied.append(db_id)

    rows = []
    with open(questions_file, newline="") as handle:
        for record in csv.DictReader(handle):
            rows.append(
                {
                    "db_id": record["db_name"],
                    # Kept verbatim: may hold several ;-separated alternatives
                    # and {col_a,col_b} column choices, both of which the
                    # evaluator expands into acceptable answers.
                    "query": record["query"],
                    "question": record["question"],
                    "query_category": record.get("query_category", ""),
                    "instructions": record.get("instructions", ""),
                }
            )

    (out / "questions_gen.json").write_text(json.dumps(rows, indent=2))

    unknown = sorted({r["db_id"] for r in rows} - set(copied))
    print(f"defog -> {out}")
    print(f"  {len(rows)} questions across {len(set(r['db_id'] for r in rows))} databases")
    print(f"  {len(copied)} schemas copied: {', '.join(copied)}")
    if unknown:
        print(f"  WARNING: questions reference databases with no schema: {unknown}")
    print()
    print("  Databases themselves are Postgres. Load them with defog-data's setup.sh,")
    print("  then set POSTGRES_* in .env and run with --executor postgres.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="dataset", required=True)

    wiki = sub.add_parser("wikisql", help="Convert WikiSQL into premsql layout")
    wiki.add_argument("--source", required=True, type=Path, help="Clone of salesforce/WikiSQL")
    wiki.add_argument("--split", default="test", choices=["test", "dev", "train"])

    defog = sub.add_parser("defog", help="Convert Defog into premsql layout")
    defog.add_argument("--defog-data", required=True, type=Path, help="Clone of defog-ai/defog-data")
    defog.add_argument("--sql-eval", required=True, type=Path, help="Clone of defog-ai/sql-eval")
    defog.add_argument(
        "--questions-csv",
        default="questions_gen_postgres.csv",
        help="Which question set to use (default: the Postgres one, which is "
             "where the {col_a,col_b} alternatives appear)",
    )

    args = parser.parse_args()
    if args.dataset == "wikisql":
        prepare_wikisql(args.source, args.split)
    else:
        prepare_defog(args.defog_data, args.sql_eval, args.questions_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
