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
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

# Upstream sources, cloned here when --auto is used. Kept outside
# source/datasets so a re-conversion never re-downloads, and gitignored.
SOURCES_ROOT = Path(".dataset_sources")
REPOS = {
    "wikisql": "https://github.com/salesforce/WikiSQL",
    "defog-data": "https://github.com/defog-ai/defog-data",
    "sql-eval": "https://github.com/defog-ai/sql-eval",
}

OUT_ROOT = Path("source/datasets")

# WikiSQL's own operator tables (lib/query.py). Indices in the structured form
# refer into these.
AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
COND_OPS = ["=", ">", "<", "OP"]

SAFE_QUESTION = re.compile(r"^[A-Za-z0-9 .,!?'\"()\[\]{}:;\-]+$")
NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+")


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, **kwargs)


def ensure_repo(key: str) -> Path:
    """Clone an upstream benchmark repo once; reuse it on later runs."""
    target = SOURCES_ROOT / key
    if (target / ".git").is_dir():
        print(f"  [{key}] already cloned at {target}")
        return target
    SOURCES_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"  [{key}] cloning {REPOS[key]} ...")
    run(["git", "clone", "--depth", "1", REPOS[key], str(target)])
    return target


def ensure_wikisql_data(repo: Path) -> None:
    """
    WikiSQL ships its questions, table definitions and prebuilt SQLite files
    inside data.tar.bz2 rather than checked in, so the archive has to be
    unpacked before anything can read them.
    """
    if (repo / "data" / "test.jsonl").exists():
        print("  [wikisql] data already unpacked")
        return
    archive = repo / "data.tar.bz2"
    if not archive.exists():
        sys.exit(f"Expected {archive} in the WikiSQL clone but it is missing.")
    print("  [wikisql] unpacking data.tar.bz2 ...")
    with tarfile.open(archive, "r:bz2") as tar:
        tar.extractall(repo)


def postgres_role() -> str:
    """
    Pick the role to load Defog's databases with. Homebrew/initdb installs
    create a role named after the OS user rather than "postgres", which is
    what defog-data's setup.sh assumes, so prefer whatever is configured and
    fall back to the OS user.
    """
    return os.environ.get("POSTGRES_USER") or os.environ.get("USER") or "postgres"


def postgres_reachable(role: str) -> tuple[bool, str]:
    if not shutil.which("psql"):
        return False, "psql is not on PATH"
    probe = subprocess.run(
        ["psql", "-h", os.environ.get("POSTGRES_HOST", "localhost"),
         "-p", str(os.environ.get("POSTGRES_PORT", "5432")),
         "-U", role, "-d", "postgres", "-tAc", "SELECT 1"],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        return False, probe.stderr.strip().splitlines()[-1] if probe.stderr.strip() else "connection failed"
    return True, ""


def load_defog_postgres(defog_data: Path) -> None:
    """
    Runs defog-data's own setup.sh, which drops and recreates 11 databases
    from its .sql dumps.

    Two environment values are forced because the script's first psql call
    passes neither -d nor a database name: it connects as ${DBUSER:-postgres}
    to a database of the same name, so on an install whose role is the OS user
    it fails twice over. PGDATABASE points that call at the maintenance
    database; DBUSER supplies the role that actually exists.
    """
    role = postgres_role()
    ok, why = postgres_reachable(role)
    if not ok:
        sys.exit(
            f"Cannot reach PostgreSQL as role '{role}': {why}\n"
            "\n"
            "Defog's databases are Postgres, so a running server is required.\n"
            "Start one (e.g. `brew services start postgresql@16`), or skip this\n"
            "step and run the conversion alone without --load-db."
        )

    print(f"  [defog] loading 11 databases as role '{role}' (drops and recreates them) ...")
    env = {**os.environ, "DBUSER": role, "PGDATABASE": "postgres"}
    run(["bash", "setup.sh"], cwd=defog_data, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print("  [defog] databases loaded")


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="dataset", required=True)

    wiki = sub.add_parser("wikisql", help="Convert WikiSQL into premsql layout")
    wiki.add_argument("--source", type=Path, help="Existing clone of salesforce/WikiSQL")
    wiki.add_argument(
        "--auto", action="store_true",
        help="Clone the benchmark and unpack its data automatically",
    )
    wiki.add_argument("--split", default="test", choices=["test", "dev", "train"])

    defog = sub.add_parser("defog", help="Convert Defog into premsql layout")
    defog.add_argument("--defog-data", type=Path, help="Existing clone of defog-ai/defog-data")
    defog.add_argument("--sql-eval", type=Path, help="Existing clone of defog-ai/sql-eval")
    defog.add_argument(
        "--auto", action="store_true",
        help="Clone both benchmark repos automatically",
    )
    defog.add_argument(
        "--load-db", action="store_true",
        help="Also load the 11 Postgres databases via defog-data's setup.sh. "
             "Requires a running server; DROPS AND RECREATES those databases.",
    )
    defog.add_argument(
        "--questions-csv",
        default="questions_gen_postgres.csv",
        help="Which question set to use (default: the Postgres one, which is "
             "where the {col_a,col_b} alternatives appear)",
    )

    args = parser.parse_args()

    if args.dataset == "wikisql":
        if not args.source and not args.auto:
            sys.exit("Pass --source <clone of WikiSQL>, or --auto to fetch it.")
        source = args.source
        if args.auto and not source:
            print("fetching sources")
            source = ensure_repo("wikisql")
        ensure_wikisql_data(source)
        print()
        prepare_wikisql(source, args.split)
        return 0

    if not (args.defog_data and args.sql_eval) and not args.auto:
        sys.exit("Pass --defog-data and --sql-eval, or --auto to fetch them.")
    defog_data, sql_eval = args.defog_data, args.sql_eval
    if args.auto:
        print("fetching sources")
        defog_data = defog_data or ensure_repo("defog-data")
        sql_eval = sql_eval or ensure_repo("sql-eval")
        print()
    prepare_defog(defog_data, sql_eval, args.questions_csv)
    print()
    if args.load_db:
        load_defog_postgres(defog_data)
        print()
        print("  Set POSTGRES_* in .env, then run with --executor postgres.")
    else:
        print("  The databases themselves are Postgres and were not loaded.")
        print("  Re-run with --load-db, or load them yourself via defog-data/setup.sh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
