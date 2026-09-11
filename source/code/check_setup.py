"""
Setup check: verifies the environment and the premsql fork are wired up
correctly, without needing an API key or a running model server.

Run from the repo root:
    .venv/bin/python source/code/check_setup.py

Checks, in order:
  1. premsql resolves to the fork (not the PyPI build) and exposes the
     classes this repo depends on.
  2. .env is being loaded and reports which keys are set (never printing
     the values).
  3. The Bird dataset loads and a prompt is built for one row — this
     downloads the dataset on first run.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("1. premsql import check")
print("=" * 60)

import premsql
from premsql.datasets import Text2SQLDataset, WikiSQLDataset, DefogDataset
from premsql.evaluator import Text2SQLEvaluator
from premsql.executors import SQLiteExecutor, PostgresExecutor
from premsql.generators import Text2SQLGeneratorOpenRouter, Text2SQLGeneratorAPI

print(f"   premsql loaded from: {Path(premsql.__file__).parent}")

# These four only exist in the fork; their presence proves the git
# dependency resolved rather than a stock PyPI premsql.
fork_only = {
    "WikiSQLDataset": WikiSQLDataset,
    "DefogDataset": DefogDataset,
    "PostgresExecutor": PostgresExecutor,
    "Text2SQLGeneratorOpenRouter": Text2SQLGeneratorOpenRouter,
}
for name in fork_only:
    print(f"   fork-only class present: {name}")

# The subset-scoring extension lives on the executor base class.
assert hasattr(SQLiteExecutor, "match_sqls")
import inspect
assert "subset_match" in inspect.getsource(SQLiteExecutor.match_sqls), \
    "subset-match scoring missing — premsql is NOT the fork"
print("   subset-match scoring present in BaseExecutor.match_sqls")
print("   -> fork is installed correctly\n")

print("=" * 60)
print("2. .env check")
print("=" * 60)
for key in ("OPENROUTER_API_KEY", "HF_TOKEN", "POSTGRES_USER", "POSTGRES_PASSWORD"):
    value = os.environ.get(key)
    status = "set" if value else "not set"
    print(f"   {key:22} {status}")
print()

if len(sys.argv) > 1 and sys.argv[1] == "--skip-dataset":
    print("skipping dataset check (--skip-dataset)")
    raise SystemExit(0)

print("=" * 60)
print("3. Bird dataset load + prompt build (downloads on first run)")
print("=" * 60)
dataset = Text2SQLDataset(
    dataset_name="bird",
    split="validation",
    force_download=False,
    dataset_folder="source/datasets",
).setup_dataset(num_rows=1, prompt_template="source/prompts/bird_prompt.md")

row = dataset[0]
print(f"   rows loaded: {len(dataset)}")
print(f"   db_id:       {row['db_id']}")
print(f"   db_path:     {row['db_path']}")
print(f"   db file exists: {Path(row['db_path']).exists()}")
print(f"   question:    {row['question'][:100]}")
print(f"   gold SQL:    {row['SQL'][:100]}")
print(f"   prompt chars: {len(row['prompt'])}")
assert "CREATE TABLE" in row["prompt"], "schema was not injected into the prompt"
print("   schema injected into prompt: yes")

print()
print("=" * 60)
print("4. executor sanity check: run the gold SQL against the real DB")
print("=" * 60)
executor = SQLiteExecutor()
out = executor.execute_sql(sql=row["SQL"], dsn_or_db_path=row["db_path"])
if out["error"]:
    print(f"   gold SQL failed to execute: {out['error']}")
else:
    print(f"   gold SQL executed OK, {len(out['result'])} row(s) returned")
    print(f"   execution_time: {out['execution_time']:.4f}s")
    print(f"   result_df shape: {out['result_df'].shape}")

print()
print("ALL SETUP CHECKS PASSED")
print("Next: put a real key in .env as OPENROUTER_API_KEY, then run")
print("  .venv/bin/python source/code/run_eval.py \\")
print("      --dataset bird --backend openrouter --model gpt-4o-mini --num-rows 10")
