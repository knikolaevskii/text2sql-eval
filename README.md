# Text-to-SQL Model Evaluation

An evaluation harness for text-to-SQL models: run a benchmark against a model,
execute the generated SQL against the real database, and score it on whether
the results match.

It compares models across four benchmarks and five serving backends through one
command, so the same evaluation can be pointed at a hosted API, a self-hosted
inference server, or a model loaded locally.

```bash
python source/code/run_eval.py --dataset bird --backend openrouter \
    --model gpt-4o-mini --num-rows 100
```

```
results
------------------------------------------------------------
  challenging     31.03%  exact=9     subset=2     logic_err=15    db_err=3      n=29
  moderate        43.75%  exact=21    subset=4     logic_err=18    db_err=5      n=48
  simple          58.49%  exact=31    subset=3     logic_err=14    db_err=5      n=53
  overall         48.15%  exact=61    subset=9     logic_err=47    db_err=13     n=130
```

*(shape of the output; numbers depend on the model and row count)*

## What this is built on

This harness is built on [premsql](https://github.com/premAI-io/premsql) by
PremAI, which provides the core dataset/generator/executor/evaluator
abstractions. The library is used through a fork that adds the capabilities
this project needed — see [Extensions to premsql](#extensions-to-premsql).

> Note: upstream premsql does not publish a license file, so this repository
> depends on the fork privately rather than redistributing modified copies of
> its source.

## Supported matrix

| Dataset | Split | Auto-downloads | Notes |
|---------|-------|----------------|-------|
| `bird` | validation | yes (~1.4GB) | Scored by `difficulty` (simple/moderate/challenging) |
| `spider` | validation | yes | Scored by `db_id` |
| `defog` | questions_gen | no | Per-`db_id` JSON schemas; can run against Postgres |
| `wikisql` | test | no | One combined SQLite file, JSON column metadata |

| Backend | Endpoint | Use for |
|---------|----------|---------|
| `openrouter` | OpenRouter API | Hosted models (GPT, Claude, Llama, DeepSeek, Qwen) |
| `hf` | `http://0.0.0.0:7860/v1` | Self-hosted inference server (vLLM, TGI) |
| `lmstudio` | `http://localhost:1234/v1` | LM Studio, typically quantized models |
| `runpod` | `http://127.0.0.1:8000/v1` | Model served on a RunPod GPU pod |
| `local` | in-process | Loaded directly with transformers, no server |

`wikisql` and `defog` do not auto-download. Convert them from their upstream
repositories first:

```bash
# WikiSQL: 13,640 questions across 5,230 tables
git clone https://github.com/salesforce/WikiSQL /tmp/WikiSQL
tar -xvjf /tmp/WikiSQL/data.tar.bz2 -C /tmp/WikiSQL
python source/code/prepare_datasets.py wikisql --source /tmp/WikiSQL

# Defog: 210 questions across 11 databases
git clone https://github.com/defog-ai/defog-data /tmp/defog-data
git clone https://github.com/defog-ai/sql-eval /tmp/sql-eval
python source/code/prepare_datasets.py defog \
    --defog-data /tmp/defog-data --sql-eval /tmp/sql-eval
```

Defog is distributed as Postgres dumps and is evaluated against a live server,
so load the databases too:

```bash
cd /tmp/defog-data && DBUSER="$(whoami)" PGDATABASE=postgres ./setup.sh
```

`DBUSER`/`PGDATABASE` are needed on installs where the role is your own
username rather than `postgres` (the Homebrew default) — the script otherwise
connects as a `postgres` role to a database of the same name, and fails if
neither exists. Put matching values in `.env` as `POSTGRES_*`, then:

```bash
python source/code/run_eval.py --dataset defog --backend openrouter \
    --model gpt-4o-mini --executor postgres --filter-by query_category
```

The database for each question is taken from its `db_id`, so one run covers
all 11; `--db-name` forces every row at a single database instead.

## Metrics

**Execution Accuracy** — the generated SQL is executed against the real
database and compared to the gold query's results. A query that looks nothing
like the reference still counts as correct if it returns the same rows, which
is the point: there are many valid ways to express one question.

**Valid Efficiency Score (VES)** (`--metric ves`) — accuracy weighted by how
the generated query's runtime compares to the reference.

**Subset matching** — partial credit when the gold result set is *contained* in
the prediction, e.g. the model selected the right answer plus extra columns.
Scored separately from an outright miss.

Results break down into four buckets, which matter more than the headline
number: `exact`, `subset`, `logic_err` (ran, wrong rows) and `db_err` (failed
to execute). A model failing mostly on `db_err` has a syntax problem; one
failing on `logic_err` misunderstands the schema.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

cp .env.example .env      # add OPENROUTER_API_KEY for hosted models
```

Verify the setup — checks the library resolves, reports which keys are set
(without printing them), downloads the Bird validation split, and runs a gold
query against the real database:

```bash
.venv/bin/python source/code/check_setup.py
```

Then run an evaluation:

```bash
.venv/bin/python source/code/run_eval.py \
    --dataset bird --backend openrouter --model gpt-4o-mini --num-rows 10
```

See [`examples/`](examples/) for worked walkthroughs of the OpenRouter and
self-hosted-server backends, including the full flag reference.

## Extensions to premsql

The fork this depends on adds:

- **Subset-match scoring** — `BaseExecutor.match_sqls` reports whether the gold
  result set is contained in the prediction, alongside exact match, so partial
  answers are distinguishable from wrong ones.
- **Bracket-expansion gold queries** — a gold query written as
  `SELECT {a, b} FROM t` expands to every column-subset permutation, letting one
  reference query accept several valid answers.
- **WikiSQL and Defog datasets** — via a pluggable schema source. Upstream
  always introspects a SQLite file to build the schema prompt; these datasets
  ship JSON column metadata instead (one combined file for WikiSQL, one per
  `db_id` for Defog), so `Text2SQLBaseInstance` now takes a `schema_source`.
- **`PostgresExecutor`** — for Defog benchmarks that run against Postgres,
  reading credentials from the environment.
- **OpenRouter, generic-API, and WikiSQL generators** — for OpenAI-compatible
  endpoints that aren't OpenAI.
- **Error-correction tracking** — execution-guided decoding now reports how
  often a first attempt failed and how often re-prompting with the error
  recovered it.

It also carries fixes needed to make the library usable here: a too-tight
`huggingface-hub` pin that made premsql uninstallable alongside current
transformers, a Bird download that fetched the training databases (100GB+) even
when only the validation split was requested, and a result-comparison bug that
counted any two result sets with matching column *names* as equal regardless of
their values.

## Layout

```
source/code/run_eval.py       the evaluation runner
source/code/check_setup.py    environment/install verification
source/prompts/               per-benchmark prompt templates
examples/                     worked examples per backend
```

## Notes

Results are cached per experiment name under
`~/Library/Caches/premsql/experiments/`. A rerun of the same
dataset/backend/model reuses that cache and makes **no model call** — pass
`--force` to regenerate.

Hosted providers are not deterministic even at `temperature=0`, so scores vary
between runs. Use a few hundred rows before comparing models seriously.
