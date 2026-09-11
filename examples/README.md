# Examples

Two worked examples of `source/code/run_eval.py`: one against a hosted model
through OpenRouter, one against a self-hosted inference server.

All commands are run **from the repository root**, not from this folder —
the runner resolves `source/prompts/...` and `source/datasets/...` relative to
the working directory.

## Prerequisites

Install and configure first — see [Quickstart](../README.md#quickstart) in the
root README. Then confirm everything resolves:

```bash
.venv/bin/python source/code/check_setup.py
```

---

## 1. OpenRouter (hosted model, no server to run)

The lowest-friction backend: it needs only an API key, so it's the best way to
confirm the pipeline end to end.

**Setup** — put a key in `.env`:

```
OPENROUTER_API_KEY=sk-or-v1-...
```

**Run** ([`openrouter_bird.sh`](openrouter_bird.sh)):

```bash
.venv/bin/python source/code/run_eval.py \
    --dataset bird \
    --backend openrouter \
    --model gpt-4o-mini \
    --num-rows 3
```

**Output:**

```
configuration
------------------------------------------------------------
  dataset      bird (split=validation)
  backend      openrouter
  model        gpt-4o-mini
  endpoint     OpenRouter API
  prompt       source/prompts/bird_prompt.md
  num_rows     3
  experiment   bird__openrouter__gpt-4o-mini
  executor     sqlite
  metric       accuracy
  filter_by    difficulty
  force        False

results
------------------------------------------------------------
  moderate         0.00%  exact=0     subset=0     logic_err=1     db_err=0     n=1
  simple           0.00%  exact=0     subset=0     logic_err=2     db_err=0     n=2
  overall          0.00%  exact=0     subset=0     logic_err=3     db_err=0     n=3
```

Reading the columns:

| column      | meaning |
|-------------|---------|
| `exact`     | executed results matched the gold query exactly |
| `subset`    | gold's result set was contained in the prediction's (e.g. extra columns selected) — partial credit |
| `logic_err` | query ran, but returned the wrong rows |
| `db_err`    | query failed to execute at all (syntax error, timeout, missing column) |

A high `db_err` count means the model is producing invalid SQL; a high
`logic_err` count means it produces valid SQL that answers the wrong question.
That distinction is the main reason to look at the breakdown rather than the
headline percentage.

`--model` accepts either a short name from the generator's mapping
(`gpt-4o-mini`, `claude-3-sonnet`, `llama-3.1-70b`, `deepseek-v3`, ...) or any
full OpenRouter model id such as `anthropic/claude-3.5-sonnet`.

---

## 2. Self-hosted inference server (`--backend hf`)

For a model you serve yourself. The runner talks to it over HTTP, so anything
exposing an **OpenAI-compatible `/v1/completions` endpoint** works —
[hf_model_server](https://github.com/knikolaevskii/hf_model_server), vLLM,
text-generation-inference, or LM Studio.

**Setup** — start a server first. Using
[hf_model_server](https://github.com/knikolaevskii/hf_model_server), a
companion project built for this (CUDA-optimised, dynamic model switching,
defaults to the same port this backend expects):

```bash
python hf_server.py --model premai-io/prem-1B-SQL --port 7860
```

Or with vLLM:

```bash
vllm serve premai-io/prem-1B-SQL --port 7860
```

Any server works as long as `POST {api_base_url}/completions` accepts
`model` / `prompt` / `temperature` / `max_tokens` / `stop`. Confirm it's up:

```bash
curl http://0.0.0.0:7860/v1/models
```

**Run** ([`hf_server_bird.sh`](hf_server_bird.sh)):

```bash
.venv/bin/python source/code/run_eval.py \
    --dataset bird \
    --backend hf \
    --model premai-io/prem-1B-SQL \
    --api-base-url http://0.0.0.0:7860/v1 \
    --num-rows 10
```

`--api-base-url` is optional — `hf` defaults to `http://0.0.0.0:7860/v1`. Pass
it when your server is on a different port.

The `hf` backend differs from `openrouter` in three ways, all defaults you can
override:

- `max_new_tokens` defaults to 4000 rather than 256, since local servers aren't
  billed per token
- requests are posted directly rather than through the OpenAI client, so extra
  sampling parameters the client would silently drop still reach the server
- `stop=[";", "```"]` is set, which keeps models that like to keep talking from
  emitting prose after the query

If the server isn't running you'll get a connection error from the generator's
retry loop rather than a silent zero score.

**Other server backends** are the same command with a different `--backend`,
which only changes the default URL:

| backend    | default endpoint              |
|------------|-------------------------------|
| `hf`       | `http://0.0.0.0:7860/v1`      |
| `lmstudio` | `http://localhost:1234/v1`    |
| `runpod`   | `http://127.0.0.1:8000/v1`    |
| `local`    | in-process, no server needed  |

For **LM Studio**, load a model and start its server (Developer tab → Start
Server), then pass the identifier LM Studio shows — not the Hugging Face repo
id, which it won't recognise:

```bash
.venv/bin/python source/code/run_eval.py \
    --dataset bird --backend lmstudio --model prem-1b-sql --num-rows 10
```

`runpod`'s default assumes an SSH tunnel forwarding the pod's port 8000 to
localhost; pass `--api-base-url` if you reach it another way.

`--backend local` loads the model with transformers in the same process, so it
needs no server at all — useful for a small model on a laptop:

```bash
.venv/bin/python source/code/run_eval.py \
    --dataset bird --backend local \
    --model premai-io/prem-1B-SQL --device cpu --num-rows 5
```

---

## Things worth knowing

**Results are cached by experiment name.** `generate_and_save_results` writes to
`~/Library/Caches/premsql/experiments/<type>/<experiment name>/predict.json`,
and reuses that file verbatim if it exists — **no model call is made**, and
`--num-rows` is ignored. The runner derives the name as
`<dataset>__<backend>__<model>` so unrelated runs can't overwrite each other,
but rerunning the same combination reuses the cache. Pass `--force` to
regenerate:

```bash
.venv/bin/python source/code/run_eval.py --dataset bird --backend openrouter \
    --model gpt-4o-mini --num-rows 3 --force
```

If a run prints `Already results found`, that's the cache — not a fresh
evaluation.

**Scores vary between runs even at `temperature=0`.** Hosted providers don't
guarantee determinism, so the same prompt can produce a different query. At
small `--num-rows` that variance dominates; use a few hundred rows before
comparing models seriously.

**Dataset availability differs.** `bird` and `spider` download automatically on
first use. `wikisql` and `defog` do not — they expect files already present
under `source/datasets/wikisql/` and `source/datasets/defog/`, and raise
`ValueError: ... dataset not found` otherwise.

**Useful flags:**

| flag | effect |
|------|--------|
| `--num-rows N` | evaluate only the first N rows (default: all) |
| `--force` | regenerate instead of reusing cached results |
| `--metric ves` | score Valid Efficiency Score instead of accuracy |
| `--filter-by db_id` / `--no-filter` | change or disable the score breakdown |
| `--no-execution-guided` | disable re-prompting the model with SQL errors |
| `--executor postgres --db-name X` | run Defog against Postgres (needs `POSTGRES_*` in `.env`) |
