"""
One parameterized text-to-SQL evaluation runner.

Replaces the ~30 near-identical per-backend scripts that previously lived under
HF_server_models/, LM_Studio/, OpenRouter/ and Runpod_Server/. Those differed
only in which endpoint they pointed at and which dataset they loaded, but each
copy drifted its own way — inconsistent num_rows, force flags, retry counts,
filter keys, and in several cases an experiment_name copied from a different
model, so unrelated runs silently overwrote each other's results.

Examples
--------
    # hosted model through OpenRouter (needs OPENROUTER_API_KEY in .env)
    python source/code/run_eval.py --dataset bird --backend openrouter \
        --model gpt-4o-mini --num-rows 10

    # local LM Studio server
    python source/code/run_eval.py --dataset spider --backend lmstudio \
        --model premAI_quantized --num-rows 50

    # self-hosted HF inference server on a non-default port
    python source/code/run_eval.py --dataset defog --backend hf \
        --model premAI --api-base-url http://0.0.0.0:8001/v1

    # model loaded in-process with transformers, no server required
    python source/code/run_eval.py --dataset bird --backend local \
        --model premai-io/prem-1B-SQL --device cpu --num-rows 10

    # Defog against Postgres instead of SQLite (needs POSTGRES_* in .env)
    python source/code/run_eval.py --dataset defog --backend openrouter \
        --model gpt-4o-mini --executor postgres --db-name academic

Run from the repository root, so the relative paths to source/prompts and
source/datasets resolve.
"""
import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from dotenv import load_dotenv

load_dotenv()

from premsql.datasets import Text2SQLDataset
from premsql.evaluator import Text2SQLEvaluator
from premsql.executors import PostgresExecutor, SQLiteExecutor
from premsql.generators import (
    Text2SQLGeneratorAPI,
    Text2SQLGeneratorHF,
    Text2SQLGeneratorOpenRouter,
    WikiSQLText2SQLGeneratorAPI,
)

PROMPTS = Path("source/prompts")


@dataclass(frozen=True)
class DatasetSpec:
    split: str
    prompt_template: str
    # Column to break the score down by. Must exist on every response row:
    # Text2SQLEvaluator raises KeyError otherwise. Bird ships a "difficulty"
    # label; Spider only has db_id; Defog and WikiSQL have neither worth
    # grouping on, so they report a single overall number.
    filter_by: Optional[str] = None
    # String values may reference {dataset_folder}; it is substituted at run
    # time so these stay correct when --dataset-folder points somewhere other
    # than the default (test fixtures, a scratch copy, an external drive).
    setup_kwargs: dict = field(default_factory=dict)

    def resolved_setup_kwargs(self, dataset_folder: str) -> dict:
        return {
            key: value.format(dataset_folder=dataset_folder)
            if isinstance(value, str)
            else value
            for key, value in self.setup_kwargs.items()
        }


@dataclass(frozen=True)
class BackendSpec:
    # "openrouter" uses the OpenRouter generator (hosted, API key);
    # "api" uses the generic OpenAI-compatible client against a URL;
    # "local" loads the model in-process with transformers.
    kind: str
    default_url: Optional[str] = None
    gen_defaults: dict = field(default_factory=dict)


DATASETS: dict[str, DatasetSpec] = {
    "bird": DatasetSpec(
        split="validation",
        prompt_template="bird_prompt.md",
        filter_by="difficulty",
    ),
    "spider": DatasetSpec(
        split="validation",
        prompt_template="spider_prompt.md",
        filter_by="db_id",
    ),
    "defog": DatasetSpec(
        split="questions_gen",
        prompt_template="defog_prompt.md",
    ),
    "wikisql": DatasetSpec(
        split="test",
        prompt_template="wikisql_prompt.md",
        # WikiSQL ships one combined SQLite file rather than a database per
        # db_id, so every row's db_path is redirected at it.
        setup_kwargs={"custom_db_path": "{dataset_folder}/wikisql/database/test.db"},
    ),
}

BACKENDS: dict[str, BackendSpec] = {
    "openrouter": BackendSpec(
        kind="openrouter",
        gen_defaults={"max_new_tokens": 256},
    ),
    "hf": BackendSpec(
        kind="api",
        default_url="http://0.0.0.0:7860/v1",
        # Self-hosted HF servers accept the extended sampling parameters the
        # stock OpenAI client would drop.
        gen_defaults={
            "max_new_tokens": 4000,
            "use_extended_api": True,
            "stop": [";", "```"],
        },
    ),
    "lmstudio": BackendSpec(
        kind="api",
        default_url="http://localhost:1234/v1",
        gen_defaults={"max_new_tokens": 256},
    ),
    "runpod": BackendSpec(
        kind="api",
        default_url="http://127.0.0.1:8000/v1",
        gen_defaults={"max_new_tokens": 256},
    ),
    # Loads the model in-process with transformers rather than calling a
    # server. --model is a HF repo id or a local checkout path.
    "local": BackendSpec(
        kind="local",
        gen_defaults={"max_new_tokens": 256},
    ),
}


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def build_generator(args, dataset_spec: DatasetSpec, experiment_name: str):
    backend = BACKENDS[args.backend]

    if backend.kind == "openrouter":
        # The system prompt is chosen by dialect: WikiSQL needs the
        # lowercase-literals/col0-style instructions, Postgres runs need
        # Postgres syntax rather than SQLite.
        if args.dataset == "wikisql":
            db_type = "wikisql"
        elif args.executor == "postgres":
            db_type = "postgresql"
        else:
            db_type = "sqlite"
        return Text2SQLGeneratorOpenRouter(
            model_name=args.model,
            experiment_name=experiment_name,
            type=args.experiment_type,
            data_base_type=db_type,
        )

    if backend.kind == "local":
        return Text2SQLGeneratorHF(
            model_or_name_or_path=args.model,
            experiment_name=experiment_name,
            type=args.experiment_type,
            device=args.device,
        )

    # Generic OpenAI-compatible endpoint. WikiSQL gets the subclass that also
    # lowercases quoted string literals, matching how its gold queries are
    # written.
    generator_cls = (
        WikiSQLText2SQLGeneratorAPI if args.dataset == "wikisql" else Text2SQLGeneratorAPI
    )
    return generator_cls(
        model_name=args.model,
        experiment_name=experiment_name,
        type=args.experiment_type,
        api_base_url=args.api_base_url or backend.default_url,
    )


def build_executor(args):
    if args.executor == "postgres":
        # db_name is optional: without it the executor takes the database
        # from each row's db_id, which is what a benchmark spanning several
        # databases (Defog covers 11) needs. Passing it forces every row at
        # one database instead.
        return PostgresExecutor(db_name=args.db_name)
    return SQLiteExecutor()


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run a text-to-SQL benchmark against a model backend.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--backend", required=True, choices=sorted(BACKENDS))
    p.add_argument("--model", required=True, help="Model name as the backend expects it")

    p.add_argument("--api-base-url", help="Override the backend's default endpoint")
    p.add_argument(
        "--device",
        help="Torch device for --backend local (default: cuda:0 if available, else cpu)",
    )
    p.add_argument("--dataset-folder", default="source/datasets")
    p.add_argument("--prompt-template", help="Override the dataset's default prompt file")
    p.add_argument("--num-rows", type=int, help="Limit rows evaluated (default: all)")

    p.add_argument(
        "--experiment-name",
        help="Default: <dataset>__<backend>__<model>, which keeps unrelated runs "
             "from overwriting each other's cached results",
    )
    p.add_argument("--experiment-type", default="test")
    p.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if cached results exist for this experiment name. "
             "Without it, a previous run's predict.json is reused verbatim and "
             "no model call is made.",
    )

    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, help="Override the backend default")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument(
        "--no-execution-guided",
        action="store_true",
        help="Skip execution-guided decoding (no re-prompting on SQL errors)",
    )

    p.add_argument("--executor", default="sqlite", choices=["sqlite", "postgres"])
    p.add_argument(
        "--db-name",
        help="Force every row at one Postgres database. Omit to take the "
             "database from each row's db_id, which is what a benchmark "
             "spanning several databases needs.",
    )

    p.add_argument("--metric", default="accuracy", choices=["accuracy", "ves"])
    p.add_argument("--filter-by", help="Override the dataset's default breakdown column")
    p.add_argument("--no-filter", action="store_true", help="Report only an overall score")
    p.add_argument("--meta-time-out", type=int, default=10)

    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    spec = DATASETS[args.dataset]
    backend = BACKENDS[args.backend]

    experiment_name = args.experiment_name or (
        f"{args.dataset}__{args.backend}__{slugify(args.model)}"
    )
    prompt_template = args.prompt_template or str(PROMPTS / spec.prompt_template)

    filter_by = None if args.no_filter else (args.filter_by or spec.filter_by)

    print("configuration")
    print("-" * 60)
    for key, value in [
        ("dataset", f"{args.dataset} (split={spec.split})"),
        ("backend", args.backend),
        ("model", args.model),
        (
            "endpoint",
            args.api_base_url
            or backend.default_url
            or ("in-process (transformers)" if backend.kind == "local" else "OpenRouter API"),
        ),
        ("prompt", prompt_template),
        ("num_rows", args.num_rows if args.num_rows else "all"),
        ("experiment", experiment_name),
        ("executor", args.executor),
        ("metric", args.metric),
        ("filter_by", filter_by or "(overall only)"),
        ("force", args.force),
    ]:
        print(f"  {key:12} {value}")
    print()

    dataset = Text2SQLDataset(
        dataset_name=args.dataset,
        split=spec.split,
        dataset_folder=args.dataset_folder,
    ).setup_dataset(
        num_rows=args.num_rows,
        prompt_template=prompt_template,
        **spec.resolved_setup_kwargs(args.dataset_folder),
    )

    generator = build_generator(args, spec, experiment_name)
    executor = build_executor(args)

    gen_kwargs: dict[str, Any] = {
        "temperature": args.temperature,
        "force": args.force,
        "postprocess": True,
        "max_retries": args.max_retries,
        **backend.gen_defaults,
    }
    if args.max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = args.max_new_tokens
    if not args.no_execution_guided:
        gen_kwargs["executor"] = executor

    responses = generator.generate_and_save_results(dataset=dataset, **gen_kwargs)

    evaluator = Text2SQLEvaluator(
        executor=executor, experiment_path=generator.experiment_path
    )
    results = evaluator.execute(
        metric_name=args.metric,
        model_responses=responses,
        filter_by=filter_by,
        meta_time_out=args.meta_time_out,
    )

    print()
    print("results")
    print("-" * 60)
    metric_key = f"{args.metric}_percentage"
    for group, scores in sorted(results.items(), key=lambda kv: kv[0] == "overall"):
        total = (
            scores.get("success_count", 0)
            + scores.get("subset_count", 0)
            + scores.get("logic_error_count", 0)
            + scores.get("db_error_count", 0)
        )
        print(
            f"  {group:14} {scores.get(metric_key, 0):6.2f}%  "
            f"exact={scores.get('success_count', 0):<5} "
            f"subset={scores.get('subset_count', 0):<5} "
            f"logic_err={scores.get('logic_error_count', 0):<5} "
            f"db_err={scores.get('db_error_count', 0):<5} "
            f"n={total}"
        )
    print()
    print(f"full results written to: {generator.experiment_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
