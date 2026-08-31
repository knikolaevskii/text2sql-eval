#!/usr/bin/env bash
#
# Evaluate a hosted model on BIRD through OpenRouter.
#
# Needs OPENROUTER_API_KEY in .env. Costs a few API calls — NUM_ROWS is kept
# small so a smoke test is cents rather than dollars.
#
#   ./examples/openrouter_bird.sh                 # defaults below
#   MODEL=claude-3-sonnet NUM_ROWS=50 ./examples/openrouter_bird.sh
#
set -euo pipefail

# Run from the repo root so source/prompts and source/datasets resolve.
cd "$(dirname "$0")/.."

MODEL="${MODEL:-gpt-4o-mini}"
NUM_ROWS="${NUM_ROWS:-3}"

if [ ! -f .env ]; then
    echo "error: .env not found. Copy .env.example to .env and set OPENROUTER_API_KEY." >&2
    exit 1
fi

if ! grep -qE '^OPENROUTER_API_KEY=.+' .env; then
    echo "error: OPENROUTER_API_KEY is empty in .env." >&2
    exit 1
fi

exec .venv/bin/python source/code/run_eval.py \
    --dataset bird \
    --backend openrouter \
    --model "$MODEL" \
    --num-rows "$NUM_ROWS" \
    "$@"
