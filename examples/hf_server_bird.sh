#!/usr/bin/env bash
#
# Evaluate a self-hosted model on BIRD via an OpenAI-compatible server.
#
# Start a server first, e.g.:
#     vllm serve premai-io/prem-1B-SQL --port 7860
#
# Any server exposing POST {API_BASE_URL}/completions works.
#
#   ./examples/hf_server_bird.sh
#   MODEL=defog/sqlcoder-7b-2 API_BASE_URL=http://0.0.0.0:8001/v1 ./examples/hf_server_bird.sh
#
set -euo pipefail

# Run from the repo root so source/prompts and source/datasets resolve.
cd "$(dirname "$0")/.."

MODEL="${MODEL:-premai-io/prem-1B-SQL}"
API_BASE_URL="${API_BASE_URL:-http://0.0.0.0:7860/v1}"
NUM_ROWS="${NUM_ROWS:-10}"

# Fail fast with a clear message rather than burning through the generator's
# retry loop when nothing is listening.
if ! curl -sf --max-time 5 "${API_BASE_URL}/models" >/dev/null 2>&1; then
    echo "error: no server responding at ${API_BASE_URL}" >&2
    echo "       start one first, e.g.: vllm serve ${MODEL} --port 7860" >&2
    echo "       or set API_BASE_URL to point at your server." >&2
    exit 1
fi

exec .venv/bin/python source/code/run_eval.py \
    --dataset bird \
    --backend hf \
    --model "$MODEL" \
    --api-base-url "$API_BASE_URL" \
    --num-rows "$NUM_ROWS" \
    "$@"
