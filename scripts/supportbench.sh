#!/usr/bin/env bash
# supportbench: run the three arms (memory, knowledge, linked) N times against a Hindsight server.
#
#   scripts/supportbench.sh                       # local server (http://localhost:8888), 3 runs per arm
#   TARGET=dev scripts/supportbench.sh            # Hindsight dev (key in ~/.hindsight/dev-api-key)
#   ARMS="linked" RUNS=1 scripts/supportbench.sh  # one arm, once
#
# The judge (and AMB's unused answer model) is gemma-4-26b-a4b on Hindsight Inference dev, keyed by
# ~/.hindsight/dev-inference-key. JUDGE_BASE_URL / JUDGE_API_KEY / OMB_JUDGE_* judge elsewhere.
# EXTRA="--skip-ingestion" answers again over the banks a previous run of the same name recorded.
set -euo pipefail
cd "$(dirname "$0")/.."

case "${TARGET:-local}" in
  local) export HINDSIGHT_HTTP_URL="${HINDSIGHT_HTTP_URL:-http://localhost:8888}" ;;
  dev)
    export HINDSIGHT_HTTP_URL=https://api.dev.hindsight.vectorize.io
    export HINDSIGHT_HTTP_KEY="${HINDSIGHT_HTTP_KEY:-$(cat ~/.hindsight/dev-api-key)}"
    ;;
  *) echo "TARGET must be local or dev" >&2; exit 2 ;;
esac

export OMB_JUDGE_LLM="${OMB_JUDGE_LLM:-openai}" OMB_JUDGE_MODEL="${OMB_JUDGE_MODEL:-gemma-4-26b-a4b}"
export OMB_ANSWER_LLM="${OMB_ANSWER_LLM:-openai}" OMB_ANSWER_MODEL="${OMB_ANSWER_MODEL:-gemma-4-26b-a4b}"
# A shell's own OPENAI_API_KEY is for OpenAI, not this endpoint: the judge's key is its own setting.
export OPENAI_BASE_URL="${JUDGE_BASE_URL:-https://inference.dev.hindsight.vectorize.io/v1}"
export OPENAI_API_KEY="${JUDGE_API_KEY:-$(cat ~/.hindsight/dev-inference-key)}"

mkdir -p logs
for arm in ${ARMS:-memory knowledge linked}; do
  for i in $(seq 1 "${RUNS:-3}"); do
    name="support-$arm-${TARGET:-local}-$i"
    echo "== $name"
    uv run amb run --dataset supportbench --split nextcloud --mode agent \
      --memory "hindsight-support-$arm" --name "$name" ${EXTRA:-} > "logs/$name.log" 2>&1 \
      || { echo "!! $name failed, see logs/$name.log"; continue; }
    grep -E "Accuracy|accuracy" "logs/$name.log" | tail -2 || true
  done
done
