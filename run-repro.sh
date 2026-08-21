#!/bin/bash
# Reproduction run: PersonaMem 32k through upstream AMB harness
# Lab notes: This script documents the exact commands for the arXiv paper

set -euo pipefail

# Load credentials
eval "$(grep 'GEMINI_API_KEY' ~/.secrets)"
export GOOGLE_API_KEY="$GEMINI_API_KEY"

eval "$(python3 -c "
import configparser
c = configparser.ConfigParser()
c.read('$HOME/.config/memoryhub/credentials')
print(f'export MEMORYHUB_URL=\"{c.get(\"mcp-rhoai\", \"url\")}\"')
print(f'export MEMORYHUB_API_KEY=\"{c.get(\"mcp-rhoai\", \"api_key\")}\"')
")"

export MEMORYHUB_PROJECT_ID="amb-upstream-repro"
export OMB_ANSWER_LLM=gemini
export OMB_ANSWER_MODEL=gemini-2.5-flash-lite

echo "=== Reproduction Run Configuration ==="
echo "MEMORYHUB_URL: ${MEMORYHUB_URL%/mcp/*}/..."
echo "MEMORYHUB_PROJECT_ID: $MEMORYHUB_PROJECT_ID"
echo "Answer LLM: $OMB_ANSWER_LLM ($OMB_ANSWER_MODEL)"
echo "Judge LLM: gemini (gemini-2.5-flash-lite, harness default)"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "======================================="

uv run amb run \
    --dataset personamem \
    --split 32k \
    --memory memoryhub \
    --name memoryhub \
    --output-dir outputs \
    "$@"
