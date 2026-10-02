#!/usr/bin/env bash
# Send a test chat-completion request to a local vLLM OpenAI server.
#
# Usage: request_vllm.sh <port>

PORT="${1:?Usage: request_vllm.sh <port>}"

curl "http://localhost:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
    "model": "dsv4_ct",
    "messages": [
        {"role": "user", "content": "Explain quantum computing."}
    ],
    "max_tokens": 100
}'
