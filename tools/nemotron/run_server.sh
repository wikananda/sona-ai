#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_root=$(CDPATH= cd -- "$script_dir/../.." && pwd)
model_path=${SONA_NEMOTRON_MODEL_PATH:-$project_root/.models/nemotron-3.5/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf}
server_host=${SONA_NEMOTRON_HOST:-127.0.0.1}
server_port=${SONA_NEMOTRON_PORT:-8080}
runtime_bin=${SONA_NEMOTRON_BIN:-nemo-speech}

if ! command -v "$runtime_bin" >/dev/null 2>&1; then
    echo "error: nemo-speech was not found. Install pinned NeMo-Speech.cpp v0.1.0 first." >&2
    exit 1
fi

if [ ! -f "$model_path" ]; then
    echo "error: Nemotron GGUF was not found at: $model_path" >&2
    echo "Download Nemotron 3.5 from Sona's model manager, then retry." >&2
    exit 1
fi

# The NVIDIA runtime reads this name. Sona uses SONA_NEMOTRON_API_KEY so one
# exported secret can protect both sides without placing it on the command line.
if [ -n "${SONA_NEMOTRON_API_KEY:-}" ] && [ -z "${NEMO_SPEECH_HTTP_API_KEY:-}" ]; then
    export NEMO_SPEECH_HTTP_API_KEY=$SONA_NEMOTRON_API_KEY
fi

exec "$runtime_bin" serve \
    --asr-model "$model_path" \
    --host "$server_host" \
    --port "$server_port" \
    --no-ui
