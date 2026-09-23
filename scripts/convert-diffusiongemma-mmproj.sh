#!/usr/bin/env bash
# Convert the official DiffusionGemma vision encoder/projector to llama.cpp's
# GEMMA4V mmproj GGUF. This deliberately does not use an Unsloth artifact.
set -euo pipefail

repo="${DIFFUSIONGEMMA_REPO:-google/diffusiongemma-26B-A4B-it}"
# The verified source revision is f7f5b7f5fa82ffc52addd066915886d497f5517b.
# convert_hf_to_gguf.py --remote has no revision flag, so use a local checkout
# at that revision when a strictly reproducible conversion is required.
out="${1:-mmproj-diffusiongemma-26B-A4B-it-f16.gguf}"

if ! command -v python3 >/dev/null; then
    echo "python3 is required" >&2
    exit 1
fi

# Requires requirements/requirements-convert_hf_to_gguf.txt (torch,
# transformers, safetensors). --remote range-reads only the vision shard.
python3 convert_hf_to_gguf.py \
    --remote --mmproj --outtype f16 \
    --outfile "$out" \
    "$repo"

sha256sum "$out"
