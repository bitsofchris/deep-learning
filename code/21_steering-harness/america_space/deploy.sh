#!/usr/bin/env bash
# Assemble and deploy the America AI Space.
#
# Copies the runtime (source of truth: america_ai/runtime.py) and the exported
# steering bundle into this folder, then uploads everything to the Space.
#
# Usage: ./deploy.sh [space_id]   (default: bitsofchris/america-ai)
set -euo pipefail

SPACE_ID="${1:-bitsofchris/america-ai}"
HERE="$(cd "$(dirname "$0")" && pwd)"
HARNESS="$(dirname "$HERE")"
BUNDLE="$HARNESS/results/google_gemma_2_2b_it/deploy/steering_bundle.pt"

[ -f "$BUNDLE" ] || { echo "Bundle missing — run: python america_export.py --results-name google_gemma_2_2b_it" >&2; exit 1; }

cp "$HARNESS/america_ai/runtime.py" "$HERE/runtime.py"
cp "$BUNDLE" "$HERE/steering_bundle.pt"

hf upload "$SPACE_ID" "$HERE" . --type space \
  --exclude "deploy.sh" \
  --commit-message "Deploy America AI space"
echo "Deployed: https://huggingface.co/spaces/$SPACE_ID"
