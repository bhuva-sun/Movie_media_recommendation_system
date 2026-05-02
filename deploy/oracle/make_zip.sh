#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

rm -f movie-reco.zip
zip -r movie-reco.zip . \
  -x "venv/**" \
  -x "**/__pycache__/**" \
  -x ".DS_Store" \
  -x "staticfiles/**" \
  -x "logs/**"

echo "Created: $ROOT_DIR/movie-reco.zip"

