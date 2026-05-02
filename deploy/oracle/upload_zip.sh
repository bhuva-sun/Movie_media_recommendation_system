#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <VM_PUBLIC_IP> <SSH_KEY_PATH>"
  exit 1
fi

VM_PUBLIC_IP="$1"
SSH_KEY_PATH="$2"

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

if [[ ! -f "$ROOT_DIR/movie-reco.zip" ]]; then
  echo "movie-reco.zip not found. Run ./deploy/oracle/make_zip.sh first."
  exit 1
fi

scp -i "$SSH_KEY_PATH" "$ROOT_DIR/movie-reco.zip" "ubuntu@${VM_PUBLIC_IP}:~/"

echo "Uploaded movie-reco.zip to ubuntu@${VM_PUBLIC_IP}:~/"

