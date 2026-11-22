#!/usr/bin/env bash
# Create .venv and install uniqpick's dependencies (CPU-only torch).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if ! command -v uv >/dev/null 2>&1; then
  echo "install.sh needs uv (https://docs.astral.sh/uv/). Install it, then rerun." >&2
  exit 1
fi

uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt

echo
echo "Done. Try:"
echo "  ./run.sh /path/to/images -k 24"
