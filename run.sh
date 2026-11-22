#!/usr/bin/env bash
# Run uniqpick from its own venv: ./run.sh /path/to/images -k 24
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${UNIQPICK_PYTHON:-$HERE/.venv/bin/python}"

if [ ! -x "$PYTHON" ]; then
  echo "uniqpick: no environment at $PYTHON - run ./install.sh first" >&2
  exit 1
fi

# Reuse the machine's Hugging Face cache when it exists, so the SigLIP2
# weights are not downloaded twice.
if [ -z "${HF_HOME:-}" ] && [ -d /mnt/nvme/lib/huggingface ]; then
  export HF_HOME=/mnt/nvme/lib/huggingface
fi

# Once the default model is in the cache, keep the run off the network:
# reproducible, and no HF warning noise on stderr.
if [ -z "${HF_HUB_OFFLINE:-}" ] && [ -d "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--timm--ViT-B-16-SigLIP2" ]; then
  export HF_HUB_OFFLINE=1
fi

export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
exec "$PYTHON" -m uniqpick "$@"
