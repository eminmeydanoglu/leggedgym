#!/usr/bin/env bash
# Durable A100-side continuation for the Eval V2 campaign.
#
# Start this script ONLY from an allocated compute node.  It can wait for an
# already-running validation selection, then starts final eval, aggregate and
# report without any dependency on the initiating SSH/laptop session.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"

host="$(hostname -s)"
if [[ ! "$host" =~ ^a[0-9]+$ ]]; then
  echo "[automation] REFUSING: host=$host is not an Altay A100 compute node" >&2
  exit 2
fi
if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "[automation] REFUSING: no NVIDIA GPU is visible on $host" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "[automation] REFUSING: repo worktree is dirty" >&2
  git status --short >&2
  exit 2
fi

export SIMULATOR=genesis
export WANDB_MODE=disabled
python="$repo/.venv/bin/python"
config="configs/eval/go2_tripler_v2.yaml"
root="$repo/logs/eval/go2_tripler_v2"
mkdir -p "$root"

timestamp() { date -Is; }
validation_count() {
  find "$root/raw/validation" -type f -name '*.npz' 2>/dev/null | wc -l | tr -d ' '
}
selection_complete() {
  "$python" - "$root/checkpoint_selection.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    raise SystemExit(1)
d = json.loads(p.read_text())
raise SystemExit(0 if d.get("complete") else 1)
PY
}

echo "[automation] $(timestamp) host=$host commit=$(git rev-parse --short HEAD)"

# A selection may already have been started separately.  Its PID is not our
# child, so poll it rather than using shell `wait`.
selection_pid="$root/selection.pid"
if [[ -s "$selection_pid" ]] && kill -0 "$(<"$selection_pid")" 2>/dev/null; then
  pid="$(<"$selection_pid")"
  echo "[automation] $(timestamp) waiting for existing validation pid=$pid"
  while kill -0 "$pid" 2>/dev/null; do
    echo "[automation] $(timestamp) validation $(validation_count)/144 artifacts"
    sleep 120
  done
fi

# If the preceding selection died or was never started, resume from the valid
# validation artifacts already atomically promoted to raw/validation.
if ! selection_complete; then
  echo "[automation] $(timestamp) running/resuming checkpoint selection"
  "$python" -u -m legged_gym.scripts.eval.campaign select-checkpoints \
    --config "$config" --shard 0/1
fi
echo "[automation] $(timestamp) checkpoint selection complete"

echo "[automation] $(timestamp) starting final campaign (resume enabled)"
"$python" -u -m legged_gym.scripts.eval.campaign run \
  --config "$config" --suite all --shard 0/1 --resume

echo "[automation] $(timestamp) aggregating results"
"$python" -u -m legged_gym.scripts.eval.campaign aggregate --config "$config"
echo "[automation] $(timestamp) generating report"
"$python" -u -m legged_gym.scripts.eval.campaign report --config "$config"
echo "[automation] $(timestamp) COMPLETE"
