#!/usr/bin/env bash
# Full Eval V2 suite for HIM + SysID seed1 on the interactive a115 A100.
# Run ONLY on a115 (or another allocated A100 compute node).
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"

host="$(hostname -s)"
if [[ ! "$host" =~ ^a[0-9]+$ ]]; then
  echo "[him-sysid-v2] REFUSING: host=$host is not an Altay A100 compute node" >&2
  exit 2
fi
if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "[him-sysid-v2] REFUSING: no NVIDIA GPU visible" >&2
  exit 2
fi

export SIMULATOR=genesis
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python="${repo}/.venv/bin/python"
config="configs/eval/go2_him_sysid_seed1_v2.yaml"
root="$repo/logs/eval/go2_him_sysid_seed1_v2"
mkdir -p "$root"

ts() { date -Is; }
echo "[him-sysid-v2] $(ts) host=$host commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[him-sysid-v2] config=$config artifact_root=$root"
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,utilization.gpu --format=csv

echo "[him-sysid-v2] $(ts) plan"
"$python" -u -m legged_gym.scripts.eval.campaign plan --config "$config" --suite all \
  | tee -a "$root/automation.log"

echo "[him-sysid-v2] $(ts) select-checkpoints"
"$python" -u -m legged_gym.scripts.eval.campaign select-checkpoints \
  --config "$config" --shard 0/1 \
  2>&1 | tee -a "$root/automation.log"

echo "[him-sysid-v2] $(ts) run suite=all (resume enabled)"
"$python" -u -m legged_gym.scripts.eval.campaign run \
  --config "$config" --suite all --shard 0/1 --resume \
  2>&1 | tee -a "$root/automation.log"

echo "[him-sysid-v2] $(ts) aggregate"
"$python" -u -m legged_gym.scripts.eval.campaign aggregate --config "$config" \
  2>&1 | tee -a "$root/automation.log"

echo "[him-sysid-v2] $(ts) report"
"$python" -u -m legged_gym.scripts.eval.campaign report --config "$config" \
  2>&1 | tee -a "$root/automation.log"

echo "[him-sysid-v2] $(ts) COMPLETE"
nvidia-smi
