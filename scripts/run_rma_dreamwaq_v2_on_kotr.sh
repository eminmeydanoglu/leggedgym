#!/usr/bin/env bash
# Full Eval V2 for RMA + DreamWaQ on the interactive node named "kotr".
# Speed strategy (measured bottleneck is serial cell loop, not VRAM):
#   * Offline select OFF (training best_tracking.pt)
#   * PARALLEL_SHARDS independent campaign processes (default 2) on one A100
#   * Full-node CPU thread pools per process
#   * Shared artifact_root + --resume so shards don't clobber completed cells
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"

host="$(hostname -s)"
if [[ ! "$host" =~ ^a[0-9]+$ ]]; then
  echo "[rma-dw-v2] REFUSING: host=$host is not an Altay A100 compute node" >&2
  exit 2
fi
if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "[rma-dw-v2] REFUSING: no NVIDIA GPU visible" >&2
  exit 2
fi

export SIMULATOR=genesis
export WANDB_MODE=disabled
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Node has 64 cores (a100q). Split across parallel shards.
NPROC="$(nproc)"
SHARDS="${PARALLEL_SHARDS:-2}"
if [[ "$SHARDS" -lt 1 ]]; then SHARDS=1; fi
THREADS_PER=$(( NPROC / SHARDS ))
if [[ "$THREADS_PER" -lt 4 ]]; then THREADS_PER=4; fi
export OMP_NUM_THREADS="$THREADS_PER"
export MKL_NUM_THREADS="$THREADS_PER"
export OPENBLAS_NUM_THREADS="$THREADS_PER"
export NUMEXPR_NUM_THREADS="$THREADS_PER"
export TORCH_NUM_THREADS="$THREADS_PER"
export TORCH_INTRAOP_NUM_THREADS="$THREADS_PER"
export TORCH_INTEROP_NUM_THREADS="$THREADS_PER"

python="${repo}/.venv/bin/python"
config="configs/eval/go2_rma_dreamwaq_seed1_v2.yaml"
root="$repo/logs/eval/go2_rma_dreamwaq_seed1_v2"
mkdir -p "$root"

ts() { date -Is; }
echo "[rma-dw-v2] $(ts) host=$host commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[rma-dw-v2] config=$config shards=$SHARDS threads_per_shard=$THREADS_PER nproc=$NPROC"
nvidia-smi -L
nvidia-smi --query-gpu=index,memory.total,utilization.gpu --format=csv

echo "[rma-dw-v2] $(ts) plan"
"$python" -u -m legged_gym.scripts.eval.campaign plan --config "$config" --suite all \
  | tee -a "$root/automation.log"

if [[ "${SELECT_CHECKPOINTS:-0}" == "1" ]]; then
  echo "[rma-dw-v2] $(ts) select-checkpoints (opt-in)"
  "$python" -u -m legged_gym.scripts.eval.campaign select-checkpoints \
    --config "$config" --shard 0/1 2>&1 | tee -a "$root/automation.log"
else
  echo "[rma-dw-v2] $(ts) skip select-checkpoints; training best_tracking.pt"
fi

# Materialize selection once (first run) so parallel shards share the same file.
"$python" -u - <<'PY' | tee -a "$root/automation.log"
from legged_gym.scripts.eval.campaign import load_config, artifact_root, _materialize_training_best_selection
from pathlib import Path
cfg = load_config("configs/eval/go2_rma_dreamwaq_seed1_v2.yaml")
root = artifact_root(cfg)
root.mkdir(parents=True, exist_ok=True)
if not (root / "checkpoint_selection.json").exists():
    _materialize_training_best_selection(cfg, root)
else:
    print(f"[selection] existing {root / 'checkpoint_selection.json'}")
PY

echo "[rma-dw-v2] $(ts) launching $SHARDS parallel run shards"
pids=()
for ((i=0; i<SHARDS; i++)); do
  shard_log="$root/shard_${i}_of_${SHARDS}.log"
  echo "[rma-dw-v2] start shard $i/$SHARDS -> $shard_log"
  (
    export OMP_NUM_THREADS="$THREADS_PER"
    export MKL_NUM_THREADS="$THREADS_PER"
    export OPENBLAS_NUM_THREADS="$THREADS_PER"
    export TORCH_NUM_THREADS="$THREADS_PER"
    # Spread CPU affinity so shards don't fight on the same cores.
    base=$(( i * THREADS_PER ))
    end=$(( base + THREADS_PER - 1 ))
    if command -v taskset >/dev/null 2>&1; then
      exec taskset -c "${base}-${end}" \
        "$python" -u -m legged_gym.scripts.eval.campaign run \
          --config "$config" --suite all --shard "${i}/${SHARDS}" --resume
    else
      exec "$python" -u -m legged_gym.scripts.eval.campaign run \
        --config "$config" --suite all --shard "${i}/${SHARDS}" --resume
    fi
  ) >"$shard_log" 2>&1 &
  pids+=("$!")
  echo "[rma-dw-v2] shard $i pid=${pids[-1]} cpus=${base}-${end}"
done

# GPU watcher
(
  while true; do
    alive=0
    for pid in "${pids[@]}"; do
      kill -0 "$pid" 2>/dev/null && alive=1 && break
    done
    [[ "$alive" -eq 1 ]] || break
    {
      echo "===== $(date -Is) ====="
      nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
      ps -o pid,pcpu,pmem,etime,cmd -p "$(IFS=,; echo "${pids[*]}")" 2>/dev/null || true
    } >>"$root/gpu_watch.log"
    sleep 30
  done
) &
watch_pid=$!

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[rma-dw-v2] shard $i OK"
  else
    rc=$?
    echo "[rma-dw-v2] shard $i FAIL rc=$rc" >&2
    status=1
  fi
done
kill "$watch_pid" 2>/dev/null || true
wait "$watch_pid" 2>/dev/null || true

if [[ "$status" -ne 0 ]]; then
  echo "[rma-dw-v2] $(ts) run FAILED; skip aggregate" >&2
  exit "$status"
fi

echo "[rma-dw-v2] $(ts) aggregate"
"$python" -u -m legged_gym.scripts.eval.campaign aggregate --config "$config" \
  2>&1 | tee -a "$root/automation.log"
echo "[rma-dw-v2] $(ts) report"
"$python" -u -m legged_gym.scripts.eval.campaign report --config "$config" \
  2>&1 | tee -a "$root/automation.log"
echo "[rma-dw-v2] $(ts) COMPLETE"
nvidia-smi
