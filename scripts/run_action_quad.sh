#!/usr/bin/env bash
# Launch the four adaptive benchmark methods in parallel on one 4×A100 node.
# One process per GPU: HIM / RMA / SysID / DreamWaQ, single shared seed, 3000 iters.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"

host="$(hostname -s)"
if [[ ! "$host" =~ ^a[0-9]+$ ]]; then
  echo "[action] REFUSING: host=$host is not an Altay A100 compute node" >&2
  exit 2
fi
if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "[action] REFUSING: no NVIDIA GPU visible on $host" >&2
  exit 2
fi

gpu_count="$(nvidia-smi -L | wc -l | tr -d ' ')"
if [[ "$gpu_count" -lt 4 ]]; then
  echo "[action] REFUSING: expected >=4 GPUs, found $gpu_count on $host" >&2
  nvidia-smi -L >&2 || true
  exit 2
fi

export SIMULATOR=genesis
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python="${ACTION_PYTHON:-$repo/.venv/bin/python}"
if [[ ! -x "$python" ]]; then
  echo "[action] REFUSING: python not executable: $python" >&2
  exit 2
fi

seed="${ACTION_SEED:-1}"
max_iterations="${ACTION_MAX_ITERS:-3000}"
num_envs="${ACTION_NUM_ENVS:-}"
stamp="$(date +%Y%m%d_%H%M%S)"
job_tag="${SLURM_JOB_ID:-local}_${stamp}"
log_root="$repo/logs/action_quad/${job_tag}"
mkdir -p "$log_root"

# Fixed GPU mapping keeps utilization accounting honest.
declare -a TASKS=(go2_bench_him go2_bench_rma go2_bench_sysid go2_bench_dreamwaq)
declare -a LABELS=(him rma sysid dreamwaq)
declare -a GPUS=(0 1 2 3)

commit="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
dirty="$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
echo "[action] host=$host job=${SLURM_JOB_ID:-none} commit=$commit dirty_files=$dirty"
echo "[action] seed=$seed max_iterations=$max_iterations log_root=$log_root"
echo "[action] gpus:"
nvidia-smi -L
echo "[action] nvidia-smi snapshot:"
nvidia-smi --query-gpu=index,name,memory.total,utilization.gpu --format=csv

pids=()
for i in "${!TASKS[@]}"; do
  task="${TASKS[$i]}"
  label="${LABELS[$i]}"
  gpu="${GPUS[$i]}"
  out="$log_root/${label}.out"
  err="$log_root/${label}.err"
  cmd=(
    "$python" -u legged_gym/scripts/train.py
    --task "$task"
    --headless
    --seed "$seed"
    --max_iterations "$max_iterations"
  )
  if [[ -n "$num_envs" ]]; then
    cmd+=(--num_envs "$num_envs")
  fi

  echo "[action] launch label=$label task=$task gpu=$gpu -> $out"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    cd "$repo"
    echo "[action:$label] start $(date -Is) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "[action:$label] cmd: ${cmd[*]}"
    "${cmd[@]}"
    status=$?
    echo "[action:$label] exit=$status at $(date -Is)"
    exit "$status"
  ) >"$out" 2>"$err" &
  pids+=("$!")
  echo "[action] pid=${pids[-1]} label=$label"
done

# Lightweight GPU utilization sampler while the four trainings run.
(
  while true; do
    alive=0
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        alive=1
        break
      fi
    done
    [[ "$alive" -eq 1 ]] || break
    {
      echo "===== $(date -Is) ====="
      nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader
      ps -o pid,pcpu,pmem,etime,cmd -p "$(IFS=,; echo "${pids[*]}")" 2>/dev/null || true
    } >>"$log_root/gpu_watch.log"
    sleep 60
  done
) &
watch_pid=$!

status=0
declare -a exits=()
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  label="${LABELS[$i]}"
  if wait "$pid"; then
    exits+=("$label=0")
    echo "[action] DONE label=$label pid=$pid ok"
  else
    rc=$?
    exits+=("$label=$rc")
    status=1
    echo "[action] FAIL label=$label pid=$pid rc=$rc" >&2
  fi
done

kill "$watch_pid" 2>/dev/null || true
wait "$watch_pid" 2>/dev/null || true

{
  echo "host=$host"
  echo "job=${SLURM_JOB_ID:-none}"
  echo "commit=$commit"
  echo "seed=$seed"
  echo "max_iterations=$max_iterations"
  echo "finished=$(date -Is)"
  echo "exits=${exits[*]}"
  echo "log_root=$log_root"
} | tee "$log_root/summary.txt"

echo "[action] final nvidia-smi:"
nvidia-smi
exit "$status"
