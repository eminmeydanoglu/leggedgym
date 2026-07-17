#!/usr/bin/env bash
# Run one V4 seed-1 three-method bundle on an Altay 1xA100 allocation.
# Two methods share the 80 GB A100 first; the third starts as soon as either ends.
set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"

group="${1:-}"
case "$group" in
  a126)
    first_task="go2_v4_rma"
    second_task="go2_v4_sysid"
    third_task="go2_v4_mlp"
    ;;
  a115)
    first_task="go2_v4_superset_oracle"
    second_task="go2_v4_dreamwaq"
    third_task="go2_v4_him_fixed"
    ;;
  *)
    echo "Usage: $0 {a126|a115}" >&2
    exit 2
    ;;
esac

host="$(hostname -s)"
if [[ "$host" != "$group" ]]; then
  echo "[v4] REFUSING: bundle '$group' must run on $group, current host is $host" >&2
  exit 2
fi
if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "[v4] REFUSING: no NVIDIA GPU visible on $host" >&2
  exit 2
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
  echo "[v4] REFUSING: GPU already has compute processes; will not share it accidentally." >&2
  nvidia-smi >&2
  exit 2
fi

python="$repo/.venv/bin/python"
if [[ ! -x "$python" ]]; then
  echo "[v4] REFUSING: Python is not executable: $python" >&2
  exit 2
fi

# Keep the V3-comparable protocol intact: 4,096 environments and 3,000 PPO
# updates per model.  CPU threads are split evenly between the two processes.
export SIMULATOR=genesis
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export WANDB_MODE=disabled
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export OMP_DYNAMIC=FALSE
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

stamp="$(date +%Y%m%d_%H%M%S)"
job_tag="${SLURM_JOB_ID:-manual}_${stamp}"
log_root="$repo/logs/v4_train_seed1/${group}_${job_tag}"
live_link="$repo/logs/v4_train_seed1/${group}_live"
mkdir -p "$log_root"
ln -sfn "$log_root" "$live_link"

{
  echo "[v4] start=$(date -Is) host=$host group=$group job=${SLURM_JOB_ID:-none}"
  echo "[v4] simulator=$SIMULATOR seed=1 num_envs=4096 max_iterations=3000"
  echo "[v4] tasks=$first_task,$second_task -> $third_task"
  echo "[v4] log_root=$log_root"
  nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader
} | tee "$log_root/launcher.log"

declare -a pids=()
declare -a labels=()

launch() {
  local task="$1"
  local out="$log_root/${task}.log"
  echo "[v4] launch task=$task at $(date -Is)" | tee -a "$log_root/launcher.log"
  (
    echo "[v4:$task] start=$(date -Is) simulator=$SIMULATOR visible_gpu=$CUDA_VISIBLE_DEVICES"
    "$python" -u legged_gym/scripts/train.py \
      --task "$task" --seed 1 --num_envs 4096 --max_iterations 3000 --headless
    rc=$?
    echo "[v4:$task] exit=$rc at $(date -Is)"
    exit "$rc"
  ) >"$out" 2>&1 &
  LAST_PID=$!
  pids+=("$LAST_PID")
  labels+=("$task")
  printf '%s\t%s\n' "$LAST_PID" "$task" | tee -a "$log_root/pids.tsv"
}

launch "$first_task"; first_pid="$LAST_PID"
launch "$second_task"; second_pid="$LAST_PID"

# These are direct children of this shell: unlike command substitution, wait -n
# is valid and starts the third task only after an initial task really finishes.
wait -n "$first_pid" "$second_pid"
first_exit=$?
echo "[v4] one initial task ended rc=$first_exit at $(date -Is); launching $third_task" | tee -a "$log_root/launcher.log"
launch "$third_task"; third_pid="$LAST_PID"

status=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  label="${labels[$i]}"
  if wait "$pid" 2>/dev/null; then
    echo "[v4] done task=$label pid=$pid rc=0" | tee -a "$log_root/launcher.log"
  else
    rc=$?
    # The process consumed by wait -n is already reaped; its log still carries
    # the authoritative exit status, so do not turn that bookkeeping case into
    # a false launcher failure.
    if [[ "$rc" -ne 127 ]]; then
      status=1
    fi
    echo "[v4] done task=$label pid=$pid rc=$rc" | tee -a "$log_root/launcher.log"
  fi
done

echo "[v4] finished=$(date -Is) launcher_status=$status" | tee -a "$log_root/launcher.log"
exit "$status"
