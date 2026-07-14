#!/usr/bin/env bash
#SBATCH -J action
#SBATCH -A kds3by
#SBATCH -p a100q
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --time=12:00:00
#SBATCH --chdir=/ari/users/btutak/auv/sim/genesis-wp/LeggedGym-Ex
#SBATCH --output=/ari/users/btutak/auv/sim/genesis-wp/LeggedGym-Ex/logs/slurm/%x-%j.out
#SBATCH --error=/ari/users/btutak/auv/sim/genesis-wp/LeggedGym-Ex/logs/slurm/%x-%j.err

set -euo pipefail
# Required env from submitter:
#   ACTION_TASK   e.g. go2_bench_him
#   ACTION_LABEL  e.g. him
seed="${ACTION_SEED:-1}"
max_iterations="${ACTION_MAX_ITERS:-3000}"
task="${ACTION_TASK:?ACTION_TASK required}"
label="${ACTION_LABEL:?ACTION_LABEL required}"

repo="$(pwd)"
python="$repo/.venv/bin/python"
export SIMULATOR=genesis
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

stamp="$(date +%Y%m%d_%H%M%S)"
log_root="$repo/logs/action_quad/${SLURM_JOB_ID}_${label}_${stamp}"
mkdir -p "$log_root" logs/slurm

{
  echo "[action] $(date -Is) host=$(hostname -s) job=$SLURM_JOB_ID"
  echo "[action] task=$task label=$label seed=$seed max_iterations=$max_iterations"
  echo "[action] commit=$(git rev-parse --short HEAD)"
  nvidia-smi -L
  nvidia-smi --query-gpu=index,name,memory.total,utilization.gpu --format=csv
} | tee "$log_root/header.txt"

cmd=(
  "$python" -u legged_gym/scripts/train.py
  --task "$task"
  --headless
  --seed "$seed"
  --max_iterations "$max_iterations"
)
if [[ -n "${ACTION_NUM_ENVS:-}" ]]; then
  cmd+=(--num_envs "$ACTION_NUM_ENVS")
fi

echo "[action] cmd: ${cmd[*]}" | tee -a "$log_root/header.txt"
set +e
"${cmd[@]}" >"$log_root/${label}.out" 2>"$log_root/${label}.err"
rc=$?
set -e
echo "[action] exit=$rc at $(date -Is)" | tee -a "$log_root/header.txt"
echo "rc=$rc" >"$log_root/summary.txt"
exit "$rc"
