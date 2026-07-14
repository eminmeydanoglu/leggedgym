#!/usr/bin/env bash
#SBATCH -J action
#SBATCH -A kds3by
#SBATCH -p a100x4q
#SBATCH --gres=gpu:a100:4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --time=24:00:00
#SBATCH --chdir=/ari/users/btutak/auv/sim/genesis-wp/LeggedGym-Ex
#SBATCH --output=/ari/users/btutak/auv/sim/genesis-wp/LeggedGym-Ex/logs/slurm/%x-%j.out
#SBATCH --error=/ari/users/btutak/auv/sim/genesis-wp/LeggedGym-Ex/logs/slurm/%x-%j.err

set -euo pipefail
mkdir -p logs/slurm logs/action_quad
export ACTION_SEED="${ACTION_SEED:-1}"
export ACTION_MAX_ITERS="${ACTION_MAX_ITERS:-3000}"
# Leave ACTION_NUM_ENVS empty to use each method's config default (4096).
export ACTION_NUM_ENVS="${ACTION_NUM_ENVS:-}"
echo "[sbatch-action] $(date -Is) host=$(hostname -s) job=$SLURM_JOB_ID seed=$ACTION_SEED iters=$ACTION_MAX_ITERS"
echo "[sbatch-action] commit=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
bash scripts/run_action_quad.sh
