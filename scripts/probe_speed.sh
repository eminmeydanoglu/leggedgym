#!/usr/bin/env bash
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"; export SIMULATOR=genesis
PY=.venv/bin/python; OUT=logs/eval; mkdir -p "$OUT"
declare -A RUN=( [nodr]=Jul09_13-16-36_bench_nodr_genesis [mlp]=Jul09_13-17-02_bench_mlp_genesis [oracle]=Jul09_13-17-28_bench_oracle_genesis )
for VX in 1.0 1.5; do
  for M in nodr mlp oracle; do
    echo "=== $(date +%H:%M:%S) added_mass vx=$VX method=$M ==="
    $PY legged_gym/scripts/eval/sweep.py --task go2_bench_${M} --load_run "${RUN[$M]}" \
      --axis added_mass --per_point 128 --steps 2000 --warmup 100 --command_vx $VX \
      --out "$OUT/mass_vx${VX}_${M}.npz" --label "${M}_vx${VX}" 2>&1 \
      | grep -E "grid|^\s+-?[0-9]|saved|Error|CUDA"
  done
done
echo "=== PROBE DONE $(date +%H:%M:%S) ==="
