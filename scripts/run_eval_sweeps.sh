#!/usr/bin/env bash
# OOD eval sweeps for the 3 benchmark reference policies (friction + added_mass).
# Sequential (4GB VRAM); full frozen protocol per_point=256, steps=2000.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
export SIMULATOR=genesis
PY=.venv/bin/python
OUT=logs/eval
mkdir -p "$OUT"
declare -A RUN=( [nodr]=Jul09_13-16-36_bench_nodr_genesis [mlp]=Jul09_13-17-02_bench_mlp_genesis [oracle]=Jul09_13-17-28_bench_oracle_genesis )
for AXIS in friction added_mass; do
  for M in nodr mlp oracle; do
    echo "=== $(date +%H:%M:%S) sweep axis=$AXIS method=$M ==="
    $PY legged_gym/scripts/eval/sweep.py \
      --task go2_bench_${M} --load_run "${RUN[$M]}" \
      --axis $AXIS --per_point 256 --steps 2000 --warmup 100 \
      --out "$OUT/${AXIS}_${M}.npz" --label $M 2>&1 \
      | grep -E "grid|falls|^\s+-?[0-9]|saved|Error|error|CUDA|OutOfMemory"
  done
done
echo "=== ALL DONE $(date +%H:%M:%S) ==="
