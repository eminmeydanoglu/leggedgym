# Eval harness (Faz 1)

Method-agnostic OOD evaluation: sweep one domain-randomization axis over a grid,
run every method under the SAME frozen benchmark protocol, and report
distributional metrics. The deliverable of the thesis is *this harness*, not any
single policy.

## Pipeline

1. `sweep.py` builds a benchmark task, loads its checkpoint, tiles the axis grid
   across the parallel envs (one value per contiguous block => `per_point` envs
   act as `per_point` seeds), runs `steps` steps under a fixed forward command,
   and bins per-env metrics back by grid value. Output: one `.npz` per method.
2. `plot_sweep.py` overlays several `.npz` into method x OOD-point degradation
   curves, shading the in-distribution band.

Only the swept axis varies across envs; every other physics axis is pinned to
nominal (`dr_axes.pin_others_to_nominal`) so the curve is purely that axis's
effect. `auto_reset` stays ON, so each env yields many episodes -> real
distributions.

## Metrics (`metrics.py`)

`fall_rate` (headline separator under OOD), `mean_ep_len` (survival), `falls_per_1k`,
`mean_return`, `tracking_lin_err`, `tracking_ang_err`. Each reported as
mean / std / p25 / p50 / p75 across the seeds at every grid point.

## Axes (`dr_axes.py`)

`friction` (implemented), `added_mass` (implemented). Terrain and the OOD-hidden
axes (pd_gain, latency, ...) plug in behind the same `Axis` interface later.
Setters write Genesis simulator internals directly (no public per-env setter);
friction/mass are build-time only and are NOT re-drawn on reset, so writing them
once after build holds for the whole run.

## Run (on the GPU box, env activated)

```bash
source /ari/users/btutak/auv/sim/genesis-wp/activate.sh   # or local .venv

# smoke test: any checkpoint, tiny, one iteration's worth
python legged_gym/scripts/eval/sweep.py --task go2_bench_oracle \
    --per_point 64 --steps 200 --warmup 50 --out logs/eval/smoke.npz

# real curve, 2 baselines on the friction axis
python legged_gym/scripts/eval/sweep.py --task go2_bench_nodr \
    --per_point 256 --steps 1000 --out logs/eval/friction_nodr.npz
python legged_gym/scripts/eval/sweep.py --task go2_bench_mlp \
    --per_point 256 --steps 1000 --out logs/eval/friction_mlp.npz

python legged_gym/scripts/eval/plot_sweep.py \
    logs/eval/friction_nodr.npz logs/eval/friction_mlp.npz \
    --out logs/eval/friction_curve.png
```

`--load_run <dir>` / `--ckpt <n>` pick a specific checkpoint (default: latest).
Repeat a run with a different `--seed` to stack more seeds.

## Known caveats

- **Oracle is a valid ceiling at every sweep grid point.**
  The oracle is trained over the full eval-sweep range (`friction_range = [0.1, 2.5]`,
  `added_mass_range = [-2.0, 5.0]`), matching the axes in `dr_axes.py`.  Because
  `dr_normalize` uses these same widened ranges to compute the privileged labels, the
  [-1, 1] normalisation clamp never saturates at any grid point.  The oracle curve
  is a valid performance ceiling across the entire plotted range -- not just inside
  the narrow training band of the other benchmark methods.
- **Observation noise stays ON** (part of the frozen 45-obs protocol) and is not
  per-step seeded, so repeat runs are not bit-identical. Stack seeds instead of
  expecting exact reproduction.

## Status

Written offline (local box has no Genesis env). NOT yet run against a live sim --
first execution is the UHeM smoke test above. Reviewed by a second pass:
command-curriculum freeze bug fixed. Baseline checkpoints
(`go2_bench_nodr`, `go2_bench_mlp`, `go2_bench_oracle`) still need training.
