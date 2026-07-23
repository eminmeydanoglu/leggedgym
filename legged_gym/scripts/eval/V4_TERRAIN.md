# V4 Terrain-Native Eval Harness

Terrain-first evaluation for the V4 policies (V3 physics contract trained on the
ETH game-terrain **curriculum**). The flat V3 s-suites are invalid for V4: on
flat ground the Superset-Oracle's 17×11 height map is all-zero, so `oracle ≈ MLP`
and every gap-closed cell returns `no_oracle_headroom`. This harness instead
makes the **terrain the primary controlled axis** and measures how much of the
blind-MLP → all-seeing-Oracle gap each proprioceptive method (SysID / RMA /
DreamWaQ / HIM) closes on rough ground.

Runner: `legged_gym/scripts/eval/v3_eval.py` (unchanged flat-suite behaviour;
terrain suites are additive). Campaign: `configs/eval/v4_terrain.yaml`.

## Mechanism — controlled `Terrain` grid + pinned level

`campaign.build_terrain_session` stands up the **real training `Terrain` class**,
not the synthetic `_terrain_raw` bumpy field, so geometry and height sampling are
bit-identical to training:

- `mesh_type="heightfield"`, **`curriculum=True`** → the deterministic
  `curiculum()` generator: difficulty grows along rows (`diff = row/num_rows`),
  terrain *type* varies along columns (`choice = col/num_cols`). This is the only
  mode that produces the type-by-difficulty grid — `curriculum=False` calls
  `randomized_terrain()` and randomises **both** type and difficulty per cell,
  which is why the harness does **not** use it.
- `fixed_terrain_level=L` pins every env's spawn row to difficulty `L`.
- The simulator splits envs across `num_cols` types for free:
  `terrain_types = arange(num_envs) // (num_envs / num_cols)`.
- After the env is built (geometry materialised; the initial reset is guarded by
  `if not self.init_done`), the harness flips `env.cfg.terrain.curriculum=False`
  so the reset-time `_update_terrain_curriculum` guard never fires during the
  rollout. Levels stay pinned (std 0) even under `auto_reset`.

With `num_cols=5` on the benchmark proportions `[0.2,0.1,0.25,0.25,0.2]` the five
columns are, in order: **slope, random-uniform, stairs-down, stairs-up,
discrete**. With `num_rows=10`, levels `1/3/5/7/9` give stair heights
`7/11/15/19/23 cm` (`step = 0.05 + 0.2·diff m`) and slope `0.4·diff rad`. The
random-uniform / discrete generators draw from numpy's global RNG; `make_env`'s
`set_seed(eval_seed)` reseeds numpy immediately before the heightfield is built,
so the geometry is byte-identical across all six methods that share
`protocol.eval_seed`. `aggregate` **enforces** this: every baseline/oracle/method
row in a `(type × level/payload)` cell must carry one `terrain_hash` or it raises
(`terrain geometry mismatch`). Geometry is coupled to `eval_seed` by design — a
second eval seed yields a second grid.

## Suites

All three keep the metric contract (`tracking_lin/yaw`, `fall_rate`,
`return_per_step`) and command forward locomotion (`vx=1.0`).

- **t0 — Terrain ID (descriptive).** One nominal difficulty (`id_level`).
  Reports each method's tracking/fall delta vs MLP per type (like s0 `ID_delta`);
  never converted to gap-closed.
- **t1 — Difficulty sweep (the scorecard).** Same fixed rows
  `severity_levels=[1,3,5,7,9]` for every method; the env→type split gives the
  `type × difficulty` breakdown for free. Gap-closed: baseline=MLP,
  oracle=Superset-Oracle, method=SysID/RMA/DreamWaQ/HIM. Scenario granularity is
  `(type × level)`; scope `GapClosed_terrain_{type}`.
- **t2 — Terrain × payload.** Fixed mid difficulty (`payload_level`) × small
  payload set, reusing the flat V3 payload axis under terrain. Scope
  `GapClosed_terrain_payload_{tier}`.

**OOD philosophy.** The useful band is where MLP already struggles but the oracle
still stands. `score_cell`'s existing `oracle_unstable` / `oracle_speed_saturated`
gates handle this automatically: the difficulty the oracle topples on drops out
as a *physical ceiling*. Extend `severity_levels` only while the oracle survives.

## Workflow

```bash
# 1. Discover finalised runs (append-only; re-run as seeds/methods finish).
python -m legged_gym.scripts.eval.v3_eval resolve-runs --config configs/eval/v4_terrain.yaml
# 2. Plan / run / aggregate / report.
python -m legged_gym.scripts.eval.v3_eval plan      --config configs/eval/v4_terrain.yaml --suite all
python -m legged_gym.scripts.eval.v3_eval run       --config configs/eval/v4_terrain.yaml --suite all --resume
python -m legged_gym.scripts.eval.v3_eval aggregate --config configs/eval/v4_terrain.yaml
python -m legged_gym.scripts.eval.v3_eval report    --config configs/eval/v4_terrain.yaml
```

`--suite` also accepts `t0` / `t1` / `t2` individually. `--shard i/n` partitions
cells across workers. `--resume` skips completed, finite artifacts.

## A100 GPU smoke (UHeM `ssh makine`, interactive node)

Run before the full campaign. Small `num_envs`, one checkpoint, `--suite t1` on a
single severity level. Prove:

1. **Height-map readback** — for the oracle/RMA, `simulator.measured_heights` is
   non-zero and scales with `fixed_level`; obs dim 187 assembled correctly.
2. **Terrain determinism** — same cell twice → identical `terrain_hash`, tracking
   diff < 1e-3.
3. **Cross-method geometry identity** — build the *same* level on all six tasks
   and assert one shared `terrain_hash`. This is the other side of the
   `aggregate` fail-loud guard: it proves no task consumes numpy RNG differently
   before the terrain is built. (The local smoke only re-ran one task twice.)
4. **Even type split** — `bincount(terrain_types)` equal across the 5 types; no
   NaN/Inf in any type.
5. **Level pinning** — `terrain_levels.std() == 0` at `fixed_terrain_level`.
6. **Atomic promotion + `--resume`** work.

A local RTX 3050 smoke already passed 2/4/5 + a short rollout on the MLP
(`terrain_hash` deterministic, `levels.std()==0`, `bincount=[2,2,2,2,2]`,
`measured_heights (N,187)` non-zero/scaled/finite). On the A100 extend it to the
oracle for check 1 and to all six tasks for check 3.

## Full campaign

Only after the smoke is green. Open `num_envs` to the full `380` (76/type) and
run all six methods across the configured seeds. Run as a **separate job** — do
not launch it from the harness-building task.

## Caveats

- **Data availability.** Only `mlp / superset_oracle / dreamwaq / him_fixed` are
  trained under `logs/go2_v4_terrain_curriculum`. `sysid / rma` currently exist
  only under the superseded fixed-difficulty `go2_v4_medium_terrain` contract —
  a *different* training regime that must not be mixed into a curriculum
  comparison. `resolve-runs` lists them as pending; the scorecard scores them
  once they are retrained under the curriculum log root.
- **Single seed.** V4 has only seed 1 so far. First outputs are descriptive
  (no "proven" language). The config is append-only: `resolve-runs` widens the
  manifest automatically as seeds 2–3 land.
- **Checkpoint-selection confound.** `best_tracking` is selected on each method's
  *live curriculum* env, so different methods are selected at different
  difficulties (the oracle climbs higher). The t1 sweep runs on a shared fixed
  grid so the comparison is still fair, but a shared fixed-difficulty validation
  bank for selection parity remains a possible future job.
- **Reached-level bar (deferred).** The plan's descriptive "which method climbs
  higher" bar comes from each run's training TensorBoard `Episode/terrain_level`,
  not from this eval (which pins levels by design). It belongs to reporting and
  is intentionally not built into the harness here.
