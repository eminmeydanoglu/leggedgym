# Faz 0 freeze — V5 / UED frozen numbers

Single collection point for every number/rule that `lpacr/solun_plani.md` §11
("Faz 0 — Spesifikasyonu kilitle") requires dondurulmuş (frozen) before Faz B
training starts, per `lpacr/subplans/05_arms_config_and_fazB.md`. Every value
below is quoted from the merged code (Topics 01-04) and `go2_v5_config.py`
(Topic 05); nothing here is invented. File:line references point at this
working tree.

Regenerate the fingerprints below with:

```
PYTHONPATH=. SIMULATOR=genesis ./.venv/bin/python -c "
from legged_gym.envs.go2.go2_v5_config import Go2V5LPACRLCfg, build_ued_teacher
curriculum, task_space = build_ued_teacher(Go2V5LPACRLCfg())
print(task_space.fingerprint())
print(curriculum.config_fingerprint)
"
```

## 1. Task bins + task-space fingerprint (solun_plani.md §4)

- **6 terrain types** (`legged_gym/utils/ued/task_space.py:41-43`,
  `TaskSpace.TERRAIN_TYPE_NAMES`): `stairs_up, stairs_down, slope_up,
  slope_down, rough, flat` — same order used by
  `configs/eval/v5_ued.yaml:11-14` (`terrain_types` + `flat_terrain_type`).
- **4 terrain levels** per non-flat type (`legged_gym/utils/terrain.py:42`,
  `TAXONOMY_NUM_LEVELS = 4`); flat degenerates to a single level
  (`task_space.py:88-91`: `_terrain_configs` is `5 types × 4 levels + (flat,
  level 0)` = 21 terrain configs).
- **4 `v_x` bins**, width 0.5, ceiling 2.0 m/s
  (`task_space.py:44`, `VELOCITY_BIN_EDGES = (0.0, 0.5, 1.0, 1.5, 2.0)`).
- **Task-space size**: `21 terrain configs × 4 v_x bins = 84`
  (`task_space.py:94-96`, `TaskSpace.size`); asserted in
  `tests/test_v5_training_contract.py:114` and
  `tests/test_ued_checkpoint_selection.py:90`.
- **Terrain geometry parameters** frozen in
  `legged_gym/utils/terrain.py:143-158`
  (`ued_training_builder_parameters`): `step_width=0.4m`; `step_heights=(0.05,
  0.10, 0.15, 0.20)`; `slope_gradients=(0.0, 0.13, 0.27, 0.40)`;
  `rough_amplitudes=(0.02, 0.045, 0.07, 0.10)`; `horizontal_scale=0.1`,
  `vertical_scale=0.005`, `terrain_length=terrain_width=8.0` (inherited from
  `Go2BenchmarkV4TerrainCfg.terrain`); deterministic geometry seed
  `ued_training_seed = V5_TERRAIN_SEED = 0`
  (`legged_gym/envs/go2/go2_v5_config.py:55-61`, shared by all three
  UED-teacher arms so their static 4×6 grid is byte-identical).
- **TaskSpace fingerprint** for the actual V5 training grid (all three
  UED-teacher arms share this value; verified equal in
  `tests/test_v5_training_contract.py:180-196`
  `test_geometry_command_bank_and_eval_seed_shared_across_ued_arms`):

  ```
  cb948539ac69de87003ffa9c9d6bffa42ae080d5418dd6459d942cf2e843a51a
  ```

  (regenerated from `build_ued_teacher(Go2V5LPACRLCfg())[1].fingerprint()` via
  the command above, on this working tree, 2026-07-23). The fingerprint
  payload is `{terrain_type_names, velocity_bin_edges, builder_parameters}`
  (`task_space.py:75-87`); a bare `TaskSpace()` with no `builder_parameters`
  (used only by CPU-only unit tests) hashes to a different, harmless value
  (`ce331c8b10c2f8b642b6c4795492adb3913e90293ff040cbebbb963ae9a8ef7c`) since it
  carries no geometry description at all.
- **Curriculum config fingerprint** (`{algorithm, stage_length_control_steps,
  beta}`, `legged_gym/utils/ued/episode_curriculum.py:103-106`) for
  `lp_acrl` at the frozen stage length/beta below:
  `c1a0682c13675e2b1d64a3159ef58622797ae8c4221848b7a4c7c6e48de7724a`.

## 2. Standstill rho (solun_plani.md §4 "Standstill kontratı", §14.4)

- `V5_STANDSTILL_RHO = 0.12` (`legged_gym/envs/go2/go2_v5_config.py`) —
  shared `commands.zero_cmd_prob` on `Go2V5CommonCfg` for all four arms so
  standing exposure never becomes an experiment variable (lower than the
  legacy base default of 0.4 to keep LP-valid cell density higher).
- For the three UED-teacher arms standstill is a **reserved mixture bucket**,
  drawn per-env at assignment time: each env is standstill with probability
  `rho` and an LP moving task otherwise (`legged_robot._assign_ued_batch` /
  `_ued_standstill_mask`). rho is an explicit budget line, not a hidden
  contamination rate. `handcrafted_v4` keeps the legacy batch-wide draw
  (`per_env_standstill` absent/`False`, asserted in
  `test_v5_training_contract.py:123`); v3/v4 are bit-for-bit unaffected.
- A standstill episode is **born labelled** (`GenesisUEDAdapter.assign_standstill`
  sets `episode_standstill=True`): it stands on an LP-weighted placement terrain
  (`EpisodeCurriculum.draw_placements`, trusting LP for the easy→hard ordering of
  standing) but its return is never attributed to that cell. There is no post-hoc
  invalidation flag — the env routes standstill outcomes to the adapter's own
  standstill bucket (`record_standstill_outcomes` / `standstill_diagnostics`) and
  only moving episodes ever reach the curriculum. Standstill still feeds
  PPO/reward data (§4, §14.4).

## 3. Stage control-step length (solun_plani.md §5, §11 Faz 0)

- `V5_STAGE_LENGTH_CONTROL_STEPS = 2000` (`go2_v5_config.py:40-47`), passed as
  `stage_length_control_steps` into every UED-teacher arm's curriculum
  (`go2_v5_config.py:109`, `262`). `EpisodeCurriculum.advance()` gates a stage
  boundary on `global_control_steps - stage_start_global_steps >=
  stage_length_control_steps`
  (`legged_gym/utils/ued/episode_curriculum.py:180-185`), not PPO iteration or
  episode count, matching §5's "Stage sınırı ... sabit `global_control_steps`"
  rule.
- At `num_steps_per_env=24` (shared PPO HP, `test_v5_training_contract.py:168`)
  this is ~83 PPO iterations/stage (~36 stages across the 3000-iteration
  budget) — see the worked-through comment at `go2_v5_config.py:42-46`.

## 4. Missing-task / minimum-count rule (solun_plani.md §5)

- A task cell counts as "observed" in a stage iff it received **at least one**
  completed moving-task episode that stage (standstill never reaches the
  curriculum): `observed = self._stage_episode_counts > 0`
  (`legged_gym/utils/ued/episode_curriculum.py:186`).
- Cells that were *not* observed this stage do **not** get a fabricated LP
  score. Their existing probability mass is retained verbatim
  (`episode_curriculum.py:192-199`: `progress_mask = observed &
  self._observed_masks`; only `progress_mask` cells get a new softmax score,
  scaled so the retained mass plus the redistributed mass still sums to 1).
  This is the concrete mechanism behind §5's "Bir stage'de yeterli gözlem
  almayan hücre için sahte reward yazılmaz."
- The very first stage builds only `R_0` and produces no LP (`_observed_masks`
  starts all-`False`, so `progress_mask` is empty until the second valid
  measurement — §5 "İlk stage yalnız `R_0`yı kurar").

## 5. Beta procedure + chosen value (solun_plani.md §5, §11 Faz B)

- `V5_BETA = 1.0` (`go2_v5_config.py:49-53`) is the **frozen starting value**
  for the Faz-0 pilot, explicitly documented as pilot-revisable but only via a
  re-freeze in this file and in `go2_v5_config.py`, never tuned in-flight.
- Softmax temperature use: `_softmax(values, beta)` divides the max-shifted
  score by `beta` before `exp`/normalize, in `float64`, and raises on
  non-finite input or non-finite/non-positive normalizer
  (`episode_curriculum.py:166-178`), matching §5's "Softmax `float64` ve
  max-shift/logsumexp ile hesaplanır; NaN/Inf fail-fast hatadır."
- Procedure: `scripts/run_v5_fazB.sh --beta-pilot` runs a single-seed
  `go2_v5_lpacrl` short training (default 400 iterations, full 4096 envs) with
  no post-hoc validation, so a human can inspect its sampler-health
  diagnostics (`EpisodeCurriculum.diagnostics()`:
  `finite_probabilities, entropy, effective_sample_size,
  max_cell_probability, task_assignment_coverage,
  valid_completed_outcome_coverage` — `episode_curriculum.py:226-235`) before
  headline seeds start. The script cannot auto-select beta; it only produces
  the pilot run to review (§11 Faz B: "beta development pilotu ...
  headline seed'lerden önce koşulur ve sabitlenir").

## 6. SPNTE v1 formula + v_scale + first-fall semantics (solun_plani.md §10, §14.1, §14.3)

- Implementation: `legged_gym/scripts/eval/metrics.py` `MetricAccumulator`.
  Per-step: `spnte_err_sum += clip(|cmd_x - v_x| / v_scale, 0, 1)` only while
  `still_first` (no fall recorded yet for that env) — `metrics.py:131-144`.
  `first_fall_step` freezes at the physics step of the first `done &
  ~time_out` — `metrics.py:145-149`.
- `compute()`: `spnte_lin = (spnte_err_sum + (steps - first_fall_step)) /
  steps` — `metrics.py:189,210` — i.e. every step at/after the first fall is
  charged the maximum error `1.0` (the "tail penalty"), and a policy that
  never falls scores plain normalized tracking error. This is the
  auto-reset-safe realisation of §14.3: `MetricAccumulator` keeps `auto_reset`
  ON (old metrics still consume the full horizon) while SPNTE freezes its own
  stream via `first_fall_step`.
- **Command is 3-axis** (`vx, vy, omega_z`): `heading_command=False` on
  `Go2V5CommonCfg.commands`, so the yaw axis commands angular VELOCITY omega_z
  directly (gyro-observable, sim-to-real robust) instead of a world-frame
  heading pose. The vx bin is the UED task axis; vy / omega_z ride along as
  within-cell nuisance over the shared `[-1, 1]` support
  (`commands.ranges.lin_vel_y` / `ang_vel_yaw`), identical across all four arms
  and mirrored by the offline validation bank.
- **v_scale is never hardcoded**: `v_scale = max(|lin_vel_x_min|,
  |lin_vel_x_max|)`, derived from the active eval config's command bank
  (§14.1). For V5 (`commands.ranges.lin_vel_x = [0.0, 2.0]`,
  `go2_v5_config.py:98`) this is `2.0`
  (asserted in `tests/test_v5_training_contract.py:140-146`
  `test_shared_forward_command_support_and_v_scale`); the frozen validation
  bank derives the same value via
  `legged_gym/scripts/eval/ued_validation.py:139-141`
  (`support_scale`) from `velocity_bin_edges` in `configs/eval/v5_ued.yaml:15`,
  and stores it as `spnte_v_scale` on every artifact (`ued_validation.py:152`,
  `261`). V4's `[-1,1]` support gives `v_scale=1.0`.
- `K = 1000` fixed evaluation episode length is the frozen
  `rollout.steps` in `configs/eval/v5_ued.yaml:32`
  (`warmup_steps=100` unmeasured settling before the window starts, line 31).

## 7. Fixed `84×12` validation matrix, seeds, success threshold (solun_plani.md §10)

**Deliberate re-freeze from §10's original prose (approved 2026-07-23)**:
`solun_plani.md` §10 originally froze "48 replika/hücre, 84×48=4032". After
review, the validation bank was **deliberately re-frozen to 12 replicas/cell
(1,008 total)** plus a sparse checkpoint-scoring *schedule* (§8 below), to cut
the offline-validation compute bill ~4×. `FROZEN_REPLICAS_PER_CELL = 12` is the
frozen, test-enforced value (`ued_validation.py:25`,
`tests/test_ued_checkpoint_selection.py`). This freeze document is the single
source of truth for the validation matrix size; the earlier `48` value from
`solun_plani.md` §10/§11 (now archived under `archive/lpacr/`) is superseded.

All frozen in `configs/eval/v5_ued.yaml` and enforced by
`legged_gym/scripts/eval/ued_validation.py::validate_config`:

- `validation_seed = 31001` (checkpoint-selection bank), `eval_seed = 41001`
  (held-out final bank) — `v5_ued.yaml:10-11`; enforced at
  `ued_validation.py:100-101`.
- 84 cells = `(5 moving terrain types × 4 levels) + 1 flat = 21` terrain
  configs `× 4 v_x bins`; **12 deterministic replicas/cell**
  (`replicas_per_cell: 12`, `v5_ued.yaml:21`; `FROZEN_REPLICAS_PER_CELL = 12`,
  `ued_validation.py:25`) → `84 × 12 = 1008` total env-replicas
  (`FROZEN_NUM_CELLS * n_replicas` check, `ued_validation.py:143-145`).
- Per-replica **3-axis** command draws are generated once, deterministically, by
  `np.random.Generator(PCG64(validation_seed))`
  (`build_validation_bank`) and reused verbatim by every checkpoint/method:
  each replica freezes `command_vx` (uniform in its velocity bin) plus
  `command_vy` and `command_yaw` (omega_z, uniform over the shared `[-1, 1]`
  nuisance support pinned in `validation_bank.command_support`).
  `validate_measurements` rejects any reported vx/vy/yaw that diverges from the
  frozen draw. This mirrors the online 3-axis training command
  (`heading_command=False`, so omega_z is commanded directly, not derived from a
  heading pose) so training and offline eval share ONE command distribution.
- Two disjoint command streams over the SAME terrain grid (§7 / reviewer item
  7): the **validation** bank (`validation_seed=31001`) selects the checkpoint;
  the **held-out** bank (`eval_seed=41001`) draws INDEPENDENT commands and
  measures the winner ONCE via `ued_rollout.py --bank holdout`, writing to
  `run_dir/heldout_final/` — a directory `select_checkpoint` never scans, so the
  held-out result can never leak back into selection.
- Frozen bank fingerprints (config + full replica matrix hash, `sha256` over
  `{schema_version, bank_kind, validation_seed, eval_seed, geometry_hash_version,
  geometry_hashes, rows}`, `schema_version=3`), verified by recomputing them
  live (`bank_fingerprint(load_config("configs/eval/v5_ued.yaml"), kind=...)`):

  ```
  validation dd1a2cd006a3774fd5f58ffe573e40a3fb63c2d3d917a5427bfa064abab09bcc
  holdout    a00aa2ed52fd7a975c74004c75f83905f4dc5bfb7749a4b47ffe52c5faebebd5
  ```

  pinned in `configs/eval/v5_ued.yaml` (`bank_fingerprint` /
  `holdout_bank_fingerprint`) and asserted in
  `tests/test_ued_checkpoint_selection.py`.
- Geometry hashes: one pinned 64-hex-char SHA-256 per **(terrain type, level)**
  tile, computed over the ACTUAL int16 heightfield bytes of the built tile plus
  its scale metadata (`geometry_hash_version=v5_ued_geometry_v2_hfbytes`,
  regenerable headless via `ued_validation.py --emit-geometry-pins` /
  `terrain.build_taxonomy_geometry_hashes`). The rollout RECOMPUTES this hash
  from the live Genesis scene per replica (`ued_rollout._runtime_geometry_hash`),
  so `validate_measurements` compares an independent runtime observation against
  the pin and fails closed on the wrong / corrupted geometry — it is no longer a
  self-referential passthrough. Level-0 slopes and flat share a hash because a
  0.0-gradient tile IS flat (expected, not a bug).
- **Success threshold** (`v5_ued.yaml:40-43`): `minimum_survival_steps=900`
  (of 1000), `spnte_lin_lt=0.30`, `replica_success_rate_threshold=0.90` (a
  cell counts "successful" if ≥90% of its 12 replicas individually satisfy
  both survival and SPNTE bounds — `ued_validation.py:254-262`). Frozen and
  never revised after results are seen (§10).

## 8. `best_spnte.pt` selection + tie-break (solun_plani.md §10, §12)

Implementation: `legged_gym/scripts/eval/select_checkpoint.py`.

Offline only: training does **not** auto-run bank validation
(`configs/eval/v5_ued.yaml:5-7`). After a run, score the scheduled iterations
with `ued_rollout.py`, then call `select_checkpoint.py` (see
`scripts/run_v5_fazB.sh`'s `validate_one`, which computes the schedule and
only invokes `ued_rollout.py` for those iterations).

**Checkpoint-scoring schedule** (`select_checkpoint.py:66-116`
`scheduled_iterations`/`selection_schedule_from_config`, config fields at
`v5_ued.yaml:50-56`): over existing `model_*.pt` (`save_interval=200` may
stay unchanged in `go2_v5_config.py`):

- start target `min_iteration=1000`;
- for target T, take the largest existing checkpoint `<= T`
  (`>= min_iteration`, strictly after the previous pick); if nothing floors
  to T, advance T by the stride instead of picking early
  (`select_checkpoint.py:90-99`);
- next target = chosen iteration + `iteration_stride=500`;
- always include the final periodic checkpoint
  (`always_include_final=True`, `v5_ued.yaml:56`).

Example with saves every 200 through 3000: `1000, 1400, 1800, 2200, 2600,
3000` (worked through in `select_checkpoint.py:74-76`'s docstring example).
`load_candidates` (`select_checkpoint.py:166-183`) refuses selection unless
every SCHEDULED iteration has a checkpoint AND a complete validation
artifact — a non-scheduled checkpoint (e.g. `model_1200.pt` when the schedule
picked `1400`) never needs scoring at all.

- Primary rule: minimum 84-cell **equal-weighted** macro-mean SPNTE
  (`ued_validation.py:234-274` `aggregate_measurements`;
  `select_checkpoint.py:186-204` `is_better`).
- Ties only inside absolute tolerance `primary_tolerance = 1e-6`
  (`v5_ued.yaml:48`); tie-break order:
  1. lower worst-10% task CVaR SPNTE (`worst_task_fraction=0.10`,
     `v5_ued.yaml:49`);
  2. lower macro fall rate;
  3. higher macro success rate;
  4. earlier iteration.
- Only scheduled, existing `model_<iteration>.pt` files with a matching,
  complete, hash-verified validation artifact participate
  (`select_checkpoint.py:144-183`); an incomplete bank or a
  checkpoint/geometry hash mismatch raises, it never silently skips
  (`ued_validation.py:193-231`, `select_checkpoint.py:148-158`).
- Selection never overwrites `best_tracking.pt`
  (`select_checkpoint.py:253-264` `materialize_best_spnte` writes a
  *separate* `best_spnte.pt`, refusing if the source path already equals the
  target); resume preserves both the prior `best_spnte` metadata block and
  any `resume_metadata.best_spnte`
  (`select_checkpoint.py:235-250` `_merge_best_spnte_metadata`).
- `model_3000.pt` is retained as an end-of-training provenance artifact and
  is also on the schedule as the always-included final checkpoint, but it is
  never special-cased as the automatic winner (§10).

## 9. Curriculum checkpoint schema (solun_plani.md §9)

Schema version **2** (`legged_gym/utils/ued/checkpoint.py`,
`SCHEMA_VERSION = 2`; v2 dropped the standstill-era `valid_task_completion_counts`
and `invalid_outcome_count` fields now that standstill is a reserved bucket that
never reaches the curriculum). `EpisodeCurriculum.state_dict()` writes exactly:

```
schema_version, algorithm, task_space_fingerprint, config_fingerprint,
stage_index, sampler_revision, stage_start_global_steps, probabilities,
previous_returns, current_returns, learning_progress, observed_masks,
stage_return_sums, stage_episode_counts, task_assignment_counts,
task_completion_counts, transition_occupancy, source_label,
rng_bit_generator_state
```

- `load_state_dict` fails closed (raises `ValueError`, never silently resets)
  on schema-version mismatch, algorithm mismatch, task-space fingerprint
  mismatch, or config-fingerprint mismatch
  (`checkpoint.py:10-31` `validate_checkpoint_state`; exercised by
  `tests/test_v5_training_contract.py:198-223`
  `test_curriculum_checkpoint_refuses_foreign_fingerprint`).
- The teacher RNG is `numpy.random.Generator(PCG64)`, checkpointed via
  `rng_bit_generator_state` and restored bit-for-bit
  (`episode_curriculum.py:83`, `251`, `292-296`, `308`) — separate from any
  physics/observation RNG.
- The PPO checkpoint carries this dict under the `"episode_curriculum"` key
  only when `env_cfg.env.ued_enabled` is true and a curriculum is installed
  (`rsl_rl/runners/on_policy_runner.py:556-562` `save()`,
  `607-621` `load()`); every non-UED task (`ued_enabled=False`, including
  `handcrafted_v4` and every pre-existing v3/v4/bench task) leaves this key
  absent and is byte-for-byte unaffected.

## 10. Faz B arm/seed/budget contract (solun_plani.md §11 Faz B, §3)

- Four arms, one task family, flag-selected
  (`curriculum.algorithm ∈ {handcrafted_v4, uniform, lp_acrl, alp}`,
  `go2_v5_config.py:34`, `100-111`), registered as `go2_v5_handcrafted`,
  `go2_v5_uniform`, `go2_v5_lpacrl`, `go2_v5_alp`
  (`legged_gym/envs/__init__.py:258-261`).
- Shared, allowlist-enforced substrate across all four arms
  (`tests/test_v5_training_contract.py`): `num_envs=4096`,
  `episode_length_s=20`, `max_iterations=3000`, `num_steps_per_env=24`,
  `save_interval=200`, `eval_seed=12345`, identical reward/DR/actor-critic
  dicts, 3-axis command with vx support `lin_vel_x=[0,2]` (`v_scale=2.0`) plus
  shared `lin_vel_y`/`ang_vel_yaw=[-1,1]` nuisance and `heading_command=False`.
- ≥3 paired training seeds (not parallel-env replicas) per §3/§11 Faz B; the
  Faz B launcher (`scripts/run_v5_fazB.sh`) defaults to seeds `1 2 3`.
- Checkpoint **saving** may stay every 200 PPO updates; offline SPNTE bank
  validation is **manual** on the scheduled iterations above → `best_spnte.pt`
  is the primary endpoint; online tracking eval (if enabled) still writes
  `best_tracking.pt` only.

## Status vs. subplans/05's "done tanımı"

- `go2_v5_*` registered and `tests/test_v5_training_contract.py` green
  (allowlist enforced) — see verification run in this session.
- This document collects the frozen numbers in one place.
- `scripts/run_v5_fazB.sh` exists, is parameterized (arms/seeds/num_envs/
  max_iterations/num_shards are all overridable, `--smoke` shrinks them for a
  short end-to-end mechanics check), and does not auto-launch the full
  campaign. See this session's smoke-test report for what was actually run
  end-to-end versus what still needs a real multi-seed GPU campaign.
