# V4 nominal headroom matrix: campaign contract

`configs/eval/v4_headroom.yaml` is the authoritative declarative protocol for
the next V4 headroom measurement.  It holds `eval_seed=1` fixed and compares
the matched blind MLP and privileged Superset-Oracle on the same selected terrain
geometry.  It is a headroom precondition, not an adaptation-method leaderboard:
SysID/RMA/DreamWaQ/HIM follow only if this pair has usable headroom. The primary
matrix is deliberately small and decision-oriented: it is a discovery bank,
not a terrain leaderboard.

- terrain type: `random_uniform` and `stairs_up`;
- pinned terrain level: `3` (moderate);
- forward command: `vx=0.6, 1.0` (`vy=yaw_rate=0`);
- exactly one physical axis per primary cell: `mass_kg`, `com_x_m`, or
  `friction`, each at a declared ID point and nearest OOD extension; the mass
  axis carries the sole nominal anchor (`0 kg`). Every unswept
  named axis is pinned to nominal (`0 kg`, `0 m`, `1.0`).

ID/OOD labels are checked against the V4 training support stored in the config:
mass `[-2,+5] kg`, CoM-x `[-0.08,+0.08] m`, and friction `[0.5,1.25]`.
Every ID point must lie inside its interval and every OOD point outside it.

There is deliberately no secondary correlated-stress/payload tier in this
discovery. Testing such combinations before knowing that the matched MLP/Oracle
pair has usable headroom is budget-expensive and does not answer the adaptation
decision. The schema records the secondary tier as empty, so any later addition
changes the campaign fingerprint and budget explicitly.

There are 24 unique primary worlds: `2 terrain families × 1 level × 2 commands
× 6 isolated-physics points`. MLP and Oracle are each evaluated for both
training seeds, yielding **96 planned discovery cells** and zero secondary
cells. A cell has 16 replicas, 50 warmup steps, and 500 measured steps (1 s +
10 s at the V4 50 Hz control rate), for 844,800 total simulated
environment-control steps including warmup. The runner calculates and enforces
this explicit budget; it prevents accidental duplication when a loop is
changed.

The tracking inclusion gates are the explicit `scorecard` values in the config:
MLP and Oracle fall rate at most `fall_gate_pp`, Oracle achieved-speed ratio at
least `achieved_speed_ratio`, absolute error headroom at least
`absolute_headroom`, and relative headroom
`(MLP error - Oracle error) / MLP error` at least `relative_headroom`. These
values are part of the protocol fingerprint; changing one starts a distinct
campaign. An adaptation method that exceeds the same fall gate remains visible
in survival output but contributes `0` to the headline GapClosed score and is
marked `survival-gated` in the HTML instead of receiving a misleading percent.

## Current runner status

`legged_gym.scripts.eval.v3_eval` now recognizes
`v4_headroom_matrix_v1`. It uses a dedicated `h4` path rather than the legacy
mixed-column terrain suites:

| Required contract | Dedicated `h4` behavior |
| --- | --- |
| A cell selects one named terrain type | Terrain type and level are pinned and verified at runtime. |
| Plan `type × level × vx` | Every dimension is present in the cell identity and artifact fingerprint. |
| Isolate `mass/com_x/friction` | The selected axis is applied; unswept axes are reset to nominal and read back. |
| Keep combined stress separate | Secondary cells carry a separate tier and cannot enter the primary headline. |
| Fail loud on protocol drift | Resume and aggregation reject mismatched fingerprints, hashes, and identities. |

The `v4_headroom_matrix_v1` planner/runner:

1. rejects unsupported schemas and invalid matrix declarations;
2. expands primary cells as
   `model × policy_seed × terrain_type × level × vx × axis × band × value`,
   setting all other named physics axes to `nominal_physics` before warmup;
3. expands secondary cells independently as
   `model × policy_seed × terrain_type × terrain_level × vx × scenario`;
4. writes the cell identity (`tier`, type, level, command, selected axis, band,
   requested and validated physics, `eval_seed`, and terrain hash) into every
   artifact; and
5. rejects aggregation that combines tiers, geometry hashes for a fixed
   `(type, level, eval_seed)`, or an incomplete baseline/oracle/method identity.

Each model/seed is pinned with `run_paths`; ambiguous wildcards are a hard
error, never a newest-mtime choice. Every cell also contains a campaign
fingerprint covering the training-seed population, compared model definitions,
checkpoint-selection file, and follow-up manifest (when present). This is
separate from the world-protocol fingerprint so a policy/checkpoint drift
cannot reuse old artifacts merely because the terrain matrix is unchanged.

`v4_headroom_smoke.yaml` is intentionally tiny (four methods × two training
seeds, one terrain type/level/velocity, two primary mass points, and one
secondary payload cell). It is an operational adapter smoke, not a subset claim
or scientific result: it validates matrix expansion, paired-seed scoring,
method adapters, pinning, height-map/terrain-hash identity, strict loading,
report generation, and resume before discovery. Its primary terrain is the
moderate `stairs_up/L3` world at `vx=0.6`, inside discovery support.

## Discovery to adaptation follow-up

Do **not** hand-pick worlds after discovery. Run
`python -m legged_gym.scripts.eval.v4_followup` with the completed discovery
config/root. It writes two immutable outputs:

1. a manifest containing only primary worlds for which `tracking_include=true`
   for *every* configured training seed; and
2. an adaptation-only V4 config for DreamWaQ/HIM (or another explicitly named
   adaptation policy).

The manifest records the discovery artifact-root; SHA-256 values for
`headline.json`, `scorecard_worlds.csv`, `raw_cells.csv`, and
`run_selection.json`; both fingerprints; exact world identity; both reference
seeds; selected checkpoint identity; and the SHA-256 plus frozen metrics of
every MLP/Oracle NPZ. The follow-up runner plans only declared adaptation
labels; the canonical DreamWaQ/HIM deploy entries are injected from
`configs/eval/v4_adaptive_methods.yaml`, not inferred from discovery models.
Its aggregate imports only the frozen MLP/Oracle rows after all of those hashes
and metrics match. A changed source hash, reference artifact, fingerprint,
incomplete reference, absent method checkpoint, secondary-tier world, or zero
eligible worlds is a hard stop. It never silently expands a method run to
non-eligible worlds.

Example, after an authorized completed discovery:

```bash
python -m legged_gym.scripts.eval.v4_followup \
  --source-config configs/eval/v4_headroom.yaml \
  --methods-template configs/eval/v4_adaptive_methods.yaml \
  --output-manifest logs/eval/v4_headroom/followup/manifest.json \
  --output-config logs/eval/v4_headroom/followup/config.yaml \
  --output-root logs/eval/v4_adaptive_followup

python -m legged_gym.scripts.eval.v3_eval resolve-runs \
  --config logs/eval/v4_headroom/followup/config.yaml \
  --models DreamWaQ,HIM-fixed --strict
python -m legged_gym.scripts.eval.v3_eval run \
  --config logs/eval/v4_headroom/followup/config.yaml --suite all --resume
python -m legged_gym.scripts.eval.v3_eval aggregate \
  --config logs/eval/v4_headroom/followup/config.yaml
```
