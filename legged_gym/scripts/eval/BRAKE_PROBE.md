# Brake probe: why the rear legs lift on a MuJoCo hard stop

Sim-to-sim play showed the Go2 lifting its rear legs after a fast run is stopped
in MuJoCo, and (to the eye) not in Genesis. This is the measurement of that.

* Recorder: `brake_probe.py` — raw per-step time series, one backend per run.
* Metrics: `brake_analyze.py` — everything derived offline, so a definition can
  be revised without re-running rollouts.
* Campaigns: `scripts/run_brake_probe.sh` (baseline + teacher A/B),
  `scripts/run_brake_replay.sh` (open-loop replay ladder).

Policy: `logs/go2_moects/Aug03_12-01-45_moe_cts_genesis`, `model_20500.pt`.
Grid: vx ∈ {1.0, 1.5, 2.0} m/s × stop ramp ∈ {0, 0.3, 0.6} s × 16 gait phases
= 144 conditions per backend. Flat plane, nominal physics, observation noise
off, deterministic spawn.

## What had to be fixed before anything could be measured

**The reset is randomized and is not config-gated.** `legged_robot._reset_dofs`
jitters the joints (the moects mixin makes it multiplicative, `default * U(0.5,
1.5)`), `_reset_root_states` draws base linear and angular velocity from
U(±0.5), and the moects config sets `yaw_random_scale = 3.14`. Measured, two
envs in one batch differed by 0.5 m/s *before* either stop trigger. A trigger
offset would then have meant "a different random rollout", not "a different gait
phase" — which is the one confound this probe exists to remove.

**The command is resampled at reset too.** Even with the command overwritten
every step, the observation `reset()` returns is computed before the first
overwrite, so the first action of each rollout was taken against a random
command in [-2, 2]. In a gait that is a large kick, and it was the residual
after the spawn fix.

Both are handled in `make_reset_deterministic` (instance-level overrides; the
class is untouched for every other caller). After it, MuJoCo is bit-exact
across rollouts and Genesis is within 0.002 m/s (batched-GPU float ordering).

## Metric choice

Peak foot height is the wrong metric — a nominal smoke had a *higher* peak foot
height in Genesis while the interesting event was twice as frequent in MuJoCo.
The event is **both rear feet off the ground together**: contact force < 5 N AND
clearance > 3 cm above that backend's own measured resting height, held for
≥ 40 ms.

Two calibrations matter. Foot height is measured against each backend's
observed resting height, because the foot frame origin sits at a different
height in the URDF Genesis loads (~0.019 m) than in the MJCF MuJoCo compiles
(~0.015 m). And every event metric is reported next to the same metric on the
pre-trigger cruise window, because a trot at 1.5–2 m/s already has flight
phases.

## Results

### 1. The event is real, brake-specific, and dose-dependent

Pooled over 144 conditions, student policy:

| metric | Genesis | MuJoCo |
| --- | --- | --- |
| rear pair lift rate | 0.10 | 0.52 |
| rear pair airborne duty | 0.015 | 0.121 |
| longest rear lift | 14 ms | 112 ms |
| max rear clearance (excess over cruise) | 0.061 (−0.006) | 0.080 (+0.040) |
| peak \|pitch\| | 3.1° | 6.9° |
| rear normal impulse | 58.0 Ns | 38.2 Ns |
| front normal impulse | 105.1 Ns | 123.6 Ns |
| torque saturation duty | 0.000 | 0.024 |
| action rate | 0.150 | 0.298 |
| cruise vx (commanded 1.0/1.5/2.0) | 1.37 | 1.55 |
| fall rate | 0.00 | 0.00 |

Front pair lift rate is 0.00 and all-four flight duty is 0.000 in *both*
backends, and the cruise baseline for the rear event is zero — so this is not
the gait's flight phase. Lift rate rises monotonically with brake severity
(ramp 0.6 s → 0.3 s → 0 s gives 0.25 → 0.52 → 0.79 in MuJoCo) and with speed
(0.83 at vx = 2.0), which is the signature of a mechanical braking-impulse
event rather than an encoder artifact.

Note the plant already disagrees *before* the brake: the same command produces
1.37 m/s in Genesis and 1.55 m/s in MuJoCo.

### 2. The privileged teacher does not fix it

| metric | GS/student | GS/teacher | MJ/student | MJ/teacher |
| --- | --- | --- | --- | --- |
| rear pair lift rate | 0.10 | 0.16 | 0.52 | 0.52 |
| rear pair airborne duty | 0.015 | 0.037 | 0.121 | 0.166 |
| peak \|pitch\| | 3.1° | 4.8° | 6.9° | 8.9° |
| action rate | 0.150 | 0.154 | 0.298 | 0.229 |

`act_teacher` reads the privileged vector and still brakes the same way in
MuJoCo (identical lift rate, slightly longer airborne time). **The history
encoder and the distillation contract are exonerated** — this is not a
representation failure, so the 100 ms / 5×20 ms history horizon is not the
thing to change.

### 3. Open-loop replay of the whole window answers nothing

The first attempt cut the control loop for the entire recorded window. Result:
44% falls, 36° pitch, and both rear feet already airborne in 100% of conditions
*before* the trigger. 0.6–1 s of open-loop torque replay destroys the cruise
gait on its own, so the post-stop numbers measured open-loop instability. The
run mode is kept as `--replay_from window` only so the failure is reproducible;
use `trigger` (the default).

Cutting the loop at the trigger instead is better but still dominated by
open-loop drift over a 1 s horizon, and the direction of the effect tracks the
state mismatch rather than the source of the torques — the tell that it is an
artifact:

| run | rear lift rate | fall rate |
| --- | --- | --- |
| GS policy in GS | 0.10 | 0.00 |
| MJ policy in MJ | 0.52 | 0.00 |
| GS torques → MJ, no state sync | 0.94 | 0.58 |
| GS torques → MJ, + state sync | 0.15 | 0.62 |
| MJ torques → GS, no state sync | 0.80 | 0.53 |
| MJ torques → GS, + state sync | 0.01 | 0.15 |

Controls pass, so this is not a plumbing problem: replaying a run's own actions
into its own backend is bit-exact in MuJoCo (max |Δ| = 0.0 on
base_pos/lin_vel/actions/feet_pos), and in Genesis the derived metrics match the
baseline to 2 decimals (0.10/0.10 lift rate, 3.06°/3.10° pitch) despite a late
chaotic divergence in the raw states.

### 4. Same state + same torques + different plant, short horizon

`--replay_sync_state` overwrites the state at the trigger with the source run's,
so the only remaining difference is the physics. Read at a 0.4 s window, before
the open-loop collapse develops:

MuJoCo's torque sequence, given to both plants:

| metric | MuJoCo plant | Genesis plant |
| --- | --- | --- |
| rear pair lift rate | 0.33 | 0.01 |
| rear pair airborne duty | 0.141 | 0.001 |
| max rear clearance (excess) | +0.029 | −0.033 |
| rear normal impulse | 11.9 Ns | 22.2 Ns |
| front normal impulse | 52.4 Ns | 36.3 Ns |

Genesis' torque sequence, given to both plants:

| metric | Genesis plant | MuJoCo plant |
| --- | --- | --- |
| rear pair lift rate | 0.09 | 0.03 |
| rear normal impulse | 20.9 Ns | 27.4 Ns |

(0.2 s window agrees: 0.10 vs 0.00, and 0.03 vs 0.03.)

## Conclusion

The event is an **interaction**, not attributable to either side alone:

* MuJoCo's brake torques lift the rear **only in MuJoCo's plant** (0.33 vs 0.01
  in Genesis, from an identical state).
* Genesis' brake torques lift the rear in **neither** plant (0.09 / 0.03).

So the policy has not simply memorized Genesis. The chain is: the plants differ
(visible before the brake at all — 1.37 vs 1.55 m/s cruise on the same command)
→ the policy sees different proprioception in MuJoCo and emits a more aggressive
brake (action rate 2×, nonzero torque saturation) → MuJoCo's contact/inertia
response to that sequence unloads the rear and pitches the base twice as far.

The privileged teacher failing identically rules out the representation branch.
That leaves the two branches Sol's tree predicts for this pattern:

1. **Targeted DR-v2**, on the axes the current DR is blind to and that the
   impulse split implicates — link inertia and pitch inertia, joint
   armature/damping/friction, contact compliance and solver time constant, foot
   radius/contact geometry. Not the friction scalar: MuJoCo's nominal friction
   is already inside the trained envelope, and the impulse *split* moving is
   what the data shows, not the friction magnitude.
2. **High-speed hard-stop curriculum** plus a short fine-tune from the 20500
   checkpoint. A full-speed stop to exactly zero is a sparse corner of training:
   commands resample every 5 s and the zero-command path mostly zeroes x/y while
   leaving yaw nonzero.

Do (1) first — it is the one that fixes the transfer rather than teaching the
policy to compensate for one simulator's contact model.

## Reproducing

```bash
ROOT=/path/to/LeggedGym-Ex ./scripts/run_brake_probe.sh    # baseline + teacher A/B
ROOT=/path/to/LeggedGym-Ex ./scripts/run_brake_replay.sh   # replay ladder
python legged_gym/scripts/eval/brake_analyze.py --by vx ramp logs/eval/brake/*.npz
```

Genesis runs the 144-condition grid batched in ~15 s; MuJoCo is a single-world
backend (`num_envs == 1`) so it runs them serially, ~4 min.
