# Eval harness (Faz 1)

Method-agnostic OOD evaluation: sweep one domain-randomization axis over a grid,
run every method under the SAME frozen benchmark protocol, and report
distributional metrics. The deliverable of the thesis is *this harness*, not any
single policy.

## Three questions, kept separate

- **OOD sweep** (`sweep.py`): fixed forward command, ONE physics axis swept out of
  the training band, all other DR pinned to nominal -> degradation curve. The
  thesis headline.
- **Transient scenarios** (`transient.py`): time-resolved probes that expose the
  *recovery* the steady-state sweep averages away (see "Transient scenarios"
  below). A single-axis, constant-command sweep measures local sensitivity; these
  measure how fast the policy re-converges after a command change or a shove.
- **In-distribution** (`indist.py`): deterministic policy over the FULL command set
  and the in-dist DR band the policy trained on -> "how good is the learned gait".
  Shares the rollout code with the in-training eval, so `Eval/*` (TensorBoard) and
  the standalone after-training numbers mean the same thing. Drives `best.pt`
  selection during training (argmax `mean_return`, hard-demoted if `fall_rate` >
  `eval_fall_guard`). Config lives in the benchmark runner cfg
  (`eval_interval`/`eval_steps`/`eval_warmup`/`eval_seed`/`eval_fall_guard`;
  `eval_interval=0` disables). Standalone run:

  ```bash
  python legged_gym/scripts/eval/indist.py --task go2_bench_mlp \
      --steps 2000 --out logs/eval/indist_mlp.npz
  ```
- **Headroom / value-of-information controls**: compare policies only when their
  training physics distributions match. The narrow comparison is
  `go2_bench_mlp` vs `go2_bench_oracle_id`; the wide comparison is
  `go2_bench_mlp_wide` vs `go2_bench_oracle`. The first isolates whether true P
  helps in the intended training band; the second asks the same question across
  the full sweep range. Comparing narrow MLP directly with wide oracle confounds
  access to P with training-distribution difficulty.

No-DR remains a useful floor, but is not a substitute for either matched pair.
Do not advance to an estimator claim until at least one realistic matched pair
shows repeatable oracle headroom on a predeclared primary metric.

## Pipeline

1. `sweep.py` builds a benchmark task, loads its checkpoint, tiles the axis grid
   across the parallel envs (one value per contiguous block => `per_point` env
   replicas), runs `steps` steps under a fixed forward command,
   and bins per-env metrics back by grid value. Output: one `.npz` per method.
2. `plot_sweep.py` overlays several `.npz` into method x OOD-point degradation
   curves, shading the in-distribution band.

Only the swept axis varies across envs; every other physics axis is pinned to
nominal (`dr_axes.pin_others_to_nominal`) so the curve is purely that axis's
effect. `auto_reset` stays ON, so each env yields many episodes -> real
distributions.

## Transient scenarios (`transient.py`)

The sweep holds one command and one physics point and reports a distribution over
auto-reset episodes -- deliberately averaging away the transient. Online adaptation
is supposed to help precisely there: re-converge faster after a disturbance even
when steady-state tracking looks identical. `transient.py` adds two time-resolved
probes (`--scenario {step_response,push_recovery}`) as a sibling of `sweep.py`,
reusing its scaffolding verbatim (`resolve_load_run` checkpoint isolation,
`make_registry_args`, `override_cfg_for_eval` physics/command freeze, and the
warmup + `episode_length_buf.zero_()` clock reset). It imports those from `sweep.py`
rather than duplicating them, so `sweep.py` stays the single source of truth.

Unlike the sweep, a transient run measures a SINGLE time window per env and tiles it
across `per_point` independent replicas (== seeds); per-env transient metrics are
then aggregated to mean/std/p25/p50/p75 across replicas (same stats as
`sweep.aggregate`). Each run writes an `.npz` with the same provenance block plus the
scenario params, command schedule / push params, and the mean tracking-error and
tilt time series for the plotter.

- **`step_response`**: drives a deterministic COMMAND schedule identical for every
  env -- `stand -> forward(vx) -> reverse(-vx) -> lateral(vy) -> stop`, each phase
  `--phase_steps` long, values clamped to the training range (`|vx|<=0.5`,
  `|vy|<=1.0`, `|yaw|<=1.0`). The command is overwritten on `env.commands[:, :3]`
  each step (in-episode resampling is disabled by pushing `resampling_time` past the
  window; curriculum/heading are already off from `override_cfg_for_eval`). Metrics:
  whole-schedule tracking-error integral, per-phase settling time (steps until error
  drops below `--tol_lin` and stays), per-phase/peak error, peak tilt, fall rate.
- **`push_recovery`** (deterministic, RNG-matched): holds a fixed forward command,
  warms up, then at a PRE-SCHEDULED step applies the SAME fixed velocity impulse
  `(--dvx, --dvy)` to EVERY env's base -- a constant delta on `dofs_vel[:, 0:2]`
  (the base is a 6-DOF free joint; [0:3] are base world lin vel), mirroring
  `genesis_simulator.push_robots()` but WITHOUT the random draw. This determinism is
  load-bearing, not cosmetic: methods have different obs dims (45 vs 50) and consume
  a different number of RNG draws per forward pass, so a *random* push would draw
  from desynchronised streams and hand each method a different shove -- confounding
  "recovers better" with "got a smaller push". A fixed impulse at a fixed step is
  identical across methods by construction. Metrics over the post-push window:
  recovery-error integral, recovery time (steps to settle below `--tol_lin`), peak
  tilt / peak deviation, and fall rate within the window. Pass `--axis added_mass
  --axis_value 5.0` (any `dr_axes` axis) to run the impulse at an off-nominal physics
  point and ask "does a heavy robot recover worse, and does the oracle recover
  better?".

```bash
# smoke test (small, short) -- both scenarios
python legged_gym/scripts/eval/transient.py --scenario step_response \
    --task go2_bench_mlp --load_run <run> --per_point 64 --out logs/eval/step_mlp.npz
python legged_gym/scripts/eval/transient.py --scenario push_recovery \
    --task go2_bench_mlp --load_run <run> --per_point 64 --dvy 1.0 \
    --out logs/eval/push_mlp.npz

# overlay several methods' transients (same scenario) -> tracking-error / tilt vs step
python legged_gym/scripts/eval/plot_transient.py \
    logs/eval/push_nodr.npz logs/eval/push_mlp.npz logs/eval/push_oracle.npz \
    --out logs/eval/push_recovery.png
```

## Metrics (`metrics.py`)

`fall_rate`, `mean_ep_len`, `falls_per_1k`, `mean_return`, `tracking_lin_err`,
`tracking_ang_err`, `tracking_lin_rmse`, fraction of steps with linear tracking
error above 0.25 m/s, mean tilt, torque-squared, absolute mechanical power, and
action rate, plus contact-foot horizontal slip speed. Each is reported as mean /
std / p25 / p50 / p75 across envs at
every grid point. Fall rate remains important, but a zero fall rate is a censored
result rather than evidence that two methods are equivalent; use the continuous
tracking/stability/effort metrics before increasing difficulty.

## Axes (`dr_axes.py`)

`friction`, `added_mass`, and isolated `com_x` / `com_y` / `com_z` are implemented.
Terrain and the OOD-hidden
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

# real curve, 2 baselines on the friction axis.
# steps MUST be >= max_episode_length+1 (1001 for 20s/0.02dt), else a full-survival
# policy never completes an episode and `mean_return` reads 0. Default is 2000
# (one full episode + margin); go higher for tighter return/fall distributions.
python legged_gym/scripts/eval/sweep.py --task go2_bench_nodr \
    --per_point 256 --steps 2000 --out logs/eval/friction_nodr.npz
python legged_gym/scripts/eval/sweep.py --task go2_bench_mlp \
    --per_point 256 --steps 2000 --out logs/eval/friction_mlp.npz

python legged_gym/scripts/eval/plot_sweep.py \
    logs/eval/friction_nodr.npz logs/eval/friction_mlp.npz \
    --out logs/eval/friction_curve.png
```

`--load_run <dir>` / `--ckpt <n>` pick a specific checkpoint. Default auto-selects
the calendar-latest (mtime) run whose folder carries this task's `run_name`, and
fails loud if none match -- so a sibling method's checkpoint can never be loaded
by accident. For a published curve, prefer passing `--load_run` explicitly.
Repeat a run with a different `--seed` to stack more seeds.

The default command is `(vx, vy, yaw) = (0.5, 0, 0)`. Use `--command_vy` and
`--command_yaw` for in-range lateral/turning stress tests. Keep confirmatory
commands inside the shared training range; command-OOD probes are diagnostic and
must be labelled as such. A single-axis, constant-command sweep measures local
sensitivity, not the full value of online adaptation. The transient command and
fixed-impulse recovery scenarios that complement it live in `transient.py` (see
"Transient scenarios" above); run them before making the final estimator comparison.

## Known caveats

- **Wide-oracle inputs remain valid at every sweep grid point.**
  The oracle is trained over the full eval-sweep range (`friction_range = [0.1, 2.5]`,
  `added_mass_range = [-2.0, 5.0]`), matching the axes in `dr_axes.py`.  Because
  `dr_normalize` uses these same widened ranges to compute the privileged labels, the
  [-1, 1] normalisation clamp never saturates at any grid point. This makes it a
  correctly labelled wide-range reference, not an automatic empirical ceiling;
  only matched-distribution results can establish that it actually performs as one.
- **Observation noise stays ON** (part of the frozen 45-obs protocol) and is not
  per-step seeded, so repeat runs are not bit-identical. Stack seeds instead of
  expecting exact reproduction.

## Status

Smoke-tested on a live local Genesis/CUDA box (Genesis 1.0.0, torch 2.8.0+cu126):
loader isolation, warmup episode-clock reset, `.npz` provenance, and plot input
validation all verified. Reviewed across two passes: command-curriculum freeze bug,
checkpoint cross-method contamination, and warmup metric shift all fixed.

The first `go2_bench_nodr`, `go2_bench_mlp`, and wide `go2_bench_oracle` runs and
single-axis curves exist. The matched `go2_bench_oracle_id` and
`go2_bench_mlp_wide` controls still need training. Existing artifacts predate the
expanded continuous metrics and must be rerun to populate those fields.
