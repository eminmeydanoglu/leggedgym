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

## Upstream published MoE-CTS specialization probe

`probe_upstream_moe_cts.py` is the source-contract path for the published
`go2_rl_gym` policies. It builds the upstream Python `ActorCriticMoECTS`
network locally and records the runner-owned oldest-to-newest `5x45` history,
gate `[8]`, expert outputs `[8,32]`, raw weighted latent, post-mixture
L2-normalized latent, learned action, uniform/shuffled/top-1 gate actions, and
all eight single-expert actions. Raw `model_*.pt` training checkpoints are
loaded strictly and provide the teacher only when the checkpoint actually
contains the teacher/critic weights; a student/actor-only deployment bridge is
reported as `checkpoint_schema=deployment_bridge` and
`teacher_available=false`.

The workspace reference checkout has deployable TorchScript artifacts under
`go2_rl_gym/deploy/pre_train/go2/`. They expose the student MoE and actor
weights, so the probe can run a deployment-bridge smoke, but they are not raw
HF/upstream training checkpoints and do not make privileged observations
available:

```bash
# One short, physically independent closed-loop rollout per requested route.
# max-rollouts is a per-route cap, not a global cap.
SIMULATOR=mujoco PYTHONPATH=. .venv/bin/python \
  legged_gym/scripts/eval/probe_upstream_moe_cts.py --mode closed_loop \
  --checkpoint go2_rl_gym/deploy/pre_train/go2/go2_moe_cts_high_slope_thre_164k_0.6715.pt \
  --reference-root go2_rl_gym --terrains flat --commands 1,0,0 \
  --duration-s 0.02 --simulation-dt 0.002 --control-decimation 1 \
  --max-rollouts 1 \
  --route-modes learned,uniform,top1,fixed_expert_0,fixed_expert_1 \
  --seed 17 --out logs/eval/upstream_moe_cts/164k_route_smoke
```

Closed-loop routes are `learned`, `uniform`, `top1`, and
`fixed_expert_0` ... `fixed_expert_7`. Every route creates a fresh MuJoCo
state/data object and a fresh history, and only that route's selected action
(`route_action`) is applied to physics. A fall terminates the route; the final
row has `done=true`, `survival_duration_s` is reported, and no remaining
post-fall rows are written. `metrics.json` contains route-specific tracking
error, fall rate, survival duration, and achieved command velocity under
`closed_loop_metrics`, while `probe.npz` contains `route_mode`, `episode_id`,
`episode_step`, `tracking_error`, `achieved_command_velocity`, and all
same-state intervention actions for audit.

`shuffled` is deliberately offline-only. It assigns gates across rows in a
fixed bank and therefore has no reliable single-environment causal meaning;
the CLI rejects `shuffled` in `--route-modes` instead of silently presenting
it as a closed-loop route. The offline path remains:

```bash
SIMULATOR=mujoco PYTHONPATH=. .venv/bin/python \
  legged_gym/scripts/eval/probe_upstream_moe_cts.py --mode offline \
  --checkpoint /path/to/raw/model_164000.pt \
  --bank /path/to/fixed_bank.npz \
  --seed 17 --out logs/eval/upstream_moe_cts/164k_offline
```

`probe.npz` is the machine-readable per-step record and `metrics.json` contains
gate entropy/effective-expert count/mean-max/marginal usage, pairwise expert
latent cosine/L2/norm, same-state learned-vs-uniform/shuffled/top-1 action
MSE, pairwise single-expert action separation, and deterministic
`command_id`/`terrain_id` classifier probes. Classifier probes use a
group-aware, class-stratified train/test split and report ordinary accuracy,
balanced accuracy, majority baseline, confusion matrix, class counts, and
group disjointness. If a class is absent from either side, or a group crosses
classes, the result is `available=false` with a specific reason. Continuous
command regression remains under `continuous_command_regression` as an
auxiliary diagnostic; it is not a replacement for command classification.

The six paper command labels are `forward`, `backward`,
`strafe_left`, `strafe_right`, `turn_left`, and `turn_right`; terrain assets
are `flat` and `stairs` (exact in the reference checkout), while `wave` and
`obstacle` currently run explicitly labelled `cross_slope` and `race_track`
proxies because the checkout has no exact standalone assets for those two
families. MuJoCo records `teacher_available=false` and
`privileged_obs_source=unavailable_not_fabricated` for deployment bridges; a
raw training checkpoint and a source-faithful privileged bank are still
required for a teacher comparison. The current workspace has no raw HF
training checkpoint, so that remains an explicit blocker.

JIT parity is a separate ABI check with a concrete machine-readable result:

```bash
SIMULATOR=mujoco PYTHONPATH=. .venv/bin/python \
  legged_gym/scripts/eval/probe_upstream_moe_cts.py --mode jit_parity \
  --checkpoint go2_rl_gym/deploy/pre_train/go2/go2_moe_cts_high_slope_thre_164k_0.6715.pt \
  --seed 17 --out logs/eval/upstream_moe_cts/164k_jit_parity
```

The run writes `metrics.json` with the exact command, checkpoint SHA-256,
sample/history reset semantics, and `parity.status`,
`parity.max_abs_action_error`, and `parity.mean_abs_action_error`. The local
164k check returned `status=PASS`, `max_abs_action_error=0.0`, and
`mean_abs_action_error=0.0` at tolerance `1e-5`; this proves deploy ABI/action
parity only, not privileged-teacher availability.

Interpretation matrix:

| Observation | Safe interpretation | Do not claim |
|---|---|---|
| learned-vs-uniform/shuffled action MSE is nonzero | routing changes the policy output on the same state | lower MSE means better locomotion or tracking |
| low gate entropy/effective experts | routing is concentrated on this bank | causal expert ownership or better locomotion |
| large pairwise expert latent/action separation | experts are functionally different | any one expert is a competent policy alone |
| command/terrain classifier above held-out majority baseline | gate/latent encodes that ID label on this bank | source-environment causality without grouped class-stratified validation |
| closed-loop tracking/survival differs by route | that route's physically measured outcome differs under the stated protocol | action MSE or one short smoke proves a method win |
| teacher/oracle unavailable | deployment-only measurement boundary is honest | a fabricated privileged teacher comparison |

The short smoke is a wiring/lifecycle check, not a method comparison. Re-run
each checkpoint with the same bank, route list, terrain/command protocol, and
seed before comparing metrics, and keep the exact checkpoint/bank SHA-256
values from `metrics.json`.

## Status

Smoke-tested on a live local Genesis/CUDA box (Genesis 1.0.0, torch 2.8.0+cu126):
loader isolation, warmup episode-clock reset, `.npz` provenance, and plot input
validation all verified. Reviewed across two passes: command-curriculum freeze bug,
checkpoint cross-method contamination, and warmup metric shift all fixed.

The first `go2_bench_nodr`, `go2_bench_mlp`, and wide `go2_bench_oracle` runs and
single-axis curves exist. The matched `go2_bench_oracle_id` and
`go2_bench_mlp_wide` controls still need training. Existing artifacts predate the
expanded continuous metrics and must be rerun to populate those fields.
