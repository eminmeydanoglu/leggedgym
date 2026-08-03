# RSL-RL MODULE

**Generated:** 2025-04-03

## OVERVIEW

RL algorithms module with 9 PPO variants for legged robot locomotion.
Specialized implementations for sim-to-real transfer, terrain awareness, and motion priors.

## STRUCTURE

```
rsl_rl/
├── algorithms/     # PPO implementations (9 variants)
├── modules/        # Actor-critic networks per algorithm
├── runners/        # Training orchestration + registry
├── storage/        # Rollout buffers per algorithm
├── env/            # VecEnv abstract interface
└── utils/          # Runner registry, symmetry utils
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Base PPO | `algorithms/ppo.py` | Standard proximal policy optimization |
| Base runner | `runners/on_policy_runner.py` | Generic training loop |
| Actor-critic | `modules/actor_critic.py` | Base policy network |
| Base storage | `storage/rollout_storage.py` | Standard rollout buffer |
| Add new method | Create alg + module + runner + storage | Follow existing pattern |
| Runner selection | `utils/runner_registry.py` | Dynamic runner instantiation |

## ALGORITHM VARIANTS

| Variant | File | Key Feature |
|---------|------|-------------|
| PPO | `algorithms/ppo.py` | Base algorithm |
| PPO_TS | `algorithms/ppo_ts.py` | Teacher-Student for sim-to-real |
| PPO_EE | `algorithms/ppo_ee.py` | Explicit state estimator |
| PPO_CTS | `algorithms/ppo_cts.py` | Concurrent Teacher-Student |
| PPO_DreamWaQ | `algorithms/ppo_dreamwaq.py` | Implicit terrain imagination |
| PPO_TSDepth | `algorithms/ppo_ts_depth.py` | TS with depth encoder |
| PPO_AMP | `algorithms/ppo_amp.py` | Adversarial Motion Priors |
| PPO_CTS_AMP | `algorithms/ppo_cts_amp.py` | Combined CTS + AMP |
| PPO_HIM | `algorithms/ppo_him.py` | HIMLoco (hybrid internal model) |
| PPO_MOE_CTS | `algorithms/ppo_moe_cts.py` | MoE Concurrent TS (go2_rl_gym port) |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| PPO | Class | `algorithms/ppo.py` | Base RL algorithm |
| BaseAlgorithm | Class | `algorithms/base_algorithm.py` | Algorithm interface |
| ActorCritic | Class | `modules/actor_critic.py` | Policy network |
| OnPolicyRunner | Class | `runners/on_policy_runner.py` | Training orchestration |
| RolloutStorage | Class | `storage/rollout_storage.py` | Experience buffer |
| VecEnv | ABC | `env/vec_env.py` | Environment interface |
| runner_registry | Instance | `utils/runner_registry.py` | Runner factory |

## CONVENTIONS

**Algorithm Pattern**: Extend `BaseAlgorithm`, implement `init_storage()`, `update()`, `act()`.

**Runner Registration**: Add to `runners/__init__.py`:
```python
from .my_runner import MyRunner
runner_registry.register("MyRunner", MyRunner)
```

**Component Matching**: Algorithm + Module + Storage + Runner must be compatible.
Example: `PPO_TS` uses `ActorCriticTS`, `RolloutStorageTS`, `TSRunner`.

**Config Selection**: Runner class specified in env config's `runner_class_name` field.

## TRAINING FLOW

1. **Initialization**: Runner creates actor_critic, algorithm, storage
2. **Rollout Collection**: `learn()` calls `env.step()` for N steps, stores transitions
3. **Update**: `algorithm.update()` computes PPO loss, runs backprop
4. **Logging**: TensorBoard + optional WandB, model checkpointing

## ANTI-PATTERNS

1. **Storage Mismatch**: Using base `RolloutStorage` with `PPO_TS` (needs teacher/student buffers)
2. **Missing Registration**: New runner not added to `runners/__init__.py`
3. **Device Mismatch**: Tensors must all be on same device as runner

## ALGORITHM CONFIGURATION

### PPO Parameters (cfg.algorithm)
```python
value_loss_coef = 1.0          # Value function loss weight
use_clipped_value_loss = True  # Clip value updates
clip_param = 0.2               # PPO clipping epsilon
entropy_coef = 0.01            # Entropy bonus weight
num_learning_epochs = 5        # Gradient updates per iteration
num_mini_batches = 4           # Mini-batches per update
learning_rate = 1.e-3          # Adam learning rate
schedule = 'adaptive'          # 'adaptive' or 'fixed'
gamma = 0.99                   # Discount factor
lam = 0.95                     # GAE lambda
desired_kl = 0.01              # Target KL divergence
max_grad_norm = 1.0            # Gradient clipping
use_spo = False                # Simple Policy Optimization
```

### Policy Network (cfg.policy)
```python
init_noise_std = 1.0
actor_hidden_dims = [512, 256, 128]
critic_hidden_dims = [512, 256, 128]
activation = 'elu'  # 'elu', 'relu', 'selu', 'tanh'

# RNN (optional)
rnn_type = 'lstm'
rnn_hidden_size = 512
rnn_num_layers = 1
```

### Runner Parameters (cfg.runner)
```python
num_steps_per_env = 24    # Rollout length
max_iterations = 1500     # Total training iterations
save_interval = 50        # Checkpoint frequency
experiment_name = 'test'  # Log directory name
resume = False            # Resume from checkpoint
load_run = -1             # Run ID to load (-1 = latest)
checkpoint = -1           # Checkpoint ID (-1 = latest)
```

### Method-Specific Configurations

**Teacher-Student (PPO_TS)**:
- `num_encoder_epochs`: History encoder training epochs
- `history_encoder_type`: Encoder architecture ('MLP' or 'TCN')
- `history_encoder_hidden_dims`: History network dimensions
- `privilege_encoder_hidden_dims`: Privilege network dimensions

**Explicit Estimator (PPO_EE)**:
- `estimator_target`: Target variables for estimation
- `estimator_hidden_dims`: Estimator network dimensions

**DreamWaQ (PPO_DreamWaQ)**:
- `depth_encoder_dims`: Depth encoder network dimensions
- `use_terrain_imagination`: Enable terrain imagination module

**MoE-CTS (PPO_MOE_CTS, go2_rl_gym port)**:
- Uses `MoECTSRunner` + `RolloutStorageMoECTS` (MoE-only subclasses; `CTSRunner`/`RolloutStorageCTS` unchanged); classes resolved via eval() in `cts_runner.py` namespace
- Role mapping is INTERLEAVED (reference moe_cts.py:96-102, `compute_role_env_idxs(num_envs, teacher_env_ratio, device)` in `ppo_moe_cts.py`): every `int(1/student_env_ratio)`-th env is a student, NOT the host CTS contiguous `[0, num_teacher)` / `[num_teacher:)` blocks; storage tensors stay in env order (act/bootstrap gather by role and scatter back), only the role-partitioning points (act, rollout values, per-role GAE, advantage normalization, minibatch generator) use the idxs. `teacher_env_ratio` must match `cfg.env.teacher_env_ratio` (asserted)
- Role-aware critic: `ActorCriticMoECTS.evaluate(critic_obs, obs_history=None, is_teacher=True)` — teacher envs use the privilege latent, student envs the MoE history latent; critic input is `[latent.detach(), critic_obs]` in BOTH roles (reference parity), so the critic loss never trains either encoder
- Role-aware bootstrap: `PPO_MOE_CTS.compute_returns(critic_obs, obs_history)`; rollout values also role-aware via `PPO_MOE_CTS.act()`
- Storage contract: `RolloutStorageMoECTS.mini_batch_generator` yields `(teacher_batch, student_batch, teacher_critic, student_critic)`; critic batches are role-pure `CriticMiniBatch` namedtuples with sample-aligned `observation_histories`
- Advantage normalization is GLOBAL over concatenated teacher+student advantages (reference rollout_storage_cts.py:141-143 parity), deliberately NOT the host CTS per-role normalization — per-role GAE recurrences still run on the interleaved idx gathers
- Adaptive-LR KL is computed over the CONCATENATED teacher+student batch (reference moe_cts.py:131-149 parity), NOT the teacher arm only like the host PPO_CTS; student old mu/sigma come from the sample-aligned student critic minibatch
- `load_balance_coef`: Gating load-balance loss weight (default 0.01)
- `encoder_lr`: Student MoE encoder learning rate (separate optimizer, as PPO_CTS); encoder trained ONLY by distillation + load-balance, never by RL/critic losses
- Policy kwargs: `expert_num` (8), `student_encoder_hidden_dims` ([512,256,256]), `norm_type` ('l2norm'/'simnorm')
- `ActorCriticMoECTS.history_encoder` is the MoE student encoder; gating weights come from `history_encoder.forward_with_weights(obs)`; plain `forward()` returns the latent only and is state-free so `PolicyExporterMoECTS` can `torch.jit.script` it
- Actor input is `cat([latent, obs])` — latent FIRST, unlike the host `ActorCriticCTS`. This matches go2_rl_gym so its published deploy policies load; convert them with `legged_gym/scripts/import_go2_rl_gym_policy.py` (playback/eval only — they carry no teacher encoder or critic)
- Deploy/eval contract: `MoECTSRunner.deploy_state_prefixes() == ("actor.", "history_encoder.")` and `get_eval_adapter()` returns `HistoryObsAdapter` over `act_student`
- Telemetry (per update, TB + terminal): `Loss/latent_mse`, `Loss/load_balance`, `Loss/student_encoder_total`, `MoE/gating_entropy`, `MoE/effective_experts`, `MoE/expert_usage_{min,max,std}`, `MoE/expert_usage_{i}` per expert
- Per-role training health (reference parity): `MoECTSRunner.learn` splits done-env reward sums/lengths by the interleaved `alg.teacher_env_idxs` and `MoECTSRunner._log_role_split_stats` writes `Train/mean_{teacher,student}_reward` and `Train/mean_{teacher,student}_episode_length` (+ `/time` variants) alongside the unchanged aggregate `Train/mean_reward`/`Train/mean_episode_length`
