# LEGGED_GYM CORE FRAMEWORK

Core legged robot RL framework with multi-simulator support (Genesis, IsaacGym, IsaacLab).

## STRUCTURE

| Directory | Purpose |
|-----------|---------|
| `envs/` | Robot environments extending LeggedRobot |
| `scripts/` | Training and inference entry points |
| `utils/` | Terrain generation, task registry, math utilities |
| `simulator/` | Simulator abstraction layer |

## KEY FILES

| Purpose | File |
|---------|------|
| Training entry | `scripts/train.py` |
| Inference | `scripts/play.py` |
| Base environment | `envs/base/legged_robot.py` |
| Config system | `envs/base/legged_robot_config.py` |
| Task registry | `utils/task_registry.py` |
| Terrain generation | `utils/terrain.py` |
| Math utilities | `utils/math_utils.py` |
| Simulator ABC | `simulator/simulator.py` |

## CONVENTIONS

**Config Pattern**: Nested classes with `class env`, `class rewards`, `class commands`, etc.

```python
class MyRobotCfg(LeggedRobotCfg):
    class env:
        num_observations = 48
    class rewards:
        class scales:
            tracking_lin_vel = 1.0
```

**Task Registration**: Add to `envs/__init__.py`:
```python
task_registry.register("robot_name", RobotClass, Cfg, CfgPPO)
```

**Simulator Selection**: `export SIMULATOR=genesis|isaacgym|isaaclab`

## ANTI-PATTERNS

1. **Observation Changes**: Modifying `obs_buf` requires updating ALL `_reward_*` methods
2. **IsaacGym Reset Bug**: After `reset()`, call `self.simulator.forward()` before reading rigid body states
3. **IsaacLab CPU Tensors**: Domain randomization tensors must be on CPU (`set_material_properties`, `set_masses`, `set_coms`)
4. **Genesis XML**: Must provide XML file path when using Genesis simulator
5. **Terrain Flags**: Cannot use `curriculum=True` with `selected=True` simultaneously
6. **IsaacLab Heightfield**: Heightfield terrain not implemented for IsaacLabSimulator
7. **moe_grid Terrain**: With `terrain.moe_grid=True` (go2_moects tasks) keep `curriculum=False` — the builder wins the dispatch and `WtyCurriculumMixin` drives the env-side game curriculum explicitly
8. **moects Termination**: `go2_moects`/`go2_moects_him` do NOT use the host termination (10 N + tilt + consecutive counter) — `WtyCurriculumMixin.check_termination` is a replacement override: same-step base-contact termination at `cfg.env.base_contact_terminate_threshold` (2.5 N; vendored go2_rl_gym uses 1.0), bodies `["base"]` only, no tilt check. Watch `Episode/termination_base_contact`
9. **moects Friction**: Genesis MAX-combines link and ground friction, so `go2_moects`/`go2_moects_him` set `terrain.static_friction = 0.5` and `domain_rand.friction_range = [0.5, 1.5]` (per-env ratio x MJCF base 1.0 = absolute link friction) — effective friction = max(link, 0.5) = link in [0.5, 1.5], reproducing the vendored PhysX average-combine of U[0,2] with ground 1.0. Do not "fix" the range to [0, 2]: with max-combine that collapses to effective [1, 2]

## PATTERNS

**Type Aliases**: `ObsBuf = Tensor`, `Action = Tensor`, `Reward = Tensor` in base classes

**Config Validation**: Extensive assertions in `LeggedRobot.__init__()` catch config errors early

**Debug Flags**: `cfg.env.debug*` enable visualization (height points, depth images, etc.)
