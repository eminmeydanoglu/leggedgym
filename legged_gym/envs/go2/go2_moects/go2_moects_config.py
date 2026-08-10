# Configs for the go2_rl_gym (wty-yy, RSS 2026 MoE locomotion) port.
# Go2MoECTSCommonCfg is the shared substrate (terrain / commands / rewards /
# domain_rand per the vendored go2_config.py); Go2MoECTSCfg(+PPO) is the
# MoE-CTS arm and Go2MoECTSHIMCfg(+PPO) the HIM arm -- the policy architecture
# is swapped by task name only.

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.envs.base.template_cfgs import LeggedRobotCTSCfgPPO, LeggedRobotHIMCfgPPO
from legged_gym.envs.base.common_cfgs import Go2FlatCommonCfg, get_simulator_suffix


# --- fleet size and budget anchors ---------------------------------------
# The vendored repo trains at 8192 envs.  The AWS L4 campaign uses the same
# fleet size, so one PPO update carries the source-faithful sample count.
VENDORED_NUM_ENVS = 8192
NUM_ENVS = 8192
# Planned total PPO iterations. The vendored 150k is a sentinel, not a tuned
# budget: the vendored base default is 1500, the identical 150000 is repeated
# verbatim across all eight arm configs, and the last real curriculum event
# lands at vendored iteration 50000 -- the remaining 100k iterations change
# nothing in the recipe. 30k here is a deliberate budget decision.
VENDORED_MAX_ITERATIONS = 150000
MAX_ITERATIONS = 30000

# The vendored curricula use absolute iteration thresholds, and they mix two
# different kinds of schedule. Rescaling both the same way is what compresses
# the early phase 5x at a 30k budget, so they are anchored separately here:
#
#   * Early SHAPING ramps (zero_command end 1500, lin_vel_z ramp end 1500,
#     correct_base_height ramp end 5000) track a learning-time constant -- how
#     long basic gait takes to form. That does not shrink when the total
#     budget shrinks: pulling the lin_vel_z shaping before the gait is stable
#     fails in a way that reads as "not enough iterations" but is not. It does
#     scale with iteration SIZE, so at 4096 envs the same sample count takes
#     twice as many iterations. -> _shaping_ratio
#
#   * Late command_range steps (vendored 20000, 50000) are staged difficulty
#     and are legitimately a fraction of the budget. They cannot be
#     sample-corrected anyway -- 20000 * 2 / 30000 > 1.0 would mean the first
#     widening never fires. Compression is tolerable here because the
#     difficulty curriculum that actually matters, the terrain level, is
#     performance-gated (_update_terrain_curriculum), not iteration-gated.
#     -> _staged_ratio
_SHAPING_ITER_SCALE = VENDORED_NUM_ENVS / NUM_ENVS   # 1.0 at 8192 envs


def _shaping_ratio(vendored_iter):
    """Vendored absolute iteration -> fraction of THIS run's budget, holding
    the shaping phase's SAMPLE count constant rather than its budget share.
    At 8192 envs / 30k iters: 1500 -> 0.05 (iter 1500), 5000 -> 1/6 (iter 5000).
    """
    return min(vendored_iter * _SHAPING_ITER_SCALE / MAX_ITERATIONS, 1.0)


def _staged_ratio(vendored_iter):
    """Vendored absolute iteration -> fraction of the vendored budget, i.e.
    staged difficulty compresses proportionally when the budget shrinks.
    At 30k iters: 20000 -> iter 4000, 50000 -> iter 10000.
    """
    return vendored_iter / VENDORED_MAX_ITERATIONS


class Go2MoECTSCommonCfg(Go2FlatCommonCfg):
    """Shared go2_rl_gym substrate. Host URDF/XML paths, control gains and
    asset block come from Go2FlatCommonCfg (identical values in the vendored
    repo); everything else follows the vendored go2_config.py."""

    class env(Go2FlatCommonCfg.env):
        num_envs = NUM_ENVS         # 8192 this campaign (vendored); the
                                    # shaping-ramp anchors above are derived
                                    # from this -- keep them together
        episode_length_s = 25       # vendored
        # PPO iteration length assumed by the iteration-based curricula
        # (command_range_curriculum, zero_command_curriculum,
        # curriculum_rewards); the vendored repo hardcodes 24. Keep in sync
        # with CfgPPO.runner.num_steps_per_env.
        wty_steps_per_iteration = 24
        # Planned total PPO iterations the ratio-based curricula resolve
        # against (the vendored 1500/5000/20000/50000 thresholds are stored as
        # their fractions of the vendored 150k budget). Overridden at learn()
        # time by the runner with the effective max_iterations (incl. any
        # --max_iterations CLI override) -- see set_wty_total_iterations.
        # Default matches CfgPPO.runner.max_iterations below (30k).
        curriculum_total_iterations = 30000
        # --- termination (vendored parity; enforced by WtyCurriculumMixin.check_termination) ---
        # Reference go2_rl_gym terminates the SAME step any termination body's
        # contact-force norm exceeds 1.0 N (its tilt/projected-gravity check is
        # commented out; no consecutive-failure counter). The host default is
        # 10 N + tilt check + a fail_to_terminal_time_s (~5-step) counter.
        # Chosen here: 2.5 N, same-step, contact-only. reference go2_rl_gym
        # uses 1.0; threshold raised to 2.5, watch Episode/termination_base_contact.
        base_contact_terminate_threshold = 2.5

    class init_state(Go2FlatCommonCfg.init_state):
        pos = [0.0, 0.0, 0.42]  # vendored; host uses 0.4
        # Reference parity (go2_rl_gym legged_robot._reset_root_states):
        # initial base yaw ~ U(-pi, pi) at every reset, roll/pitch exactly 0.
        # Host consumes this scale in LeggedRobot._reset_root_states via
        # quat_from_euler_xyz with independent per-axis uniform draws, so
        # roll/pitch stay unrandomized (host scales remain 0.0).
        yaw_random_scale = 3.14     # [rad]
        default_joint_angles = {  # vendored go2 (hips +-0.1, rear thighs 1.0)
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

            'FL_thigh_joint': 0.8,   # [rad]
            'RL_thigh_joint': 1.0,   # [rad]
            'FR_thigh_joint': 0.8,   # [rad]
            'RR_thigh_joint': 1.0,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,   # [rad]
            'FR_calf_joint': -1.5,   # [rad]
            'RR_calf_joint': -1.5,   # [rad]
        }

    class asset(Go2FlatCommonCfg.asset):
        # vendored go2_config.py terminate_after_contacts_on = ["base"]; the
        # host Go2FlatCommonCfg default is ["base", "Head"]. Consumed by
        # WtyCurriculumMixin.check_termination with the threshold above.
        terminate_after_contacts_on = ["base"]

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'heightfield'   # moe_grid is heightfield-only; Genesis trains it,
                                    # MuJoCo replays the same raster (mujoco_scene.py)
        # Ground friction 0.5 (host default 1.0). Genesis max-combines link and
        # ground friction, so with per-env link friction U[0.5, 1.5] (see
        # domain_rand.friction_range) the effective friction = max(link, 0.5)
        # = link in [0.5, 1.5]. Global constant: the terrain is a single
        # heightfield entity, per-env ground friction is not possible.
        static_friction = 0.5
        moe_grid = True             # Phase-2 builder branch; wins over curriculum/selected
        curriculum = False          # grid layout comes from moe_grid; the env-side
                                    # game curriculum is driven by WtyCurriculumMixin
        terrain_spacing = 0.5       # [m] gaps between sub-terrains
        border_size = 25            # vendored
        measure_heights = True      # 17x11 = 187 height measurements (privileged obs)
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.,
                             0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 1mx1.6m rectangle
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        # DEVIATION from vendored (5): start every env on row 0, the mildest
        # rendering of its own terrain type -- 5 cm stairs, 5.7 deg slopes,
        # 5 cm obstacles, +/-10 cm waves (see make_moe_terrain's difficulty
        # table in utils/terrain.py). The terrain TYPE mix is fixed by the
        # round-robin column assignment and cannot be ramped without
        # reassigning terrain_types at runtime, so row 0 is the "near-flat"
        # warmup this curriculum can express: envs only reach real stairs by
        # earning promotions through _update_terrain_curriculum. The vendored
        # 5 started every env at mean level 2.5, i.e. 16 cm stairs / 13 deg
        # slopes from iteration 0.
        max_init_terrain_level = 0
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10               # difficulty levels
        num_cols = 20               # terrain type columns
        # vendored go2: [wave, slope, rough_slope, stairs up, stairs down,
        #                obstacles, stepping_stones, gap, flat]
        terrain_proportions = [0.05, 0.20, 0.05, 0.25, 0.10, 0.20, 0.0, 0.0, 0.15]
        move_down_by_accumulated_xy_command = True  # vendored go2 demotion rule

    class commands(LeggedRobotCfg.commands):
        curriculum = False          # vendored uses the iteration-based
                                    # command_range_curriculum below instead of
                                    # the host performance-based command curriculum
        num_commands = 4            # lin_vel_x, lin_vel_y, ang_vel_yaw, heading
        resampling_time = 5.        # vendored; time before commands are changed [s]
        heading_command = False     # vendored
        zero_cmd_prob = 0.0         # host legacy standstill draw off; the vendored
                                    # zero_command_curriculum runs inside the mixin
        # --- vendored dynamic-resample knobs (consumed by WtyCurriculumMixin) ---
        # Iteration thresholds are RATIOS of the planned total budget
        # (curriculum_total_iterations), not absolute iterations: the vendored
        # values are written as their fraction of the vendored 150k budget, so
        # shrinking the budget rescales every curriculum proportionally.
        # start training with zero commands and then gradually increase zero command probability
        zero_command_curriculum = {'start_ratio': 0.0, 'end_ratio': 1500 / 150000,  # vendored end_iter 1500
                                   'start_value': 0.0, 'end_value': 0.1}
        limit_ang_vel_at_zero_command_prob = 0.2  # probability of adding limiting angular velocity commands when zero command is sampled
        limit_vel_prob = 0.2        # probability of limiting linear velocity command
        limit_vel_invert_when_continuous = True  # invert the limit logic when using continuous sample limit velocity commands
        limit_vel = {"lin_vel_x": [-1, 1], "lin_vel_y": [-1, 1], "ang_vel_yaw": [-1, 0, 1]}  # sample from min/zero/max range only
        stop_heading_at_limit = True  # stop heading updates when vel is limited
        dynamic_resample_commands = True  # sample commands with low bounds
        command_range_curriculum = [{  # command range updates at specific budget fractions
            'ratio': 20000 / 150000,    # vendored iter 20000 of 150k
            'lin_vel_x': [-1.0, 1.0],   # min max [m/s]
            'lin_vel_y': [-1.0, 1.0],   # min max [m/s]
            'ang_vel_yaw': [-1.5, 1.5],  # min max [rad/s]
            'heading': [-1.57, 1.57],   # min max [rad]
        }, {
            'ratio': 50000 / 150000,    # vendored iter 50000 of 150k
            'lin_vel_x': [-2.0, 2.0],   # min max [m/s]
            'lin_vel_y': [-1.0, 1.0],   # min max [m/s]
            'ang_vel_yaw': [-2.0, 2.0],  # min max [rad/s]
            'heading': [-1.57, 1.57],   # min max [rad]
        }]
        # per-terrain-type command limits, indexed by semantic terrain id:
        # [wave, slope, rough slope, stairs up, stairs down, obstacles, stepping stones, gap, flat]
        terrain_max_command_ranges = [
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # wave
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # slope
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # rough slope
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stairs up
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stairs down
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # obstacles
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stepping stones
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # gap
            {'lin_vel_x': [-2.0, 2.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-2.0, 2.0], 'heading': [-1.57, 1.57]},  # flat
        ]

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.5, 0.5]     # min max [m/s]
            lin_vel_y = [-0.5, 0.5]     # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]   # min max [rad/s]
            heading = [-1.57, 1.57]     # min max [rad]

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.38
        only_positive_rewards = False   # vendored
        max_contact_force = 147.        # vendored parity (only used by
                                        # _reward_feet_contact_forces, not in scales)
        min_legs_distance = 0.1         # vendored parity (only used by
                                        # _reward_legs_distance, not in scales)
        # Per-link contact-force norm [N] above which _reward_collision counts
        # a penalized-body (thigh/calf) hit (consumed by the WtyCurriculumMixin
        # override; strict >). Reference go2_rl_gym (PhysX) = 0.1; host
        # LeggedRobot hardcodes 10.0. Chosen = 0.1 (reference parity), justified
        # by measurement on Genesis (tests/_measure_collision_force_noise.py,
        # tmp/collision_force_stats.json): the contact-force noise floor on
        # non-contacting links is exactly 0.0 N (p99.9 = 0.0 -- 6/8 penalized
        # links never read nonzero in 512k samples; every nonzero rear-calf
        # reading was traced via robot.get_contacts() to genuine ground contact
        # with planeLink at z~0, see tmp/rear_calf_contact_probe.json), so the
        # PhysX-era worry that 0.1 sits inside solver noise does not apply.
        collision_force_threshold = 0.1
        curriculum_rewards = [  # ratio-based reward scale ramping (WtyCurriculumMixin)
            {'reward_name': 'lin_vel_z', 'start_ratio': 0.0, 'end_ratio': 1500 / 150000, 'start_value': 1.0, 'end_value': 0.0},
            {'reward_name': 'correct_base_height', 'start_ratio': 0.0, 'end_ratio': 5000 / 150000, 'start_value': 1.0, 'end_value': 10.0},
        ]
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        dynamic_sigma = {  # linear interpolation of sigma based on command velocity;
                           # requires the terrain curriculum (WtyCurriculumMixin)
            "min_lin_vel": 0.5,  # min abs linear velocity to have default sigma
            "max_lin_vel": 1.5,  # max abs linear velocity to have max sigma
            "min_ang_vel": 1.0,  # min abs angular velocity to have default sigma
            "max_ang_vel": 2.0,  # max abs angular velocity to have max sigma
            # [wave, slope, rough_slope, stairs up, stairs down, obstacles, stepping_stones, gap, flat]
            "max_sigma": [5/12, 1/4, 1/4, 1/2, 1/2, 3/4, 1, 1, 1/4]
        }

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            dof_acc = -2.5e-7
            dof_power = -2e-5
            torques = -1e-4
            correct_base_height = -1.0
            action_rate = -0.01
            action_smoothness = -0.01
            collision = -1.0
            dof_pos_limits = -2.0
            feet_regulation = -0.05
            hip_to_default = -0.05

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        # Target EFFECTIVE friction: per-env U[0.5, 1.5], drawn once at startup.
        # Reference go2_rl_gym (PhysX): absolute link friction U[0, 2],
        # average-combined with the ground (1.0) -> effective ~[0.5, 1.5].
        # Genesis instead max-combines link and ground friction
        # (genesis/engine/solvers/rigid/collider/contact.py:
        # contact_friction = max(link_base * link_ratio, ground, 1e-2)) and the
        # host draws link_ratio ~ U[friction_range] per env (same ratio on all
        # links) against the MJCF base friction (go2.xml = 1.0), i.e. the range
        # below is the ABSOLUTE per-env link friction. With the terrain at 0.5
        # (terrain.static_friction), effective = max(link, 0.5) = link, so
        # [0.5, 1.5] reproduces the reference's effective distribution.
        # (The vendored [0, 2] with ground 1.0 collapsed to effective [1, 2].)
        # Flag off -> ratio stays at its 1.0 default -> effective friction 1.0.
        friction_range = [0.5, 1.5]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]            # vendored
        randomize_com_displacement = True       # vendored randomize_base_com
        com_pos_x_range = [-0.03, 0.03]
        com_pos_y_range = [-0.03, 0.03]
        com_pos_z_range = [-0.03, 0.03]
        randomize_pd_gain = True                # vendored randomize_pd_gains (per-DOF draws)
        kp_range = [0.9, 1.1]
        kd_range = [0.9, 1.1]
        push_robots = True
        # Vendored push semantics (go2_rl_gym legged_robot._push_robots): each
        # env is pushed when its OWN episode step hits push_interval (200 steps
        # = 4s at dt 0.02, no global lockstep), and the push OVERWRITES the
        # world-frame xy lin vel U(+-0.4) and all ang vel U(+-0.6) instead of
        # the host's additive xy-only push (WtyCurriculumMixin +
        # GenesisSimulator.push_robots_overwrite).
        push_interval_s = 4                     # vendored
        max_push_vel_xy = 0.4                   # vendored
        max_push_ang_vel = 0.6                  # vendored
        randomize_ctrl_delay = True             # vendored randomize_action_delay
        # Reference-faithful delay model (go2_rl_gym legged_robot.step): the
        # delay is re-drawn PER ENV EVERY CONTROL STEP, uniform over {0..4} SIM
        # SUBSTEPS (sim dt 0.005 s x decimation 4 -> 0/5/10/15/20 ms), and the
        # simulator blends the previous/current action targets within the
        # control step (first k substeps keep the previous target, the
        # remaining 4-k use the current one).  This fixes the earlier port's
        # per-episode queue draw from {0,1} control steps: same ~10 ms mean,
        # but the reference's history encoder sees per-step delay jitter while
        # the old port saw a constant per-episode latency.  Genesis-only.
        ctrl_delay_mode = "substep"
        ctrl_delay_substep_range = [0, 4]
        ctrl_delay_step_range = [0, 1]          # legacy queue units; unused in substep mode
        randomize_motor_strength = True         # vendored
        motor_strength_range = [0.8, 1.2]       # vendored
        randomize_motor_zero_offset = True      # vendored
        motor_zero_offset_range = [-0.035, 0.035]   # vendored [rad]
        # Vendored DR deliberately NOT ported (conscious ablation, documented for parity):
        # * randomize_link_mass x[0.9,1.1] (go2_rl_gym legged_robot.py:392-397):
        #   IsaacGym asset props are SHARED across envs, so the reference's draw
        #   has shape (1, num_bodies-1) -- one run-level static scaling seen
        #   identically by every env, i.e. zero per-env entropy and nothing for
        #   the MoE gate or the history encoder to identify. The per-env part of
        #   the body-inertia axis is already covered by randomize_base_mass
        #   (+-1 kg) and randomize_com_displacement (+-3 cm). Genesis could do a
        #   true per-env version via set_mass_shift(envs_idx=...), but that would
        #   be a deliberate DIVERGENCE from the reference (and would need the new
        #   latent exposed to the oracle), so it belongs in a DR-widening
        #   experiment, not in the parity port.
        # * randomize_restitution [0.0, 0.5] (Genesis _randomize_restitution is a no-op).

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0               # vendored; host default 1.0
            height_measurements = 2.5   # vendored; host default 2.0

    class noise(LeggedRobotCfg.noise):
        add_noise = True

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_vel = 1.5   # vendored; host default 0.5


class Go2MoECTSCfg(Go2MoECTSCommonCfg):
    """MoE-CTS arm env dims (vendored GO2Cfg.env + LeggedRobotCfgCTS/MoECTS)."""

    class env(Go2MoECTSCommonCfg.env):
        num_observations = 45
        # obs(45) + base_lin_vel(3) + feet_contact_forces(4) + dof_torques(12)
        #        + dof_acc(12) + height_measurements(187)
        num_privileged_obs = 45 + 3 + 4 + 12 + 12 + 187  # 263
        teacher_env_ratio = 0.75  # vendored; WtyCurriculumMixin rescales
                                  # num_teacher from this when --num_envs differs
        num_teacher = int(Go2MoECTSCommonCfg.env.num_envs * teacher_env_ratio)
        frame_stack = 5                 # vendored history_length
        num_history_obs = num_observations * frame_stack  # 225
        num_latent_dims = 32            # vendored latent_dim
        c_frame_stack = 1               # critic sees the current privileged frame only
        num_single_critic_obs = num_privileged_obs        # 263
        num_critic_obs = c_frame_stack * num_single_critic_obs  # 263
        num_actions = 12


class Go2MoECTSCfgPPO(LeggedRobotCTSCfgPPO):
    """runner_class_name='MoECTSRunner'; ActorCriticMoECTS + PPO_MOE_CTS."""

    runner_class_name = 'MoECTSRunner'

    class policy(LeggedRobotCTSCfgPPO.policy):
        init_noise_std = 1.0                        # vendored
        actor_hidden_dims = [512, 256, 128]         # vendored
        critic_hidden_dims = [512, 256, 128]        # vendored
        privilege_encoder_hidden_dims = [512, 256]  # vendored teacher_encoder_hidden_dims
        expert_num = 8                              # vendored
        student_encoder_hidden_dims = [512, 256, 256]  # vendored
        norm_type = 'l2norm'                        # vendored

    class algorithm(LeggedRobotCTSCfgPPO.algorithm):
        load_balance_coef = 0.01    # vendored (PPO_MOE_CTS gating load-balance loss)
        teacher_env_ratio = 0.75    # vendored; MUST match Go2MoECTSCfg.env.teacher_env_ratio --
                                    # PPO_MOE_CTS derives the interleaved teacher/student
                                    # env split from it and asserts the two agree
        encoder_lr = 1.e-3          # vendored student_encoder_learning_rate
        num_encoder_epochs = 1
        # Remaining vendored PPO hyperparams match the host template defaults:
        # entropy_coef 0.01, clip_param 0.2, schedule 'adaptive', desired_kl 0.01,
        # gamma 0.99, lam 0.95, learning_rate 1e-3, num_learning_epochs 5,
        # num_mini_batches 4, max_grad_norm 1.0. Vendored seed is 0; host default 1 kept.

    class runner(LeggedRobotCTSCfgPPO.runner):
        policy_class_name = 'ActorCriticMoECTS'
        algorithm_class_name = 'PPO_MOE_CTS'
        num_steps_per_env = 24      # keep in sync with env.wty_steps_per_iteration
        experiment_name = 'go2_moects'
        run_name = 'moe_cts' + get_simulator_suffix()
        max_iterations = 30000      # budget decision (vendored 150k ~= 92 h on
                                    # UHeM A100; curricula are ratio-based and
                                    # rescale with this -- see env.curriculum_total_iterations)
        save_interval = 500         # vendored
        # Short, fixed in-distribution student eval at every periodic checkpoint.
        # 150 measured + 20 warm-up control steps costs about 12s on the L4,
        # while fall/tracking trends remain directly comparable over the run.
        eval_interval = 500
        eval_steps = 150
        eval_warmup = 20
        eval_seed = 12345
        eval_fall_guard = 0.25
        # Built once and used only at checkpoints; keeps the 8192-env PPO
        # scene untouched.  384 preserves the 3:1 MoE role split exactly.
        eval_num_envs = 384
        # Terrain-conditioned MoE gating telemetry (TELEMETRY ONLY, see
        # MoECTSRunner._log_terrain_gate_stats): every Nth iteration, snapshot
        # the student history-encoder gating weights on the current student
        # envs' obs history and bucket by semantic terrain id, emitting the
        # MoE/terrain_gate/<name>/expert_<i>, MoE/terrain_entropy/<name>,
        # MoE/terrain_max_weight/<name> and MoE/terrain_specialization
        # TensorBoard scalars. Runs under torch.no_grad() and touches no RNG
        # stream or optimizer state, so it cannot affect training dynamics or
        # resume. 0 disables it entirely; 10 costs one extra ~1.1M-param MLP
        # forward over ~2048 student rows every 10 iterations (negligible
        # next to the PPO update).
        terrain_gate_log_interval = 10


class Go2DenseCTSCfg(Go2MoECTSCfg):
    """Environment-identical dense control for ``go2_moects``."""


class Go2DenseCTSCfgPPO(Go2MoECTSCfgPPO):
    """Dense-encoder control with no isolated Genesis evaluation scene."""

    class policy(Go2MoECTSCfgPPO.policy):
        student_encoder_type = 'dense'
        # 1,087,626 params versus 1,088,264 in the existing MoE encoder.
        student_encoder_hidden_dims = [1024, 810]

    class runner(Go2MoECTSCfgPPO.runner):
        experiment_name = 'go2_dense_cts'
        run_name = 'dense_cts_param_matched' + get_simulator_suffix()
        # A persistent eval scene duplicates Genesis terrain/SDF allocations.
        # Dense-CTS is intended to run at the 8192-env MoE-CTS scale on a
        # 32 GiB L4 host, so keep a single training scene and rely on normal
        # PPO TensorBoard scalars for in-run monitoring.
        eval_interval = 0
        eval_num_envs = 0
        terrain_gate_log_interval = 0


class Go2MoECTSHIMCfg(Go2MoECTSCommonCfg):
    """HIM arm env dims (mirrors Go2BenchHIMCfg.env on the shared substrate)."""

    class env(Go2MoECTSCommonCfg.env):
        num_one_step_obs = 45                           # single proprio frame
        frame_stack = 6                                 # temporal_steps for the estimator
        num_observations = frame_stack * num_one_step_obs   # 270 (actor obs = history)
        c_frame_stack = 5
        num_single_critic_obs = 3 + num_one_step_obs + 5    # [base_lin_vel, one_step, P] = 53
        num_privileged_obs = c_frame_stack * num_single_critic_obs  # 265
        num_actions = 12


class Go2MoECTSHIMCfgPPO(LeggedRobotHIMCfgPPO):
    """runner_class_name='HIMRunner' (inherited); HIMActorCritic + PPO_HIM.

    Policy kwargs mirror Go2BenchHIMCfgPPO; the training budget matches the
    MoE-CTS arm (vendored) for a fair comparison on the shared curriculum.
    The benchmark-harness command_schedule / eval_* block is deliberately not
    carried over: it would fight the vendored dynamic command resampling.
    """

    class policy(LeggedRobotHIMCfgPPO.policy):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        enc_hidden_dims = [128, 64, 16]     # latent = 16
        tar_hidden_dims = [128, 64]
        num_prototype = 32
        temperature = 3.0
        learning_rate = 1.e-3
        max_grad_norm = 10.0

    class runner(LeggedRobotHIMCfgPPO.runner):
        experiment_name = 'go2_moects'
        run_name = 'him' + get_simulator_suffix()
        num_steps_per_env = 24      # keep in sync with env.wty_steps_per_iteration
        max_iterations = 30000      # match the MoE-CTS arm budget decision
                                    # (ratio-based curricula rescale with it)
        save_interval = 500
