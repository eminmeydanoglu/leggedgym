# Configs for the go2_rl_gym (wty-yy, RSS 2026 MoE locomotion) port.
# Go2MoECTSCommonCfg is the shared substrate (terrain / commands / rewards /
# domain_rand per the vendored go2_config.py); Go2MoECTSCfg(+PPO) is the
# MoE-CTS arm and Go2MoECTSHIMCfg(+PPO) the HIM arm -- the policy architecture
# is swapped by task name only.

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.envs.base.template_cfgs import LeggedRobotCTSCfgPPO, LeggedRobotHIMCfgPPO
from legged_gym.envs.base.common_cfgs import Go2FlatCommonCfg, get_simulator_suffix


class Go2MoECTSCommonCfg(Go2FlatCommonCfg):
    """Shared go2_rl_gym substrate. Host URDF/XML paths, control gains and
    asset block come from Go2FlatCommonCfg (identical values in the vendored
    repo); everything else follows the vendored go2_config.py."""

    class env(Go2FlatCommonCfg.env):
        num_envs = 8192             # vendored; host default 4096
        episode_length_s = 25       # vendored
        # PPO iteration length assumed by the iteration-based curricula
        # (command_range_curriculum, zero_command_curriculum,
        # curriculum_rewards); the vendored repo hardcodes 24. Keep in sync
        # with CfgPPO.runner.num_steps_per_env.
        wty_steps_per_iteration = 24

    class init_state(Go2FlatCommonCfg.init_state):
        pos = [0.0, 0.0, 0.42]  # vendored; host uses 0.4
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

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'heightfield'   # moe_grid is heightfield-only (Genesis handoff path)
        moe_grid = True             # Phase-2 builder branch; wins over curriculum/selected
        curriculum = False          # grid layout comes from moe_grid; the env-side
                                    # game curriculum is driven by WtyCurriculumMixin
        terrain_spacing = 0.5       # [m] gaps between sub-terrains
        border_size = 25            # vendored
        measure_heights = True      # 17x11 = 187 height measurements (privileged obs)
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.,
                             0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 1mx1.6m rectangle
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        max_init_terrain_level = 5  # vendored
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
        # start training with zero commands and then gradually increase zero command probability
        zero_command_curriculum = {'start_iter': 0, 'end_iter': 1500, 'start_value': 0.0, 'end_value': 0.1}
        limit_ang_vel_at_zero_command_prob = 0.2  # probability of adding limiting angular velocity commands when zero command is sampled
        limit_vel_prob = 0.2        # probability of limiting linear velocity command
        limit_vel_invert_when_continuous = True  # invert the limit logic when using continuous sample limit velocity commands
        limit_vel = {"lin_vel_x": [-1, 1], "lin_vel_y": [-1, 1], "ang_vel_yaw": [-1, 0, 1]}  # sample from min/zero/max range only
        stop_heading_at_limit = True  # stop heading updates when vel is limited
        dynamic_resample_commands = True  # sample commands with low bounds
        command_range_curriculum = [{  # command range updates at specific training iterations
            'iter': 20000,
            'lin_vel_x': [-1.0, 1.0],   # min max [m/s]
            'lin_vel_y': [-1.0, 1.0],   # min max [m/s]
            'ang_vel_yaw': [-1.5, 1.5],  # min max [rad/s]
            'heading': [-1.57, 1.57],   # min max [rad]
        }, {
            'iter': 50000,
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
        curriculum_rewards = [  # iteration-based reward scale ramping (WtyCurriculumMixin)
            {'reward_name': 'lin_vel_z', 'start_iter': 0, 'end_iter': 1500, 'start_value': 1.0, 'end_value': 0.0},
            {'reward_name': 'correct_base_height', 'start_iter': 0, 'end_iter': 5000, 'start_value': 1.0, 'end_value': 10.0},
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
        friction_range = [0.0, 2.0]             # vendored
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
        push_interval_s = 4                     # vendored
        max_push_vel_xy = 0.4                   # vendored
        randomize_ctrl_delay = True             # vendored randomize_action_delay (0~20ms = 0-1 control steps)
        ctrl_delay_step_range = [0, 1]
        # Vendored DR without a host equivalent (kept off, documented for parity):
        # randomize_link_mass x[0.9,1.1], randomize_motor_zero_offset +-0.035,
        # randomize_motor_strength [0.8,1.2], max_push_ang_vel 0.6,
        # randomize_restitution [0.0,0.5] (Genesis _randomize_restitution is a no-op).

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
        max_iterations = 150000     # vendored
        save_interval = 500         # vendored


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
        max_iterations = 150000     # match the MoE-CTS arm (vendored budget)
        save_interval = 500
