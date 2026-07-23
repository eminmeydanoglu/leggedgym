"""V4: V3 dynamic-physics methods trained on the ETH game terrain curriculum."""

from legged_gym.envs.base.common_cfgs import Go2BenchmarkV4TerrainCfg, get_simulator_suffix
from legged_gym.envs.go2.go2_v3_mlp.go2_v3_mlp_config import Go2V3CfgPPO
from legged_gym.envs.go2.go2_v3_sysid.go2_v3_sysid_config import Go2V3SysIDCfgPPO
from legged_gym.envs.go2.go2_v3_rma.go2_v3_rma_config import Go2V3RMACfgPPO
from legged_gym.envs.go2.go2_v3_dreamwaq.go2_v3_dreamwaq_config import Go2V3DreamwaqCfgPPO
from legged_gym.envs.go2.go2_v3_him_fixed.go2_v3_him_fixed_config import Go2V3HIMFixedCfgPPO
from legged_gym.envs.go2.go2_v3_superset_oracle.go2_v3_superset_oracle_config import Go2V3SupersetOracleCfgPPO


# The V3 runners inherit a command_schedule whose final stage re-widens
# lin_vel_x to [-1, 2] at iteration 1500 (a flat sprint field).  On terrain a
# 2 m/s command over 15 cm stairs is physically unachievable and only inflates
# tracking error, so V4 overrides the schedule to end on the same [-1, 1]
# validation field its env config declares.  This is a shared constant across
# all six methods, so it does not bias the method comparison.
V4_COMMAND_SCHEDULE = [
    {"start_iteration": 0, "lin_vel_x": [-0.5, 0.5]},
    {"start_iteration": 500, "lin_vel_x": [-1.0, 1.0]},
]


class Go2V4MlpCfg(Go2BenchmarkV4TerrainCfg):
    class env(Go2BenchmarkV4TerrainCfg.env):
        num_observations = 45
        num_privileged_obs = 48
        num_actions = 12


class Go2V4SysIDCfg(Go2BenchmarkV4TerrainCfg):
    class env(Go2BenchmarkV4TerrainCfg.env):
        num_single_obs = 45
        frame_stack = 20
        num_estimator_features = num_single_obs * frame_stack
        num_estimator_labels = 8
        c_frame_stack = 5
        num_single_critic_obs = num_single_obs + num_estimator_labels
        num_privileged_obs = c_frame_stack * num_single_critic_obs
        num_actions = 12


class Go2V4RMACfg(Go2BenchmarkV4TerrainCfg):
    class env(Go2BenchmarkV4TerrainCfg.env):
        num_observations = 45
        frame_stack = 20
        num_history_obs = num_observations * frame_stack
        height_map_dim = 17 * 11
        rma_teacher_height_map = True
        # Teacher sees [P5, true velocity, true 17x11 height map].  The
        # student is still 20x45 proprio history and must reproduce the same
        # fixed 8D task-sufficient latent without direct terrain perception.
        num_privileged_obs = 5 + 3 + height_map_dim
        num_latent_dims = 8
        c_frame_stack = 5
        num_single_critic_obs = num_observations + 5 + 3
        num_critic_obs = c_frame_stack * num_single_critic_obs
        num_actions = 12


class Go2V4DreamwaqCfg(Go2BenchmarkV4TerrainCfg):
    class env(Go2BenchmarkV4TerrainCfg.env):
        num_observations = 45
        frame_stack = 5
        num_history_obs = num_observations * frame_stack
        num_latent_dims = 16
        num_explicit_dims = 3
        num_decoder_output = num_observations
        c_frame_stack = 5
        num_single_critic_obs = 3 + num_observations + 5
        num_privileged_obs = c_frame_stack * num_single_critic_obs
        num_actions = 12


class Go2V4HIMFixedCfg(Go2BenchmarkV4TerrainCfg):
    class env(Go2BenchmarkV4TerrainCfg.env):
        num_one_step_obs = 45
        frame_stack = 6
        num_observations = frame_stack * num_one_step_obs
        c_frame_stack = 5
        num_single_critic_obs = 3 + num_one_step_obs + 5
        num_privileged_obs = c_frame_stack * num_single_critic_obs
        num_actions = 12


class Go2V4SupersetOracleCfg(Go2BenchmarkV4TerrainCfg):
    class env(Go2BenchmarkV4TerrainCfg.env):
        num_single_obs = 45
        frame_stack = 20
        height_map_dim = 17 * 11
        oracle_height_map = True
        # [20 x noisy proprio, true velocity, true P5, true height map]
        num_observations = num_single_obs * frame_stack + 3 + 5 + height_map_dim
        num_privileged_obs = num_observations
        num_actions = 12


class Go2V4MlpCfgPPO(Go2V3CfgPPO):
    class runner(Go2V3CfgPPO.runner):
        experiment_name = "go2_v4_terrain_curriculum"
        run_name = "v4_mlp" + get_simulator_suffix()
        command_schedule = V4_COMMAND_SCHEDULE


class Go2V4SysIDCfgPPO(Go2V3SysIDCfgPPO):
    class runner(Go2V3SysIDCfgPPO.runner):
        experiment_name = "go2_v4_terrain_curriculum"
        run_name = "v4_sysid" + get_simulator_suffix()
        command_schedule = V4_COMMAND_SCHEDULE


class Go2V4RMACfgPPO(Go2V3RMACfgPPO):
    class runner(Go2V3RMACfgPPO.runner):
        experiment_name = "go2_v4_terrain_curriculum"
        run_name = "v4_rma" + get_simulator_suffix()
        command_schedule = V4_COMMAND_SCHEDULE


class Go2V4DreamwaqCfgPPO(Go2V3DreamwaqCfgPPO):
    class runner(Go2V3DreamwaqCfgPPO.runner):
        experiment_name = "go2_v4_terrain_curriculum"
        run_name = "v4_dreamwaq" + get_simulator_suffix()
        command_schedule = V4_COMMAND_SCHEDULE


class Go2V4HIMFixedCfgPPO(Go2V3HIMFixedCfgPPO):
    class runner(Go2V3HIMFixedCfgPPO.runner):
        experiment_name = "go2_v4_terrain_curriculum"
        run_name = "v4_him_fixed" + get_simulator_suffix()
        command_schedule = V4_COMMAND_SCHEDULE


class Go2V4SupersetOracleCfgPPO(Go2V3SupersetOracleCfgPPO):
    class runner(Go2V3SupersetOracleCfgPPO.runner):
        experiment_name = "go2_v4_terrain_curriculum"
        run_name = "v4_superset_oracle" + get_simulator_suffix()
        command_schedule = V4_COMMAND_SCHEDULE
