from legged_gym.envs.base.common_cfgs import Go2BenchmarkV3CommonCfg, get_simulator_suffix
from legged_gym.envs.go2.go2_v3_mlp.go2_v3_mlp_config import Go2V3CfgPPO


class Go2V3SupersetOracleCfg(Go2BenchmarkV3CommonCfg):
    class env(Go2BenchmarkV3CommonCfg.env):
        num_single_obs = 45
        frame_stack = 20
        # [20 x proprio, true base velocity, true P5]
        num_observations = num_single_obs * frame_stack + 3 + 5
        # The asymmetric critic has the same information but a clean history.
        num_privileged_obs = num_observations
        num_actions = 12


class Go2V3SupersetOracleCfgPPO(Go2V3CfgPPO):
    class policy(Go2V3CfgPPO.policy):
        # The ceiling receives a deliberately much larger input than a method;
        # give PPO enough capacity to optimize it rather than penalizing the
        # upper bound for an avoidable bottleneck.
        actor_hidden_dims = [1024, 512, 256]
        critic_hidden_dims = [1024, 512, 256]

    class runner(Go2V3CfgPPO.runner):
        run_name = "v3_superset_oracle" + get_simulator_suffix()
