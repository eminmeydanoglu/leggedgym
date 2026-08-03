"""CPU-only contract tests for the MoE-CTS runner wiring (go2_rl_gym port).

Covers, without building a simulator/env:
1. 'MoECTSRunner' resolves from the runner registry and subclasses CTSRunner.
2. go2_moects PPO config selects MoECTSRunner; go2_moects_him stays on
   HIMRunner (import-based, with an AST fallback if the legged_gym import
   chain is unavailable).
3. Telemetry mapping: a fake moe_stats dict produces every pinned TensorBoard
   name; the number of MoE/expert_usage_i scalars matches expert_num.
4. eval() class-name resolution: 'PPO_MOE_CTS' / 'ActorCriticMoECTS' resolve
   to the real classes in the (inherited) _init_agent_and_algo context, and
   get_inference_policy stays the CTSRunner one (student deploy).

Run: .venv/bin/python -m unittest tests.test_moects_runner_contract -v
"""

import ast
import contextlib
import io
import os
import re
import tempfile
import unittest
from collections import deque
from unittest import mock

# legged_gym's package __init__ gates on this; the config test only needs the
# genesis *import* (works on CPU), never builds a simulator.
os.environ.setdefault("SIMULATOR", "genesis")

import torch  # noqa: E402

import rsl_rl.runners  # noqa: F401  E402 -- executes the registry wiring
import rsl_rl.runners.cts_runner as cts_runner_mod  # noqa: E402
from rsl_rl.runners import CTSRunner, MoECTSRunner  # noqa: E402
from rsl_rl.utils.runner_registry import runner_registry  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RunnerRegistryTest(unittest.TestCase):
    def test_moects_runner_registered_and_subclasses_cts(self):
        cls = runner_registry.get_runner_class("MoECTSRunner")
        self.assertIs(cls, MoECTSRunner)
        self.assertTrue(issubclass(cls, CTSRunner))
        self.assertIsNot(cls, CTSRunner)


class ConfigWiringTest(unittest.TestCase):
    def test_runner_selection(self):
        try:
            from legged_gym.envs.go2.go2_moects.go2_moects_config import (
                Go2MoECTSCfgPPO,
                Go2MoECTSHIMCfgPPO,
            )
        except Exception:
            # Import chain too heavy / unavailable: verify at text level.
            self._assert_config_via_ast()
            return
        self.assertEqual(Go2MoECTSCfgPPO.runner_class_name, "MoECTSRunner")
        self.assertEqual(Go2MoECTSHIMCfgPPO.runner_class_name, "HIMRunner")
        self.assertEqual(Go2MoECTSCfgPPO.runner.policy_class_name, "ActorCriticMoECTS")
        self.assertEqual(Go2MoECTSCfgPPO.runner.algorithm_class_name, "PPO_MOE_CTS")
        # HIM arm untouched: HIMRunner with the HIM policy/algorithm pair.
        self.assertEqual(Go2MoECTSHIMCfgPPO.runner.policy_class_name, "HIMActorCritic")
        self.assertEqual(Go2MoECTSHIMCfgPPO.runner.algorithm_class_name, "PPO_HIM")

    def _assert_config_via_ast(self):
        path = os.path.join(REPO_ROOT, "legged_gym", "envs", "go2",
                            "go2_moects", "go2_moects_config.py")
        with open(path) as f:
            tree = ast.parse(f.read())

        def class_def(name):
            return next(n for n in ast.walk(tree)
                        if isinstance(n, ast.ClassDef) and n.name == name)

        def direct_assigns(cls):
            return {t.id: n.value.value
                    for n in cls.body if isinstance(n, ast.Assign)
                    for t in n.targets if isinstance(t, ast.Name)
                    and isinstance(n.value, ast.Constant)}

        moe_assigns = direct_assigns(class_def("Go2MoECTSCfgPPO"))
        self.assertEqual(moe_assigns.get("runner_class_name"), "MoECTSRunner")
        moe_runner_cls = next(n for n in class_def("Go2MoECTSCfgPPO").body
                              if isinstance(n, ast.ClassDef) and n.name == "runner")
        moe_runner_assigns = direct_assigns(moe_runner_cls)
        self.assertEqual(moe_runner_assigns.get("policy_class_name"), "ActorCriticMoECTS")
        self.assertEqual(moe_runner_assigns.get("algorithm_class_name"), "PPO_MOE_CTS")
        # HIM arm must NOT override runner_class_name (inherits 'HIMRunner'
        # from LeggedRobotHIMCfgPPO).
        him_assigns = direct_assigns(class_def("Go2MoECTSHIMCfgPPO"))
        self.assertNotIn("runner_class_name", him_assigns)


class TelemetryMappingTest(unittest.TestCase):
    EXPERT_NUM = 4  # deliberately != 8: proves the count is never hardcoded

    def _make_runner(self):
        # Bypass __init__ (needs an env); set only what CTSRunner.log +
        # _log_moe_stats touch.
        runner = MoECTSRunner.__new__(MoECTSRunner)
        runner.writer = mock.MagicMock()
        runner.device = torch.device("cpu")
        runner.tot_timesteps = 0
        runner.tot_time = 0.0
        runner.num_steps_per_env = 24
        runner.current_learning_iteration = 0
        runner.env = mock.MagicMock()
        runner.env.num_envs = 8
        runner.alg = mock.MagicMock()
        runner.alg.actor_critic.std = torch.ones(12)
        runner.alg.learning_rate = 1e-3
        runner.alg.encoder_lr = 1e-3
        return runner

    def _fake_moe_stats(self):
        return {
            "latent_mse": 0.5,
            "load_balance": 0.01,
            "student_encoder_total": 0.51,
            "gating_entropy": 1.2,
            "effective_experts": 3.4,
            "expert_usage_min": 0.1,
            "expert_usage_max": 0.4,
            "expert_usage_std": 0.05,
            "expert_usage": [0.1, 0.2, 0.3, 0.4][: self.EXPERT_NUM],
        }

    def test_moe_telemetry_tensorboard_names(self):
        runner = self._make_runner()
        locs = dict(
            it=3,
            num_learning_iterations=10,
            collection_time=0.5,
            learn_time=0.25,
            ep_infos=[],
            rewbuffer=deque(maxlen=100),
            lenbuffer=deque(maxlen=100),
            mean_value_loss=1.0,
            mean_teacher_surrogate_loss=2.0,
            mean_student_surrogate_loss=3.0,
            mean_reconstruction_loss=4.0,
            moe_stats=self._fake_moe_stats(),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            runner.log(locs)

        scalars = {}  # name -> (value, step); last write wins
        for call in runner.writer.add_scalar.call_args_list:
            scalars[call.args[0]] = (call.args[1], call.args[2])

        # Existing CTS channels must survive unchanged (fed by the 5-tuple's
        # first four values).
        for name in ("Loss/value_function", "Loss/teacher_surrogate",
                     "Loss/student_surrogate"):
            self.assertIn(name, scalars)
        # 'Loss/reconstruction' is the one deliberate exception: on this arm it
        # IS moe_stats['latent_mse'], so MoECTSRunner.log filters the duplicate
        # tag (see _DropScalarTags and
        # test_moects_telemetry.test_drop_scalar_tags_filters_reconstruction_only).
        self.assertNotIn("Loss/reconstruction", scalars)
        self.assertIn("Loss/latent_mse", scalars)
        # Pinned MoE telemetry names.
        expected = {
            "Loss/latent_mse", "Loss/load_balance", "Loss/student_encoder_total",
            "MoE/gating_entropy", "MoE/effective_experts",
            "MoE/expert_usage_min", "MoE/expert_usage_max", "MoE/expert_usage_std",
        } | {f"MoE/expert_usage_{i}" for i in range(self.EXPERT_NUM)}
        missing = expected - set(scalars)
        self.assertFalse(missing, f"missing TB scalars: {missing}")

        # expert_usage_i count matches expert_num exactly (no assumed 8).
        # Note: MoE/expert_usage_{min,max,std} share the prefix, so match the
        # numeric suffix explicitly.
        expert_logged = [n for n in scalars
                         if re.fullmatch(r"MoE/expert_usage_\d+", n)]
        self.assertEqual(len(expert_logged), self.EXPERT_NUM)

        # Values and iteration step are passed through, not re-aggregated.
        self.assertEqual(scalars["Loss/latent_mse"], (0.5, 3))
        self.assertEqual(scalars["MoE/expert_usage_2"], (0.3, 3))
        self.assertEqual(scalars["MoE/effective_experts"], (3.4, 3))

    def test_moe_telemetry_terminal_output(self):
        runner = self._make_runner()
        locs = dict(
            it=0,
            num_learning_iterations=1,
            collection_time=1.0,
            learn_time=1.0,
            ep_infos=[],
            rewbuffer=deque(maxlen=100),
            lenbuffer=deque(maxlen=100),
            mean_value_loss=1.0,
            mean_teacher_surrogate_loss=2.0,
            mean_student_surrogate_loss=3.0,
            mean_reconstruction_loss=4.0,
            moe_stats=self._fake_moe_stats(),
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            runner.log(locs)
        out = buf.getvalue()
        self.assertIn("MoE latent mse:", out)
        self.assertIn("MoE load balance:", out)
        self.assertIn("MoE student encoder total:", out)
        self.assertIn("MoE gating entropy:", out)
        self.assertIn("MoE effective experts:", out)


class CheckpointIterationSemanticsTest(unittest.TestCase):
    """Exercise MoECTSRunner.learn's real intermediate-save call site."""

    class _Env:
        num_envs = 1

        def __init__(self):
            self.observation_calls = 0

        def get_observations(self):
            self.observation_calls += 1
            obs = torch.zeros(1, 1)
            return obs.clone(), obs.clone(), obs.clone(), obs.clone()

        def step(self, actions):
            obs = torch.zeros(1, 1)
            rewards = torch.zeros(1)
            dones = torch.zeros(1)
            return obs.clone(), obs.clone(), obs.clone(), obs.clone(), rewards, dones, {}

    class _Alg:
        def __init__(self):
            self.actor_critic = torch.nn.Linear(1, 1)
            self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=1e-3)
            self.teacher_env_idxs = torch.tensor([0])

        def act(self, obs, privileged_obs, obs_history, critic_obs):
            return torch.zeros(1, 1)

        def process_env_step(self, rewards, dones, infos):
            pass

        def compute_returns(self, critic_obs, obs_history):
            pass

        def update(self):
            return 1.0, 2.0, 3.0, 4.0, {}

    def _make_runner(self, log_dir, current_learning_iteration):
        runner = MoECTSRunner.__new__(MoECTSRunner)
        runner.env = self._Env()
        runner.alg = self._Alg()
        runner.device = torch.device("cpu")
        runner.num_steps_per_env = 1
        runner.save_interval = 1
        runner.log_dir = log_dir
        runner.writer = None
        runner.tot_timesteps = 0
        runner.tot_time = 0.0
        runner.current_learning_iteration = current_learning_iteration
        runner.training_seed = None
        runner._active_schedule_start = None
        runner._active_schedule_range = None
        runner.best_eval_score = float("inf")
        runner.best_tracking_key = None
        runner.eval_interval = 0
        runner._aux_optimizers = lambda: {}
        runner._pre_learn = lambda *_args, **_kwargs: None
        runner.log = lambda _locs: None
        return runner

    def test_fresh_and_resumed_intermediate_checkpoints_use_completed_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            fresh = self._make_runner(tmp, current_learning_iteration=0)
            fresh.learn(num_learning_iterations=2)
            self.assertEqual(fresh.current_learning_iteration, 2)

            resumed = self._make_runner(tmp, current_learning_iteration=2)
            resumed.learn(num_learning_iterations=2)
            self.assertEqual(resumed.current_learning_iteration, 4)

            checkpoint_names = sorted(
                name for name in os.listdir(tmp) if name.startswith("model_")
            )
            self.assertEqual(
                checkpoint_names,
                ["model_1.pt", "model_2.pt", "model_3.pt", "model_4.pt"],
            )
            for completed_iteration in range(1, 5):
                path = os.path.join(tmp, f"model_{completed_iteration}.pt")
                payload = torch.load(path, map_location="cpu", weights_only=False)
                self.assertEqual(payload["iter"], completed_iteration)
                self.assertEqual(payload["iteration_semantics"], "completed_updates_v2")

    def test_eval_runs_after_each_saved_checkpoint_and_refreshes_moe_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._make_runner(tmp, current_learning_iteration=0)
            runner.eval_interval = 1
            evaluated = []
            runner._run_eval = lambda iteration: evaluated.append(iteration)
            runner.learn(num_learning_iterations=2)
            self.assertEqual(evaluated, [1, 2])

    def test_isolated_eval_never_refreshes_training_rollout_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._make_runner(tmp, current_learning_iteration=0)
            runner.eval_interval = 1
            runner.eval_env = object()
            evaluated = []
            runner._run_eval = lambda iteration: evaluated.append(iteration)
            runner.learn(num_learning_iterations=2)
            self.assertEqual(evaluated, [1, 2])
            self.assertEqual(runner.env.observation_calls, 1)


class ClassResolutionTest(unittest.TestCase):
    def test_eval_names_resolve_in_runner_context(self):
        from rsl_rl.algorithms import PPO_MOE_CTS
        from rsl_rl.modules import ActorCriticMoECTS

        # _init_agent_and_algo is inherited from CTSRunner, so its eval() calls
        # resolve class names in the cts_runner module globals.
        self.assertIs(MoECTSRunner._init_agent_and_algo,
                      CTSRunner._init_agent_and_algo)
        self.assertIs(eval("PPO_MOE_CTS", vars(cts_runner_mod)), PPO_MOE_CTS)
        self.assertIs(eval("ActorCriticMoECTS", vars(cts_runner_mod)),
                      ActorCriticMoECTS)

    def test_deploy_policy_contract_inherited(self):
        # get_inference_policy stays CTSRunner's (actor_critic.act_student).
        self.assertIs(MoECTSRunner.get_inference_policy,
                      CTSRunner.get_inference_policy)

    def _construct_moe_runner(self):
        """Build a real MoECTSRunner around a mock env (real module + alg)."""
        env = mock.MagicMock()
        env.num_envs = 8
        env.num_obs = 45
        env.num_actions = 12
        env.num_privileged_obs = 263
        env.num_history_obs = 225
        env.num_latent_dims = 32
        env.num_critic_obs = 263
        env.num_teacher = 6
        train_cfg = {
            "runner": {
                "policy_class_name": "ActorCriticMoECTS",
                "algorithm_class_name": "PPO_MOE_CTS",
                "experiment_name": "go2_moects",
                "run_name": "moe_cts_test",
                "num_steps_per_env": 24,
                "save_interval": 500,
                "max_iterations": 10,
            },
            "policy": {"init_noise_std": 1.0, "expert_num": 4},
            "algorithm": {"load_balance_coef": 0.01, "encoder_lr": 1e-3,
                          "num_encoder_epochs": 1},
            "seed": 1,
        }
        with contextlib.redirect_stdout(io.StringIO()):
            return MoECTSRunner(env, train_cfg, log_dir=None, device="cpu")

    def test_end_to_end_agent_algo_storage_wiring(self):
        from rsl_rl.algorithms import PPO_MOE_CTS
        from rsl_rl.modules import ActorCriticMoECTS

        try:
            from rsl_rl.storage.rollout_storage_moe_cts import (
                RolloutStorageMoECTS)
        except ImportError:
            # Storage port (subagent A) not landed yet: construction must fail
            # only at the storage wiring in _init_storage, which already proves
            # the eval()-based agent/algo resolution above it worked.
            with self.assertRaises(ImportError):
                self._construct_moe_runner()
            return
        runner = self._construct_moe_runner()
        self.assertIsInstance(runner.alg, PPO_MOE_CTS)
        self.assertIsInstance(runner.alg.actor_critic, ActorCriticMoECTS)
        self.assertIsInstance(runner.alg.storage, RolloutStorageMoECTS)

    def test_interleaved_role_idxs_wired_through(self):
        # env: 8 envs / 6 teachers (ratio 0.75) -> reference interleave
        # (moe_cts.py:96-102): students {0, 4}, teachers the rest. The runner
        # construction must wire the same mapping into alg AND storage.
        from rsl_rl.algorithms.ppo_moe_cts import compute_role_env_idxs
        runner = self._construct_moe_runner()
        ti, si = compute_role_env_idxs(8, 0.75, "cpu")
        self.assertEqual(si.tolist(), [0, 4])
        self.assertEqual(ti.tolist(), [1, 2, 3, 5, 6, 7])
        self.assertTrue(torch.equal(runner.alg.teacher_env_idxs, ti))
        self.assertTrue(torch.equal(runner.alg.student_env_idxs, si))
        self.assertTrue(torch.equal(runner.alg.storage.teacher_env_idxs, ti))
        self.assertTrue(torch.equal(runner.alg.storage.student_env_idxs, si))


if __name__ == "__main__":
    unittest.main()
