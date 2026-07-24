"""CPU-only contract tests for Eval V2's non-simulator semantics."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from legged_gym.scripts.eval.campaign import (
    _planned_cells, _terrain_raw, cmd_aggregate, delay_ms, selection_order,
)
from legged_gym.scripts.eval.indist import tracking_score, tracking_selection_key
from legged_gym.scripts.eval.metrics import MetricAccumulator
from rsl_rl.modules.actor_critic_ee import ActorCriticEE
from rsl_rl.modules.vae import VAE
from rsl_rl.runners.eval_adapter import (
    HistoryObsAdapter, RMAStudentDiagnosticAdapter, load_deploy_components,
)
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class TestCheckpointSelection(unittest.TestCase):
    def test_safe_checkpoint_beats_unsafe(self):
        chosen = selection_order([
            {"iteration": 100, "fall_rate": 0.01, "tracking_score": 0.9},
            {"iteration": 200, "fall_rate": 0.10, "tracking_score": 0.01},
        ])
        self.assertEqual(chosen["iteration"], 100)

    def test_tracking_then_early_iteration_breaks_safe_ties(self):
        chosen = selection_order([
            {"iteration": 200, "fall_rate": 0.01, "tracking_score": 0.4},
            {"iteration": 100, "fall_rate": 0.02, "tracking_score": 0.4},
            {"iteration": 300, "fall_rate": 0.01, "tracking_score": 0.5},
        ])
        self.assertEqual(chosen["iteration"], 100)

    def test_all_unsafe_uses_fall_then_tracking(self):
        chosen = selection_order([
            {"iteration": 100, "fall_rate": 0.3, "tracking_score": 0.1},
            {"iteration": 200, "fall_rate": 0.2, "tracking_score": 0.9},
            {"iteration": 300, "fall_rate": 0.2, "tracking_score": 0.8},
        ])
        self.assertEqual(chosen["iteration"], 300)

    def test_training_selection_key_has_same_lexicographic_rule(self):
        safe = {"fall_rate": 0.01, "tracking_lin_err": 1.0, "tracking_ang_err": 1.0}
        unsafe = {"fall_rate": 0.20, "tracking_lin_err": 0.0, "tracking_ang_err": 0.0}
        self.assertLess(tracking_selection_key(safe, 100, .05),
                        tracking_selection_key(unsafe, 200, .05))
        self.assertAlmostEqual(tracking_score(safe), .5 * (1.0 / 2 ** .5 + 1.0))

    def test_training_selection_key_breaks_ties_by_iteration(self):
        m = {"fall_rate": 0.01, "tracking_lin_err": 0.2, "tracking_ang_err": 0.3}
        self.assertLess(tracking_selection_key(m, 100, .05), tracking_selection_key(m, 200, .05))


class TestV2MetricContract(unittest.TestCase):
    def test_window_fall_and_censored_return(self):
        acc = MetricAccumulator(2, "cpu")
        # env 0 falls once then is auto-reset; env 1 never terminates.  All four
        # rewards must count in return_per_step, regardless of completed episodes.
        for t in range(2):
            acc.update(torch.tensor([2.0, 4.0]), torch.tensor([t == 0, False]),
                       torch.tensor([False, False]), torch.zeros(2), torch.zeros(2))
        got = acc.compute()
        self.assertTrue(torch.equal(got["ever_fell"], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.equal(got["return_per_step"], torch.tensor([2.0, 4.0])))

    def test_delay_step_to_ms(self):
        self.assertEqual(delay_ms(0), 0.0)
        self.assertEqual(delay_ms(6), 120.0)

    def test_deterministic_terrain_map_and_scale(self):
        raw_a = _terrain_raw(0.05, 41001, 0.2)
        raw_b = _terrain_raw(0.05, 41001, 0.2)
        self.assertEqual(raw_a.shape, (60, 60))
        self.assertTrue((raw_a == raw_b).all())
        self.assertEqual(int(abs(raw_a).max()), 20)  # 5cm / 2.5mm


class TestAdaptiveInferenceContract(unittest.TestCase):
    def test_resume_iteration_selects_correct_command_stage(self):
        schedule = [
            {"start_iteration": 0, "lin_vel_x": [-0.5, 0.5]},
            {"start_iteration": 500, "lin_vel_x": [-1.0, 1.0]},
        ]
        self.assertEqual(OnPolicyRunner.command_stage_for_iter(schedule, 499), schedule[0])
        self.assertEqual(OnPolicyRunner.command_stage_for_iter(schedule, 500), schedule[1])
        self.assertEqual(OnPolicyRunner.command_stage_for_iter(schedule, 900), schedule[1])

    def test_checkpoint_iteration_is_completed_updates_and_resumes_at_next_update(self):
        self.assertEqual(OnPolicyRunner.completed_iteration(499), 500)

        def runner():
            obj = OnPolicyRunner.__new__(OnPolicyRunner)
            actor_critic = torch.nn.Module()
            actor_critic.actor = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adam(actor_critic.parameters(), lr=1e-3)
            obj.alg = mock.Mock(actor_critic=actor_critic, optimizer=optimizer)
            obj.current_learning_iteration = 0
            obj.best_eval_score = float("inf")
            obj.best_tracking_key = None
            obj.training_seed = 7
            obj._active_schedule_start = 0
            obj._active_schedule_range = [-0.5, 0.5]
            obj._aux_optimizers = lambda: {}
            return obj

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "model_500.pt"
            source = runner()
            # Loop index 499 has just completed update number 500.
            source.save(str(checkpoint), iteration=500)
            payload = torch.load(checkpoint, weights_only=False)
            self.assertEqual(payload["iter"], 500)
            self.assertEqual(payload["iteration_semantics"], "completed_updates_v2")

            resumed = runner()
            resumed.load(str(checkpoint))
            self.assertEqual(resumed.current_learning_iteration, 500)
            self.assertEqual(list(range(resumed.current_learning_iteration,
                                        resumed.current_learning_iteration + 2)), [500, 501])
            schedule = [
                {"start_iteration": 0, "lin_vel_x": [-0.5, 0.5]},
                {"start_iteration": 500, "lin_vel_x": [-1.0, 1.0]},
            ]
            self.assertEqual(OnPolicyRunner.command_stage_for_iter(
                schedule, resumed.current_learning_iteration), schedule[1])

            # The marker is additive: existing checkpoints without it remain
            # loadable under their historically stored ``iter`` value.
            payload.pop("iteration_semantics")
            legacy = Path(tmp) / "legacy.pt"
            torch.save(payload, legacy)
            legacy_resumed = runner()
            legacy_resumed.load(str(legacy))
            self.assertEqual(legacy_resumed.current_learning_iteration, 500)

    def test_resume_carries_best_tracking_artifact_to_new_run(self):
        def runner(log_dir):
            obj = OnPolicyRunner.__new__(OnPolicyRunner)
            actor_critic = torch.nn.Module()
            actor_critic.actor = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adam(actor_critic.parameters(), lr=1e-3)
            obj.alg = mock.Mock(actor_critic=actor_critic, optimizer=optimizer)
            obj.current_learning_iteration = 10
            obj.best_eval_score = 0.25
            obj.best_tracking_key = (0.0, 0.25, 10.0)
            obj.training_seed = 7
            obj._active_schedule_start = 0
            obj._active_schedule_range = [-0.5, 0.5]
            obj._aux_optimizers = lambda: {}
            obj.log_dir = str(log_dir)
            return obj

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_run, new_run = root / "old", root / "new"
            old_run.mkdir()
            source = runner(old_run)
            source.save(str(old_run / "best_tracking.pt"), iteration=10)
            source.save(str(old_run / "model_10.pt"), iteration=10)

            resumed = runner(new_run)
            resumed.best_tracking_key = None
            resumed.load(str(old_run / "model_10.pt"))

            carried = new_run / "best_tracking.pt"
            self.assertTrue(carried.is_file())
            self.assertEqual(carried.read_bytes(), (old_run / "best_tracking.pt").read_bytes())
            self.assertEqual(resumed.best_tracking_key, (0.0, 0.25, 10.0))

    def test_run_eval_selects_best_spnte_independently_of_best_tracking(self):
        """best_spnte.pt tracks the min spnte_lin iteration; best_tracking.pt
        keeps selecting on its own lexicographic tracking/fall rule. The two
        checkpoints must be free to diverge to different iterations."""

        def runner(log_dir):
            obj = OnPolicyRunner.__new__(OnPolicyRunner)
            actor_critic = torch.nn.Module()
            actor_critic.actor = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adam(actor_critic.parameters(), lr=1e-3)
            obj.alg = mock.Mock(actor_critic=actor_critic, optimizer=optimizer)
            obj.current_learning_iteration = 0
            obj.best_eval_score = float("inf")
            obj.best_tracking_key = None
            obj.best_spnte_score = float("inf")
            obj.best_spnte_key = None
            obj.training_seed = 7
            obj._active_schedule_start = 0
            obj._active_schedule_range = [-0.5, 0.5]
            obj._aux_optimizers = lambda: {}
            obj.log_dir = str(log_dir)
            obj.writer = None
            # None (not a Mock): `save()` probes `env.cfg.env.ued_enabled` and
            # a bare Mock auto-vivifies that as a truthy Mock, wrongly routing
            # into the UED-only save/resume branch this test isn't exercising.
            obj.env = None
            obj.eval_steps, obj.eval_warmup, obj.eval_seed = 1100, 50, 12345
            obj.eval_fall_guard = 0.25
            return obj

        # it=100: decent tracking, mediocre spnte. it=200: worse tracking but
        # the best spnte. it=300: the best tracking but the worst spnte.
        metrics_by_it = {
            100: {"fall_rate": 0.0, "tracking_lin_err": 0.5, "tracking_ang_err": 0.5,
                  "mean_return": 1.0, "mean_ep_len": 500.0, "spnte_lin": 0.6, "spnte_yaw": 0.5},
            200: {"fall_rate": 0.0, "tracking_lin_err": 0.9, "tracking_ang_err": 0.9,
                  "mean_return": 1.0, "mean_ep_len": 500.0, "spnte_lin": 0.2, "spnte_yaw": 0.5},
            300: {"fall_rate": 0.0, "tracking_lin_err": 0.1, "tracking_ang_err": 0.1,
                  "mean_return": 1.0, "mean_ep_len": 500.0, "spnte_lin": 0.9, "spnte_yaw": 0.9},
        }

        with tempfile.TemporaryDirectory() as tmp:
            obj = runner(tmp)
            with mock.patch(
                "legged_gym.scripts.eval.indist.run_indist_eval",
                side_effect=[metrics_by_it[100], metrics_by_it[200], metrics_by_it[300]],
            ):
                for it in (100, 200, 300):
                    obj._run_eval(it, adapter=mock.Mock())

            best_tracking = torch.load(Path(tmp) / "best_tracking.pt", weights_only=False)
            best_spnte = torch.load(Path(tmp) / "best_spnte.pt", weights_only=False)

            # Best tracking score belongs to it=300 (lowest tracking error,
            # all iterations equally "safe" at fall_rate=0).
            self.assertEqual(best_tracking["infos"]["selected_iteration"], 300)
            self.assertEqual(best_tracking["iter"], 300)

            # Best SPNTE belongs to it=200 (lowest spnte_lin), independently.
            self.assertEqual(best_spnte["infos"]["selected_iteration"], 200)
            self.assertEqual(best_spnte["infos"]["selection_metric"], "spnte_v1")
            self.assertAlmostEqual(best_spnte["infos"]["spnte_lin"], 0.2)
            self.assertEqual(best_spnte["iter"], 200)
            self.assertEqual(obj.best_spnte_key, (0.2, 200.0))

    def test_resume_carries_best_spnte_artifact_to_new_run(self):
        def runner(log_dir):
            obj = OnPolicyRunner.__new__(OnPolicyRunner)
            actor_critic = torch.nn.Module()
            actor_critic.actor = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adam(actor_critic.parameters(), lr=1e-3)
            obj.alg = mock.Mock(actor_critic=actor_critic, optimizer=optimizer)
            obj.current_learning_iteration = 10
            obj.best_eval_score = float("inf")
            # No best_tracking selection in this run -- isolates the
            # best_spnte carry-forward from the (separately tested)
            # best_tracking carry-forward logic.
            obj.best_tracking_key = None
            obj.best_spnte_score = 0.2
            obj.best_spnte_key = (0.2, 10.0)
            obj.training_seed = 7
            obj._active_schedule_start = 0
            obj._active_schedule_range = [-0.5, 0.5]
            obj._aux_optimizers = lambda: {}
            obj.log_dir = str(log_dir)
            return obj

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_run, new_run = root / "old", root / "new"
            old_run.mkdir()
            source = runner(old_run)
            source.save(str(old_run / "best_spnte.pt"), iteration=10)
            source.save(str(old_run / "model_10.pt"), iteration=10)

            resumed = runner(new_run)
            resumed.best_tracking_key = None
            resumed.best_spnte_key = None
            resumed.load(str(old_run / "model_10.pt"))

            carried = new_run / "best_spnte.pt"
            self.assertTrue(carried.is_file())
            self.assertEqual(carried.read_bytes(), (old_run / "best_spnte.pt").read_bytes())
            self.assertEqual(resumed.best_spnte_key, (0.2, 10.0))

    def test_best_tracking_resume_unaffected_by_missing_spnte_state(self):
        """Checkpoints saved before this feature existed carry neither
        best_spnte_key nor a best_spnte.pt sibling; resuming from one must
        behave exactly as it did before (best_tracking.pt logic untouched)."""

        def runner(log_dir):
            obj = OnPolicyRunner.__new__(OnPolicyRunner)
            actor_critic = torch.nn.Module()
            actor_critic.actor = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adam(actor_critic.parameters(), lr=1e-3)
            obj.alg = mock.Mock(actor_critic=actor_critic, optimizer=optimizer)
            obj.current_learning_iteration = 10
            obj.best_eval_score = 0.25
            obj.best_tracking_key = (0.0, 0.25, 10.0)
            obj.training_seed = 7
            obj._active_schedule_start = 0
            obj._active_schedule_range = [-0.5, 0.5]
            obj._aux_optimizers = lambda: {}
            obj.log_dir = str(log_dir)
            return obj

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_run, new_run = root / "old", root / "new"
            old_run.mkdir()
            source = runner(old_run)  # note: no best_spnte_score/key set at all
            source.save(str(old_run / "best_tracking.pt"), iteration=10)
            source.save(str(old_run / "model_10.pt"), iteration=10)

            resumed = runner(new_run)
            resumed.best_tracking_key = None
            resumed.load(str(old_run / "model_10.pt"))

            carried = new_run / "best_tracking.pt"
            self.assertTrue(carried.is_file())
            self.assertEqual(carried.read_bytes(), (old_run / "best_tracking.pt").read_bytes())
            self.assertEqual(resumed.best_tracking_key, (0.0, 0.25, 10.0))
            self.assertIsNone(resumed.best_spnte_key)
            self.assertFalse((new_run / "best_spnte.pt").exists())

    def test_resume_fails_loudly_when_persisted_best_artifact_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_run, new_run = root / "old", root / "new"
            old_run.mkdir()
            actor_critic = torch.nn.Module()
            actor_critic.actor = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adam(actor_critic.parameters(), lr=1e-3)

            def runner(log_dir):
                obj = OnPolicyRunner.__new__(OnPolicyRunner)
                obj.alg = mock.Mock(actor_critic=actor_critic, optimizer=optimizer)
                obj.current_learning_iteration = 0
                obj.best_eval_score = 0.25
                obj.best_tracking_key = (0.0, 0.25, 10.0)
                obj.training_seed = 7
                obj._active_schedule_start = 0
                obj._active_schedule_range = [-0.5, 0.5]
                obj._aux_optimizers = lambda: {}
                obj.log_dir = str(log_dir)
                return obj

            checkpoint = old_run / "model_10.pt"
            runner(old_run).save(str(checkpoint), iteration=10)
            with self.assertRaisesRegex(FileNotFoundError, "selected artifact is missing"):
                runner(new_run).load(str(checkpoint))

    def test_dreamwaq_mean_inference_preserves_rng_and_action(self):
        vae = VAE(
            num_history_input=6, num_latent_dims=2, num_explicit_dims=1,
            num_decoder_output=3, encoder_hidden_dims=[4], decoder_hidden_dims=[4],
        )
        history = torch.arange(12, dtype=torch.float).reshape(2, 6)
        torch.manual_seed(123)
        before = torch.get_rng_state().clone()
        action_input_a = vae.inference(history)
        after_a = torch.get_rng_state().clone()
        action_input_b = vae.inference(history)
        after_b = torch.get_rng_state().clone()
        self.assertTrue(torch.equal(action_input_a, action_input_b))
        self.assertTrue(torch.equal(before, after_a))
        self.assertTrue(torch.equal(before, after_b))

    def test_rma_gap_is_accumulated_over_paired_rollout_states(self):
        class ActorCritic(torch.nn.Module):
            history_encoder_type = "MLP"

            def privilege_encoder(self, privileged):
                return privileged

            def history_encoder(self, history):
                return history

            def act_teacher(self, obs, privileged):
                return obs + privileged[:, :1]

            def act_student(self, obs, history):
                return obs + history[:, :1]

        adapter = RMAStudentDiagnosticAdapter(ActorCritic(), torch.device("cpu"))
        adapter.begin_measurement()
        state_a = (torch.zeros(1, 1), torch.tensor([[2., 0.]]), torch.tensor([[0., 0.]]))
        state_b = (torch.zeros(1, 1), torch.tensor([[0., 0.]]), torch.tensor([[0., 0.]]))
        adapter.act(state_a)
        adapter.act(state_b)
        gap = adapter.diagnostics()
        # State A contributes latent squared errors [4,0] and action error 2;
        # state B contributes zeros. These are trajectory means, not state B only.
        self.assertAlmostEqual(gap["latent_mse"], 1.0)
        self.assertAlmostEqual(gap["action_mae"], 1.0)

    def test_campaign_accepts_honest_model_local_smoke_seeds(self):
        cfg = {
            "training_seeds": [], "axes": {},
            "models": [
                {"label": "RMA", "training_seeds": [102]},
                {"label": "SysID", "training_seeds": [103]},
            ],
        }
        cells = _planned_cells(cfg, "id")
        self.assertEqual([(m["label"], seed) for _, m, seed, _ in cells],
                         [("RMA", 102), ("SysID", 103)])

    def test_strict_loader_requires_every_deploy_component(self):
        module = torch.nn.Module()
        module.actor = torch.nn.Linear(3, 2)
        module.estimator = torch.nn.Linear(4, 1)
        state = {f"actor.{k}": v.clone() for k, v in module.actor.state_dict().items()}
        state.update({f"estimator.{k}": v.clone() for k, v in module.estimator.state_dict().items()})
        load_deploy_components(module, state, ("actor.", "estimator."), "synthetic.pt")
        with self.assertRaisesRegex(RuntimeError, "history_encoder"):
            load_deploy_components(module, state, ("history_encoder.",), "synthetic.pt")

    def test_history_adapter_hides_method_specific_env_tuple(self):
        class Env:
            def reset(self):
                return None

            def get_observations(self):
                return (torch.ones(2, 3), torch.zeros(2, 1), torch.full((2, 6), 2.0), None)

            def step(self, actions):
                return (*self.get_observations(), torch.ones(2), torch.zeros(2), {})

        seen = []
        adapter = HistoryObsAdapter(lambda obs, history: seen.append((obs, history)) or torch.zeros(2, 1), torch.device("cpu"))
        state = adapter.reset(Env())
        action = adapter.act(state)
        state, rewards, dones = adapter.step(Env(), action)
        self.assertEqual(tuple(seen[0][0].shape), (2, 3))
        self.assertEqual(tuple(seen[0][1].shape), (2, 6))
        self.assertTrue(torch.equal(rewards, torch.ones(2)))
        self.assertTrue(torch.equal(dones, torch.zeros(2)))

    def test_sysid_actor_receives_current_frame_plus_estimate(self):
        model = ActorCriticEE(
            num_critic_obs=5, num_actions=2, num_estimator_features=12,
            num_estimator_labels=2, num_actor_obs=3,
            actor_hidden_dims=[4], critic_hidden_dims=[4], estimator_hidden_dims=[4],
        )
        captured = []
        hook = model.actor.register_forward_pre_hook(lambda _m, args: captured.append(args[0].detach().clone()))
        features = torch.arange(24, dtype=torch.float).reshape(2, 12)
        model.act_inference(features)
        hook.remove()
        self.assertEqual(tuple(model.actor[0].weight.shape), (4, 5))
        self.assertTrue(torch.equal(captured[0][:, :3], features[:, -3:]))


class TestSeedAggregationContract(unittest.TestCase):
    def test_checkpoint_provenance_does_not_split_training_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            for model, task, offset in (("MLP", "go2_bench_mlp", 0.0), ("P5", "go2_bench_oracle", -0.1)):
                for seed in (1, 2, 3):
                    np.savez(
                        raw / f"{model}_{seed}.npz",
                        campaign="test", suite="id", scenario="id", model=model,
                        task=task, training_seed=seed, checkpoint_iter=100 * seed,
                        checkpoint_sha256=f"sha-{model}-{seed}", eval_seed=12345,
                        tracking_lin=np.array([1.0 + offset]),
                        fall_rate=np.array([0.0]), return_per_step=np.array([2.0]),
                    )
            cfg = {"campaign": "test", "artifact_root": str(root)}
            with mock.patch("legged_gym.scripts.eval.campaign._write_campaign_state"):
                cmd_aggregate(cfg)
            with (root / "tables" / "summary.csv").open() as f:
                summary = list(csv.DictReader(f))
            self.assertEqual(len(summary), 6)  # 2 models x 3 metrics
            self.assertEqual({row["n_training_seeds"] for row in summary}, {"3"})
            with (root / "tables" / "consistency.csv").open() as f:
                consistency = list(csv.DictReader(f))
            self.assertEqual(len(consistency), 3)
            self.assertEqual({row["n_training_seeds"] for row in consistency}, {"3"})

    def test_validation_checkpoints_remain_separate_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            for model, task in (("MLP", "go2_bench_mlp"), ("P5", "go2_bench_oracle")):
                for seed in (1, 2, 3):
                    for iteration in (200, 400):
                        np.savez(
                            raw / f"validation_{model}_{seed}_{iteration}.npz",
                            campaign="test", suite="validation", scenario="checkpoint_selection",
                            model=model, task=task, training_seed=seed,
                            checkpoint_iter=iteration,
                            checkpoint_sha256=f"sha-{model}-{seed}-{iteration}", eval_seed=12345,
                            tracking_lin=np.array([iteration / 1000.0]),
                            fall_rate=np.array([0.0]), return_per_step=np.array([2.0]),
                        )
            cfg = {"campaign": "test", "artifact_root": str(root)}
            with mock.patch("legged_gym.scripts.eval.campaign._write_campaign_state"):
                cmd_aggregate(cfg)
            with (root / "tables" / "summary.csv").open() as f:
                summary = list(csv.DictReader(f))
            self.assertEqual(len(summary), 12)  # 2 models x 2 checkpoints x 3 metrics
            self.assertEqual({row["n_training_seeds"] for row in summary}, {"3"})
            self.assertEqual({row["checkpoint_iter"] for row in summary}, {"200", "400"})
            with (root / "tables" / "consistency.csv").open() as f:
                consistency = list(csv.DictReader(f))
            self.assertEqual(len(consistency), 6)  # 2 checkpoints x 3 metrics
            self.assertEqual({row["n_training_seeds"] for row in consistency}, {"3"})


if __name__ == "__main__":
    unittest.main()
