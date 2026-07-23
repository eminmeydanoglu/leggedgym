"""Focused contracts for the Topic 03 Genesis UED boundary."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from legged_gym.utils.terrain import Terrain, ued_training_builder_parameters
from legged_gym.utils.ued import EpisodeOutcomeBatch, TaskAssignmentBatch, TaskSpace, TaskSpec, UniformEpisodeCurriculum
from legged_gym.utils.ued.genesis_adapter import GenesisUEDAdapter


class _Simulator:
    def __init__(self, count: int):
        self.custom_origins = True
        origins = torch.zeros(4, 6, 3)
        for level in range(4):
            for terrain_type in range(6):
                origins[level, terrain_type] = torch.tensor([level + 0.5, terrain_type + 0.5, level - terrain_type])
        self._terrain_origins = origins
        self._terrain_levels = torch.zeros(count, dtype=torch.long)
        self._terrain_types = torch.zeros(count, dtype=torch.long)
        self._env_origins = torch.zeros(count, 3)

    @property
    def terrain_levels(self): return self._terrain_levels
    @property
    def terrain_types(self): return self._terrain_types
    @property
    def env_origins(self): return self._env_origins


def _assignments(task_ids, revision=0):
    task_ids = np.asarray(task_ids, dtype=np.int64)
    return TaskAssignmentBatch(task_ids, revision, 0, np.ones(len(task_ids)), np.full(len(task_ids), "bootstrap"))


class TestGenesisUEDAdapter(unittest.TestCase):
    def setUp(self):
        self.space = TaskSpace()
        self.sim = _Simulator(8)
        self.commands = torch.zeros(8, 4)
        self.adapter = GenesisUEDAdapter(
            task_space=self.space, simulator=self.sim, commands=self.commands,
            command_ranges={"lin_vel_x": [0.0, 2.0]}, device="cpu",
        )

    def test_assignment_is_atomic_and_batch_isolated(self):
        # vx bin 0: stairs_up L0; vx bin 1: rough L2.  Reset only envs 1 and 6.
        ids = torch.tensor([1, 6])
        tasks = [self.space.encode(self.space.decode(0)), self.space.encode(self.space.decode(38))]
        self.adapter.assign(ids, _assignments(tasks, revision=7))
        decoded = self.space.decode_batch(np.asarray(tasks, dtype=np.int64))
        self.assertTrue(torch.equal(self.sim.terrain_types[ids], torch.tensor(decoded.terrain_types)))
        self.assertTrue(torch.equal(self.sim.terrain_levels[ids], torch.tensor(decoded.terrain_levels)))
        self.assertTrue(torch.equal(self.sim.env_origins[ids], self.sim._terrain_origins[self.sim.terrain_levels[ids], self.sim.terrain_types[ids]]))
        self.assertTrue(torch.equal(self.adapter.active_task_id[ids], torch.tensor(tasks)))
        self.assertTrue(torch.equal(self.adapter.active_sampler_revision[ids], torch.tensor([7, 7])))
        self.assertTrue(torch.equal(self.adapter.active_task_id[[0, 2, 3, 4, 5, 7]], torch.full((6,), -1)))

    def test_old_assignment_provenance_survives_reassignment(self):
        ids = torch.tensor([0, 2])
        self.adapter.assign(ids, _assignments([0, 25], revision=3))
        self.adapter.record_step(torch.tensor([1.5, 0., 2.5, 0., 0., 0., 0., 0.]))
        outcome = self.adapter.collect_outcomes(ids, completion_revision=5, timed_out=torch.zeros(8, dtype=torch.bool))
        self.assertEqual(outcome.task_ids.tolist(), [0, 25])
        self.assertEqual(outcome.assigned_revision.tolist(), [3, 3])
        self.assertEqual(outcome.episodic_returns.tolist(), [1.5, 2.5])
        # Outcomes carry no validity flag anymore -- standstill is routed away
        # from collect_outcomes entirely, so every collected outcome is a mover.
        self.assertFalse(hasattr(outcome, "valid_for_curriculum"))
        # Replacing the assignments cannot retroactively alter the copied outcome.
        self.adapter.assign(ids, _assignments([42, 43], revision=6))
        self.assertEqual(outcome.task_ids.tolist(), [0, 25])

    def test_standstill_is_a_reserved_bucket_never_attributed(self):
        """A standstill episode is born labelled, holds zero, and skips the LP."""
        # env 1 is a mover; envs 0 and 2 are reserved standstill.
        self.adapter.assign(torch.tensor([1]), _assignments([21], revision=1))
        self.commands[torch.tensor([0, 2]), :3] = 1.7  # pretend they had commands
        self.adapter.assign_standstill(torch.tensor([0, 2]), _assignments([0, 42], revision=1))
        self.assertFalse(bool(self.adapter.episode_standstill[1]))
        self.assertTrue(bool(self.adapter.episode_standstill[0]))
        self.assertTrue(bool(self.adapter.episode_standstill[2]))
        # assign_standstill teleports to the placement terrain but zeros command.
        self.assertTrue(torch.all(self.commands[[0, 2], :3] == 0))
        self.assertTrue(torch.equal(self.adapter.active_task_id[[0, 2]], torch.tensor([0, 42])))
        # Its return folds into the standstill bucket, not any curriculum cell.
        self.adapter.record_step(torch.tensor([0., 0., 3.0, 0., 0., 0., 0., 0.]))
        self.adapter.record_standstill_outcomes(torch.tensor([2]))
        diag = self.adapter.standstill_diagnostics()
        self.assertEqual(diag["standstill_episode_count"], 1.0)
        self.assertAlmostEqual(diag["standstill_mean_return"], 3.0)

    def test_standstill_hold_zeros_commands_every_resample(self):
        """The birth label re-zeros standstill envs on every command resample."""
        ids = torch.tensor([0, 1, 2])
        self.adapter.assign(torch.tensor([1]), _assignments([21], revision=1))
        self.adapter.assign_standstill(torch.tensor([0, 2]), _assignments([0, 42], revision=1))
        self.commands[ids, :3] = 1.5
        motion_ids = self.adapter.apply_standstill_hold(ids)
        self.assertEqual(motion_ids.tolist(), [1])
        self.assertTrue(torch.all(self.commands[[0, 2], :3] == 0))
        self.assertTrue(torch.all(self.commands[1, :3] == 1.5))
        # Hold is sticky across many "resamples".
        for _ in range(8):
            self.commands[ids, :3] = 0.7
            motion_ids = self.adapter.apply_standstill_hold(ids)
            self.assertEqual(motion_ids.tolist(), [1])
            self.assertTrue(torch.all(self.commands[[0, 2], :3] == 0))
        # A fresh moving assignment clears the label so the next episode can move.
        self.adapter.assign(torch.tensor([0]), _assignments([3], revision=2))
        self.assertFalse(bool(self.adapter.episode_standstill[0]))

    def test_resampling_stays_inside_the_active_bin(self):
        ids = torch.tensor([0, 1, 2, 3])
        self.adapter.assign(ids, _assignments([0, 21, 42, 63]))
        for _ in range(16):
            self.adapter.resample_commands_within_active_bin(ids)
            self.assertTrue(torch.all(self.commands[ids, 0] >= self.adapter.active_vx_lower[ids]))
            self.assertTrue(torch.all(self.commands[ids, 0] < self.adapter.active_vx_upper[ids]))

    def test_cross_revision_occupancy_and_eval_style_no_mutation(self):
        teacher = UniformEpisodeCurriculum(self.space, stage_length_control_steps=1, seed=4)
        before = teacher.state_dict()
        # Collecting an adapter outcome is eval-style: it must not touch teacher state.
        self.adapter.assign(torch.tensor([0]), _assignments([0], revision=0))
        self.adapter.collect_outcomes(torch.tensor([0]), completion_revision=0, timed_out=torch.zeros(8, dtype=torch.bool))
        self.assertEqual(before["rng_bit_generator_state"], teacher.state_dict()["rng_bit_generator_state"])
        teacher.observe(EpisodeOutcomeBatch(
            task_ids=np.array([0]), assigned_revision=np.array([1]), completion_revision=2,
            episodic_returns=np.array([1.0]), episode_lengths=np.array([10]),
            terminal_reasons=np.array(["timeout"]),
        ))
        teacher.advance(1)
        self.assertEqual(teacher.state_dict()["transition_occupancy"], {"1:2": 1})


class TestUEDTrainingTerrain(unittest.TestCase):
    def test_deterministic_training_grid_and_builder_fingerprint_payload(self):
        cfg = SimpleNamespace(
            mesh_type="heightfield", simplify_mesh=False, terrain_length=6.0, terrain_width=6.0,
            platform_size=2.0, terrain_proportions=[.2] * 5, num_rows=4, num_cols=6,
            horizontal_scale=.1, vertical_scale=.005, border_size=2.0, curriculum=False,
            selected=False, terrain_kwargs=None, height_field_raw_override=None,
            ued_training_grid=True, ued_training_seed=17, taxonomy_showcase=False,
        )
        first, second = Terrain(cfg), Terrain(cfg)
        self.assertTrue(np.array_equal(first.height_field_raw, second.height_field_raw))
        self.assertEqual(first.env_origins.shape, (4, 6, 3))
        params = ued_training_builder_parameters(cfg)
        self.assertEqual(TaskSpace(builder_parameters=params).size, 84)
        self.assertEqual(params["builder"], "ued_training_grid_v1")


@unittest.skipUnless(__import__("os").environ.get("SIMULATOR") == "genesis", "Genesis smoke requires SIMULATOR=genesis")
class TestZGenesisTeleportSmoke(unittest.TestCase):
    def test_eight_task_teleport_without_heightfield_rebuild(self):
        """Real 64-env Genesis check for 2 velocity x 2 type x 2 level resets."""
        import genesis as gs
        import legged_gym.envs  # registers go2
        from legged_gym.utils import task_registry

        class SequenceTeacher:
            sampler_revision = 0
            def __init__(self, task_space): self.task_space, self.observed, self.calls = task_space, [], 0
            def sample(self, count, *, global_control_steps):
                if self.calls == 0:
                    ids = np.zeros(count, dtype=np.int64)
                else:
                    specs = [TaskSpec(typ, level, velocity) for velocity in range(2) for typ in range(2) for level in range(2)]
                    ids = np.asarray([self.task_space.encode(spec) for spec in specs], dtype=np.int64)
                self.calls += 1
                return _assignments(ids, revision=self.calls - 1)
            def draw_placements(self, count, *, global_control_steps):
                return _assignments(np.zeros(count, dtype=np.int64), revision=max(self.calls - 1, 0))
            def observe(self, outcomes): self.observed.append(outcomes)
            def advance(self, global_control_steps): return None

        gs.init(backend=gs.gpu, logging_level="warning")
        cfg, _ = task_registry.get_cfgs("go2")
        cfg.env.num_envs = 64
        cfg.env.ued_enabled = True
        cfg.terrain.mesh_type = "heightfield"
        cfg.terrain.curriculum = False
        cfg.terrain.selected = False
        cfg.terrain.ued_training_grid = True
        cfg.terrain.num_rows = 4
        cfg.terrain.num_cols = 6
        cfg.commands.ranges.lin_vel_x = [0.0, 2.0]
        cfg.commands.zero_cmd_prob = 0.0  # this test isolates teleport, not the standstill mixture
        args = SimpleNamespace(task="go2", seed=19, debug=False, headless=True, cpu=False,
                               num_envs=64, max_iterations=None, resume=False, sync_wandb=False,
                               ckpt=None, load_run=None, export_onnx=False, motion_file=None, num_student=None)
        env, _ = task_registry.make_env("go2", args=args, env_cfg=cfg)
        try:
            task_space = TaskSpace(builder_parameters=ued_training_builder_parameters(cfg.terrain))
            teacher = SequenceTeacher(task_space)
            heightfield_id = id(env.simulator._terrain.height_field_raw)
            env.enable_ued(teacher, task_space)
            ids = torch.arange(8, device=env.device)
            # reset_idx observes old assignments first, then applies exactly the
            # eight requested replacement tiles and teleports their root states.
            env.reset_idx(ids)
            self.assertEqual(len(teacher.observed), 1)
            self.assertTrue(np.all(teacher.observed[0].task_ids == 0))
            decoded = task_space.decode_batch(env.ued_adapter.active_task_id[ids].cpu().numpy())
            self.assertTrue(torch.equal(env.simulator.terrain_types[ids].cpu(), torch.tensor(decoded.terrain_types, device="cpu")))
            self.assertTrue(torch.equal(env.simulator.terrain_levels[ids].cpu(), torch.tensor(decoded.terrain_levels, device="cpu")))
            self.assertTrue(torch.allclose(env.simulator.env_origins[ids], env.simulator._terrain_origins[
                env.simulator.terrain_levels[ids], env.simulator.terrain_types[ids]
            ]))
            self.assertTrue(torch.all(env.commands[ids, 0] >= env.ued_adapter.active_vx_lower[ids]))
            self.assertTrue(torch.all(env.commands[ids, 0] < env.ued_adapter.active_vx_upper[ids]))
            self.assertEqual(id(env.simulator._terrain.height_field_raw), heightfield_id)
        finally:
            if hasattr(env, "destroy"):
                env.destroy()


if __name__ == "__main__":
    unittest.main()
