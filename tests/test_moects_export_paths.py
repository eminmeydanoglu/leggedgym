"""Focused regressions for export paths and task-aware debug sizing."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("SIMULATOR", "genesis")

import torch

from legged_gym.utils.helpers import (
    PolicyExporterMoECTS,
    PolicyExporterTS,
    debug_num_envs_for_task,
)


def _export_cfg():
    env_cfg = SimpleNamespace(env=SimpleNamespace(
        num_observations=3,
        num_history_obs=6,
    ))
    train_cfg = SimpleNamespace(runner=SimpleNamespace(
        load_run="run",
        checkpoint=7,
    ))
    return env_cfg, train_cfg


def _actor_critic():
    # Both exporters have the same (obs, history) call contract; the MoE
    # variant only changes the actor input ordering in forward().
    latent_dim = 2
    actor_input = 3 + latent_dim
    return SimpleNamespace(
        actor=torch.nn.Sequential(torch.nn.Linear(actor_input, 2)),
        history_encoder=torch.nn.Sequential(torch.nn.Linear(6, latent_dim)),
    )


class TestExportPath(unittest.TestCase):
    def test_ts_jit_and_onnx_share_export_directory(self):
        env_cfg, train_cfg = _export_cfg()
        exporter = PolicyExporterTS(_actor_critic())
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch("torch.onnx.export") as onnx_export:
                exporter.export(temp_dir, env_cfg, export_onnx=True,
                                train_cfg=train_cfg)

            self.assertTrue(os.path.isfile(os.path.join(temp_dir, "run_ite7.pt")))
            self.assertEqual(
                onnx_export.call_args.args[2],
                os.path.join(temp_dir, "run_ite7.onnx"),
            )

    def test_moects_inherits_fixed_export_path(self):
        env_cfg, train_cfg = _export_cfg()
        exporter = PolicyExporterMoECTS(_actor_critic())
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch("torch.onnx.export") as onnx_export:
                exporter.export(temp_dir, env_cfg, export_onnx=True,
                                train_cfg=train_cfg)

            self.assertTrue(os.path.isfile(os.path.join(temp_dir, "run_ite7.pt")))
            self.assertEqual(
                onnx_export.call_args.args[2],
                os.path.join(temp_dir, "run_ite7.onnx"),
            )


class TestDebugEnvCount(unittest.TestCase):
    def test_moects_uses_smallest_exact_role_split(self):
        self.assertEqual(debug_num_envs_for_task("go2_moects"), 4)

    def test_other_tasks_keep_one_env_debug_behavior(self):
        self.assertEqual(debug_num_envs_for_task("go2"), 1)
        # HIM has no teacher/student MoE role split and should not inherit the
        # MoE-CTS exception merely because it shares the WTY substrate.
        self.assertEqual(debug_num_envs_for_task("go2_moects_him"), 1)
