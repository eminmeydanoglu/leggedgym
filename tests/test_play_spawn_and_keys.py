"""Tests for taxonomy spawn assignment and multi-key keyboard drive."""
from __future__ import annotations

import os
import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from legged_gym.utils.terrain import (
    TAXONOMY_NUM_LEVELS,
    TAXONOMY_NUM_TYPES,
    TAXONOMY_TYPE_NAMES,
    apply_taxonomy_tile_assignment,
    clamp_taxonomy_spawn,
    taxonomy_tile_label,
    respawn_env_for_play,
)


class TestTaxonomySpawn(unittest.TestCase):
    def test_play_respawn_overwrites_training_randomization(self):
        import torch

        sim = SimpleNamespace(
            default_dof_pos=torch.arange(12, dtype=torch.float32).reshape(1, 12),
            base_init_pos=torch.tensor([0.0, 0.0, 0.42]),
            base_init_quat=torch.tensor([0.0, 0.0, 0.0, 1.0]),
            env_origins=torch.tensor([[4.0, 8.0, 0.3]]),
            reset_dofs=MagicMock(),
            reset_root_states=MagicMock(),
        )
        env = SimpleNamespace(
            simulator=sim,
            device=torch.device("cpu"),
            reset_idx=MagicMock(),
        )

        respawn_env_for_play(env, 0)

        env.reset_idx.assert_called_once()
        dof_args = sim.reset_dofs.call_args.args
        torch.testing.assert_close(dof_args[1], sim.default_dof_pos)
        torch.testing.assert_close(dof_args[2], torch.zeros_like(sim.default_dof_pos))
        root_args = sim.reset_root_states.call_args.args
        torch.testing.assert_close(root_args[1], torch.tensor([[4.0, 8.0, 0.72]]))
        torch.testing.assert_close(root_args[2], sim.base_init_quat.reshape(1, 4))
        torch.testing.assert_close(root_args[3], torch.zeros((1, 3)))
        torch.testing.assert_close(root_args[4], torch.zeros((1, 3)))

    def test_clamp(self):
        self.assertEqual(clamp_taxonomy_spawn(-1, 99), (0, TAXONOMY_NUM_TYPES - 1))
        self.assertEqual(clamp_taxonomy_spawn(3, 5), (3, 5))
        self.assertEqual(clamp_taxonomy_spawn(10, 1, num_rows=4, num_cols=6), (3, 1))

    def test_apply_assignment_numpy(self):
        origins = np.zeros((4, 6, 3), dtype=np.float64)
        for i in range(4):
            for j in range(6):
                origins[i, j] = [(i + 0.5) * 8.0, (j + 0.5) * 8.0, 0.1 * i]
        levels = np.zeros(2, dtype=np.int64)
        types = np.zeros(2, dtype=np.int64)
        env_origins = np.zeros((2, 3), dtype=np.float64)

        applied = apply_taxonomy_tile_assignment(
            levels, types, env_origins, origins, env_ids=[0], level=3, type_idx=2
        )
        self.assertEqual(int(levels[0]), 3)
        self.assertEqual(int(types[0]), 2)
        np.testing.assert_allclose(env_origins[0], origins[3, 2])
        np.testing.assert_allclose(np.asarray(applied), origins[3, 2])
        # env 1 unchanged
        self.assertEqual(int(levels[1]), 0)
        np.testing.assert_allclose(env_origins[1], 0.0)

    def test_apply_all_24_cells_unique(self):
        origins = np.zeros((4, 6, 3), dtype=np.float64)
        for i in range(4):
            for j in range(6):
                origins[i, j] = [i, j, 0.0]
        levels = np.zeros(1, dtype=np.int64)
        types = np.zeros(1, dtype=np.int64)
        env_origins = np.zeros((1, 3), dtype=np.float64)
        seen = set()
        for i in range(4):
            for j in range(6):
                apply_taxonomy_tile_assignment(
                    levels, types, env_origins, origins, [0], i, j
                )
                seen.add((int(levels[0]), int(types[0]), tuple(env_origins[0])))
        self.assertEqual(len(seen), 24)

    def test_label_english(self):
        self.assertEqual(taxonomy_tile_label(2, 0), "Ascending stairs L2")
        self.assertIn("Flat", TAXONOMY_TYPE_NAMES)


class TestMultiKeyDrive(unittest.TestCase):
    """Unit-test the pure key-event path without standing up a Viser server."""

    def _make_drive_stub(self):
        """Minimal object with the same drive-state fields as ViserViewer."""
        from legged_gym.utils.viser_viewer import ViserViewer

        # Build an unbound-like stub by instantiating without full __init__.
        stub = object.__new__(ViserViewer)
        keys = ("T", "F", "G", "H", "R", "Y")
        stub._drive_keys = keys
        stub._key_last_ts = {k: -1e9 for k in keys}
        stub._key_repeating = {k: False for k in keys}
        stub._key_sticky = {k: False for k in keys}
        stub._repeat_gap_max = 0.15
        stub._grace_initial = 0.6
        stub._grace_repeat = 0.18
        stub._multi_key_promote_window = 0.40
        stub._max_vel = {"fwd": 1.5, "back": 1.0, "lat": 0.7, "yaw": 1.5}
        stub._tau_accel = 0.10
        stub._tau_decel = 0.30
        stub._cmd_vel = np.zeros(3, dtype=float)
        stub._get_command_last_t = None
        return stub

    def test_second_key_does_not_drop_first(self):
        """Hold T (with repeats), then press R once → T stays active (sticky)."""
        v = self._make_drive_stub()
        t0 = 1000.0
        # First press T
        v._on_drive_key_event("T", now=t0)
        self.assertTrue(v._is_drive_key_active("T", now=t0 + 0.05))
        # Auto-repeats for T
        v._on_drive_key_event("T", now=t0 + 0.10)
        v._on_drive_key_event("T", now=t0 + 0.20)
        self.assertTrue(v._key_repeating["T"])
        # Press R once (new key) — browser would stop repeating T
        v._on_drive_key_event("R", now=t0 + 0.35)
        # Immediately after: both should be active; T sticky
        self.assertTrue(v._key_sticky["T"], "T must become sticky when R is pressed")
        self.assertTrue(v._is_drive_key_active("T", now=t0 + 0.35))
        self.assertTrue(v._is_drive_key_active("R", now=t0 + 0.35))
        # Long after R's grace would have expired if it were not sticky for T:
        # T still on (sticky), R off after grace
        later = t0 + 0.35 + 1.0
        self.assertTrue(v._is_drive_key_active("T", now=later))
        self.assertFalse(v._is_drive_key_active("R", now=later))

    def test_command_target_keeps_forward_with_yaw_pulse(self):
        v = self._make_drive_stub()
        t0 = 2000.0
        v._on_drive_key_event("T", now=t0)
        v._on_drive_key_event("T", now=t0 + 0.05)  # repeat
        v._on_drive_key_event("R", now=t0 + 0.20)

        # Snap command without ramp: inspect held targets via get_command internals
        now = t0 + 0.20
        v._get_command_last_t = now
        # Force instantaneous command by zeroing ramp state and large alpha
        v._tau_accel = 1e-9
        v._tau_decel = 1e-9
        # Monkeypatch time so get_command uses our clock... call held path directly
        fwd = v._max_vel["fwd"] if v._is_drive_key_active("T", now) else 0.0
        yaw = v._max_vel["yaw"] if v._is_drive_key_active("R", now) else 0.0
        self.assertGreater(fwd, 0.0)
        self.assertGreater(yaw, 0.0)

        # After R expires, forward remains
        later = now + 1.0
        self.assertGreater(
            v._max_vel["fwd"] if v._is_drive_key_active("T", later) else 0.0, 0.0
        )
        self.assertEqual(
            v._max_vel["yaw"] if v._is_drive_key_active("R", later) else 0.0, 0.0
        )

    def test_sticky_toggle_off(self):
        v = self._make_drive_stub()
        t0 = 3000.0
        v._on_drive_key_event("T", now=t0)
        v._on_drive_key_event("R", now=t0 + 0.2)  # T sticky
        self.assertTrue(v._key_sticky["T"])
        # Second press of T clears sticky
        v._on_drive_key_event("T", now=t0 + 0.5)
        self.assertFalse(v._key_sticky["T"])
        self.assertFalse(v._is_drive_key_active("T", now=t0 + 0.5))

    def test_clear_drive_command(self):
        v = self._make_drive_stub()
        t0 = 4000.0
        v._on_drive_key_event("T", now=t0)
        v._on_drive_key_event("R", now=t0 + 0.1)
        v._cmd_vel[:] = [1.0, 0.0, 0.5]
        v.clear_drive_command()
        self.assertFalse(any(v._key_sticky.values()))
        self.assertTrue(np.allclose(v._cmd_vel, 0.0))
        self.assertFalse(v._is_drive_key_active("T", now=t0 + 0.2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
