"""Checkpoint identity tests for the V6 frontier geometry contract.

These cover curriculum draws only.  They deliberately do not claim that a
runner resume restores simulator active episodes; that remains an
episode-boundary restart, not exact continuation.
"""
from __future__ import annotations

import numpy as np
import pytest

from legged_gym.envs.go2.go2_v6_frontier_config import Go2V6FrontierCfg, build_frontier_teacher


def _teacher_with_tile_size(tile_size: float):
    cfg = Go2V6FrontierCfg()
    cfg.terrain.terrain_length = tile_size
    cfg.terrain.terrain_width = tile_size
    return build_frontier_teacher(cfg)


def test_frontier_checkpoint_rejects_abandoned_80m_geometry_fingerprint():
    legacy, _ = _teacher_with_tile_size(80.0)
    current, _ = _teacher_with_tile_size(8.0)

    with pytest.raises(ValueError, match="task_space_fingerprint mismatch"):
        current.load_state_dict(legacy.state_dict())


def test_frontier_same_geometry_checkpoint_preserves_next_draw_at_episode_boundary():
    source, _ = _teacher_with_tile_size(8.0)
    source.sample(41, global_control_steps=0)
    state = source.state_dict()
    restored, _ = _teacher_with_tile_size(8.0)
    restored.load_state_dict(state)

    expected = source.sample(257, global_control_steps=0)
    actual = restored.sample(257, global_control_steps=0)
    np.testing.assert_array_equal(actual.task_ids, expected.task_ids)
    np.testing.assert_array_equal(actual.sources, expected.sources)
