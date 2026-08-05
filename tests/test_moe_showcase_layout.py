"""Static MoE-CTS play-showcase layout contract.

The showcase terrain is a shared heightfield: Genesis consumes its raw raster
directly and MuJoCo compiles that same raster.  Keep the visible grid contract
small and explicit so neither backend silently regains the former flat column.
"""

import os

os.environ.setdefault("SIMULATOR", "genesis")

from legged_gym.envs.go2.go2_moects.go2_moects_config import Go2MoECTSCfg
from legged_gym.utils.terrain import (
    MOE_SHOWCASE_LEVELS,
    MOE_SHOWCASE_STAIR_LEVELS,
    Terrain,
    moe_showcase_center_cell,
    moe_showcase_columns,
    moe_showcase_levels_for_column,
)


def _showcase_terrain():
    cfg = Go2MoECTSCfg()
    terrain_cfg = cfg.terrain
    terrain_cfg.mesh_type = "heightfield"
    terrain_cfg.curriculum = False
    terrain_cfg.selected = False
    terrain_cfg.ued_training_grid = False
    terrain_cfg.moe_grid = True
    terrain_cfg.moe_showcase = True
    terrain_cfg.moe_showcase_levels = MOE_SHOWCASE_LEVELS
    terrain_cfg.num_rows = len(MOE_SHOWCASE_LEVELS)
    terrain_cfg.num_cols = len(moe_showcase_columns(terrain_cfg.terrain_proportions))
    terrain_cfg.border_size = 2.0
    return Terrain(terrain_cfg)


def test_showcase_keeps_three_rows_but_stairs_use_adjacent_levels():
    assert MOE_SHOWCASE_LEVELS == (0, 4, 9)
    assert MOE_SHOWCASE_STAIR_LEVELS == (1, 2, 3)
    assert moe_showcase_levels_for_column("wave") == (0, 4, 9)
    assert moe_showcase_levels_for_column("stairs_up") == (1, 2, 3)
    assert moe_showcase_levels_for_column("stairs_down") == (1, 2, 3)


def test_flat_showcase_column_remains_flat():
    terrain_cfg = Go2MoECTSCfg().terrain
    columns = moe_showcase_columns(terrain_cfg.terrain_proportions)
    names = [name for name, _ in columns]

    assert "flat" in names
    assert names.count("stairs_up") == 1
    assert names.count("flat") == 1

    terrain = _showcase_terrain()
    assert len(terrain.name2cols["stairs_up"]) == 1
    assert len(terrain.name2cols["flat"]) == 1
    assert terrain.cols2id.count(3) == 1
    assert terrain.cols2id.count(8) == 1
    per_column_levels = {name: levels for _, name, levels in terrain.moe_showcase_labels}
    assert per_column_levels["wave"] == (0, 4, 9)
    assert per_column_levels["stairs_up"] == (1, 2, 3)
    assert per_column_levels["stairs_down"] == (1, 2, 3)


def test_three_by_seven_showcase_has_a_real_middle_tile():
    assert moe_showcase_center_cell(3, 7) == (1, 3)
