"""Unit tests for the V6 frontier diagnostic bank (no Genesis)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml

from legged_gym.scripts.eval.frontier_diagnostic import (
    REGIME_A,
    REGIME_B,
    REGIME_C,
    REGIME_D,
    REGIME_E,
    SCHEMA_VERSION,
    assert_episode_matches_bank_row,
    bank_fingerprint,
    bank_row_identity,
    build_bank,
    config_fingerprint,
    expected_episode_count,
    family_for_column,
    frontier_success,
    load_config,
    merge_geometry_episode_files,
    normalized_linear_error,
    normalized_yaw_error,
    required_success_count,
    write_bank_artifacts,
    write_ndjson,
)
from legged_gym.scripts.eval.frontier_diagnostic_rollout import (
    apply_frontier_diag_env_overrides,
)
from legged_gym.utils.frontier.task_space import V4FrontierTaskSpace

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/eval/v6_frontier_diagnostic.yaml"


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIG_PATH)


def test_config_is_diagnostic_only_v2(config):
    assert config["purpose"] == "diagnostic_only"
    assert config["eligible_for_checkpoint_selection"] is False
    assert config["schema_version"] == SCHEMA_VERSION == "v6_frontier_diagnostic_v2"
    assert config["artifact"]["relative_root"] == "frontier_diagnostic_v2"


def test_artifact_dir_is_physically_separate_from_v1(config):
    from legged_gym.scripts.eval.frontier_diagnostic import artifact_dir

    p = artifact_dir(Path("logs/go2_v6_frontier/run"), 1500, config)
    assert "frontier_diagnostic_v2" in p.parts
    assert "frontier_diagnostic_v2" == p.parts[-2] or p.name == "model_1500"
    assert p.as_posix().endswith("frontier_diagnostic_v2/model_1500")


def test_bank_byte_stable(config):
    a = build_bank(config)
    b = build_bank(config)
    assert bank_fingerprint(a) == bank_fingerprint(b)
    assert [r.to_dict() for r in a] == [r.to_dict() for r in b]


def test_bank_fingerprint_changes_when_command_changes(config):
    rows = build_bank(config, geometry_seed=1)
    fp0 = bank_fingerprint(rows)
    cfg2 = json.loads(json.dumps(config))
    cfg2["commands"]["joint_seed"] = int(cfg2["commands"]["joint_seed"]) + 1
    rows2 = build_bank(cfg2, geometry_seed=1)
    assert bank_fingerprint(rows2) != fp0


def test_ten_physical_columns_and_family_map(config):
    rows = build_bank(config, geometry_seed=1)
    cols = sorted({r.physical_column for r in rows})
    assert cols == list(range(10))
    space = V4FrontierTaskSpace()
    for col in cols:
        assert family_for_column(col) == space.TERRAIN_FAMILIES[space.family_for_column(col)]


def test_level0_speed0_only(config):
    rows = build_bank(config)
    assert all(r.terrain_level == 0 and r.speed_bin == 0 for r in rows)


def test_controlled_vx_set(config):
    rows = build_bank(config, geometry_seed=1)
    vx = sorted({r.command_vx for r in rows if r.regime == REGIME_A})
    assert vx == [-0.45, -0.35, -0.25, 0.25, 0.35, 0.45]


def test_regime_matrices_complete(config):
    rows = build_bank(config, geometry_seed=1)
    by = {}
    for r in rows:
        by.setdefault(r.regime, []).append(r)
    assert len(by[REGIME_A]) == 60
    assert len(by[REGIME_B]) == 240
    assert len(by[REGIME_C]) == 360
    assert len(by[REGIME_D]) == 720
    assert len(by[REGIME_E]) == 240

    def key(r):
        return (r.physical_column, r.command_vx, r.command_vy, r.command_yaw)

    assert len({key(r) for r in by[REGIME_A]}) == 60
    assert len({key(r) for r in by[REGIME_B]}) == 240
    assert len({key(r) for r in by[REGIME_C]}) == 360
    assert len({r.pair_index for r in by[REGIME_D]}) == 12


def test_regime_e_two_segments_fixed_sign(config):
    rows = [r for r in build_bank(config, geometry_seed=1) if r.regime == REGIME_E]
    assert len(rows) == 240
    for r in rows:
        assert len(r.segment_commands) == 2
        s0, s1 = r.segment_commands
        assert s0.start_step == 0 and s1.start_step == 500
        assert math.copysign(1.0, s0.vx) == math.copysign(1.0, r.vx_sign)
        assert math.copysign(1.0, s1.vx) == math.copysign(1.0, r.vx_sign)


def test_total_episode_counts(config):
    assert expected_episode_count(config) == 4860
    assert expected_episode_count(config, geometry_seed=1) == 1620
    assert len(build_bank(config)) == 4860


def test_success_formula_parity():
    assert frontier_success(
        timed_out=True, mean_linear_error=0.35, mean_yaw_error=0.40
    )
    assert not frontier_success(
        timed_out=True, mean_linear_error=0.3500001, mean_yaw_error=0.40
    )
    assert not frontier_success(
        timed_out=True, mean_linear_error=0.35, mean_yaw_error=0.4000001
    )
    assert not frontier_success(
        timed_out=False, mean_linear_error=0.0, mean_yaw_error=0.0
    )


def test_timeout_comparison_is_strict_greater_than_documented():
    """Training uses episode_length_buf > max_episode_length (not >=)."""
    # Pure contract test: at max_episode_length the timeout is False; at +1 True.
    max_episode_length = 1000
    assert not (1000 > max_episode_length)
    assert 1001 > max_episode_length


def test_yaw_divisor_floor():
    assert normalized_yaw_error(0.0, 0.1) == pytest.approx(0.1 / 0.2)
    assert normalized_yaw_error(0.05, 0.0) == pytest.approx(0.05 / 0.2)
    assert normalized_linear_error(0.0, 0.0, 0.1, 0.0) == pytest.approx(0.1 / 0.2)


def test_required_success_count_mastery_beta11():
    assert required_success_count(32, 0.80) == 27
    assert required_success_count(32, 0.75) == 25
    assert required_success_count(32, 0.70) == 23
    assert (27 + 1) / (32 + 2) >= 0.80
    assert (26 + 1) / (32 + 2) < 0.80


def _complete_episode_stub(eid: str, bank_fp: str, cfg_fp: str, **over) -> dict:
    base = {
        "episode_id": eid,
        "schema_version": SCHEMA_VERSION,
        "purpose": "diagnostic_only",
        "eligible_for_checkpoint_selection": False,
        "bank_fingerprint": bank_fp,
        "config_fingerprint": cfg_fp,
        "checkpoint_sha256": "abc",
        "checkpoint_path": "/run/model_1500.pt",
        "checkpoint_iteration": 1500,
        "training_seed": 1,
        "git_commit": "deadbeef",
        "working_tree_dirty": False,
        "geometry_seed": 1,
        "scene_geometry_hash": "s" * 64,
        "runtime_tile_geometry_hash": "t" * 64,
        "geometry_hash": "s" * 64,
        "terrain_family": "stairs_up",
        "physical_column": 3,
        "terrain_column": 3,
        "terrain_level": 0,
        "speed_bin": 0,
        "regime": "A_baseline",
        "command_vx": 0.25,
        "command_vy": 0.0,
        "command_yaw": 0.0,
        "segment_commands": [{"start_step": 0, "vx": 0.25, "vy": 0.0, "yaw": 0.0}],
        "command_design": None,
        "command_seed": None,
        "pair_index": None,
        "schedule_index": None,
        "vx_sign": None,
        "requested_terrain_column": 3,
        "requested_terrain_level": 0,
        "runtime_terrain_column": 3,
        "runtime_terrain_level": 0,
        "mean_linear_error": 0.1,
        "mean_yaw_error": 0.1,
        "timed_out": True,
        "observed_timeout_event": True,
        "survived_measurement_horizon": True,
        "max_episode_length": 1000,
        "timeout_comparison": "episode_length_buf > max_episode_length",
        "frontier_success_at_original_thresholds": True,
    }
    base.update(over)
    return base


def test_merge_rejects_foreign_fingerprint(tmp_path, config):
    fp = bank_fingerprint(build_bank(config))
    cfg_fp = config_fingerprint(config)
    good = _complete_episode_stub("a", fp, cfg_fp)
    bad = _complete_episode_stub("b", "0" * 64, cfg_fp)
    p1 = tmp_path / "g1.ndjson"
    write_ndjson(p1, [good])
    p2 = tmp_path / "g2.ndjson"
    write_ndjson(p2, [bad])
    with pytest.raises(ValueError, match="bank_fingerprint"):
        merge_geometry_episode_files(
            [p1, p2],
            expected_bank_fp=fp,
            expected_config_fp=cfg_fp,
            expected_checkpoint_sha="abc",
            expected_count=2,
        )


def test_merge_rejects_command_tamper_against_bank(tmp_path, config):
    rows = build_bank(config, geometry_seed=1)
    row = rows[0]
    fp = bank_fingerprint(build_bank(config))
    cfg_fp = config_fingerprint(config)
    rec = _complete_episode_stub(
        row.episode_id,
        fp,
        cfg_fp,
        **bank_row_identity(row),
        terrain_column=row.physical_column,
        requested_terrain_column=row.physical_column,
        runtime_terrain_column=row.physical_column,
        requested_terrain_level=row.terrain_level,
        runtime_terrain_level=row.terrain_level,
    )
    # Tamper command while keeping header fingerprint.
    rec["command_vx"] = float(rec["command_vx"]) + 0.5
    p = tmp_path / "g1.ndjson"
    write_ndjson(p, [rec])
    with pytest.raises(ValueError, match="command_vx|mismatch"):
        merge_geometry_episode_files(
            [p],
            expected_bank_fp=fp,
            expected_config_fp=cfg_fp,
            expected_checkpoint_sha="abc",
            expected_count=1,
            bank_rows=[row],
        )


def test_write_bank_artifacts_fail_closed_on_identity_conflict(tmp_path, config):
    rows = build_bank(config, geometry_seed=1)
    out = tmp_path / "art"
    write_bank_artifacts(out, config, rows, checkpoint_sha256="sha_a")
    # Same identity is idempotent.
    write_bank_artifacts(out, config, rows, checkpoint_sha256="sha_a")
    # Conflicting checkpoint must fail closed.
    with pytest.raises(ValueError, match="refuse to overwrite|checkpoint_sha256"):
        write_bank_artifacts(out, config, rows, checkpoint_sha256="sha_b")


def test_bank_rows_include_terrain_column_alias(config):
    row = build_bank(config, geometry_seed=1)[0]
    d = row.to_dict()
    assert d["terrain_column"] == d["physical_column"]


def test_normalize_fills_training_seed_for_all_geometry_seeds():
    from legged_gym.scripts.eval.frontier_diagnostic import normalize_episode_record

    for geo in (1, 61001, 61002):
        rec = normalize_episode_record(
            {
                "episode_id": f"e{geo}",
                "physical_column": 3,
                "geometry_seed": geo,
                "run_dir": "/run",
                "checkpoint": "model_1500.pt",
                "training_seed_matched": geo == 1,
            },
            default_training_seed=1,
        )
        assert rec["training_seed"] == 1
        assert rec["terrain_column"] == 3
        assert rec["checkpoint_iteration"] == 1500


def test_env_override_keeps_curriculum_generation_flag():
    """Generation needs curriculum=True; runtime progression is disabled later."""
    from legged_gym.scripts.eval.frontier_diagnostic_rollout import (
        disable_runtime_terrain_curriculum,
    )

    class T:
        pass

    class C:
        def __init__(self):
            self.env = T()
            self.terrain = T()
            self.domain_rand = T()
            self.commands = T()
            self.control = T()
            self.sim = T()
            self.control.decimation = 4
            self.sim.dt = 0.005

    cfg = C()
    cfg.terrain.curriculum = False
    apply_frontier_diag_env_overrides(cfg, geometry_seed=1, episode_length_s=20.0)
    assert cfg.terrain.curriculum is True  # mesh builder path
    assert cfg.env.ued_enabled is False
    assert cfg.env.auto_reset is False
    assert abs(float(cfg.env.episode_length_s) - 20.0) < 1e-12

    class FakeEnv:
        def __init__(self):
            self.cfg = cfg
            self._called = False

        def _update_terrain_curriculum(self, env_ids):
            self._called = True

    env = FakeEnv()
    disable_runtime_terrain_curriculum(env)
    assert env.cfg.terrain.curriculum is False
    env._update_terrain_curriculum([0])  # no-op after disable
    assert env._called is False


def test_assert_episode_matches_bank_row_ok(config):
    row = build_bank(config, geometry_seed=1)[0]
    rec = bank_row_identity(row)
    rec["episode_id"] = row.episode_id
    assert_episode_matches_bank_row(rec, row)


def test_v5_frozen_config_untouched():
    v5 = ROOT / "configs/eval/v5_ued.yaml"
    text = v5.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    assert isinstance(data, dict)
    # Diagnostic package must not be wired into V5 selection artifact dirs.
    assert "frontier_diagnostic" not in text


def test_summarize_has_required_slices(config):
    from legged_gym.scripts.eval.frontier_diagnostic import summarize

    rows = build_bank(config, geometry_seed=1)[:20]
    recs = []
    for r in rows:
        recs.append(
            {
                **r.to_dict(),
                "timed_out": True,
                "observed_timeout_event": True,
                "mean_linear_error": 0.1,
                "mean_yaw_error": 0.1,
            }
        )
    summary = summarize(recs, config)
    assert "by_family_x_vx_magnitude" in summary
    assert "by_family_x_regime" in summary
    assert "by_vx_sign" in summary
    assert summary["eligible_for_checkpoint_selection"] is False


def test_threshold_sweep_includes_failure_decomposition(config):
    from legged_gym.scripts.eval.frontier_diagnostic import threshold_sweep

    rows = build_bank(config, geometry_seed=1)[:12]
    recs = [
        {
            **r.to_dict(),
            "timed_out": True,
            "mean_linear_error": 0.2 if i % 2 == 0 else 0.5,
            "mean_yaw_error": 0.2 if i % 3 else 0.6,
        }
        for i, r in enumerate(rows)
    ]
    sweep = threshold_sweep(recs, config)
    assert sweep["original_thresholds"]["required_success_count"] == 27
    assert "failure_decomposition" in sweep["grid"][0]


def test_report_answers_numeric_deltas(config):
    from legged_gym.scripts.eval.frontier_diagnostic import (
        recommend_v61,
        render_report_md,
        summarize,
        threshold_sweep,
    )

    rows = build_bank(config, geometry_seed=1)
    recs = []
    for r in rows:
        if r.regime == "A_baseline":
            lin, yaw = 0.30, 0.20
        elif r.regime == "B_vy_only":
            lin, yaw = 0.50, 0.20
        elif r.regime == "C_yaw_only":
            lin, yaw = 0.20, 0.20
        else:
            lin, yaw = 0.25, 0.25
        recs.append(
            {
                **r.to_dict(),
                "timed_out": True,
                "mean_linear_error": lin,
                "mean_yaw_error": yaw,
            }
        )
    summary = summarize(recs, config)
    sweep = threshold_sweep(recs, config)
    rec = recommend_v61(summary, sweep)
    report = render_report_md(
        summary,
        sweep,
        rec,
        checkpoint_sha256="abc",
        bank_fp="def",
        config_fp="ghi",
    )
    assert "Δ(B−A)" in report
    assert "Δ(C−A)" in report
    assert "27/32" in report
