"""CPU contracts for the source-faithful upstream MoE-CTS probe."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


PATH = Path(__file__).parents[1] / "legged_gym/scripts/eval/probe_upstream_moe_cts.py"
SPEC = importlib.util.spec_from_file_location("probe_upstream_moe_cts_under_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_contract_and_observation_order_are_frozen():
    c = MODULE.CONTRACT
    assert c.num_obs == 45
    assert c.num_privileged_obs == 263
    assert c.history_length == 5
    assert c.history_dim == 225
    assert c.obs_terms == (
        "base_ang_vel", "projected_gravity", "commands",
        "dof_pos_error", "dof_vel", "previous_action",
    )


def test_history_zero_then_append_and_done_reset_semantics():
    collector = MODULE.SourceHistoryCollector(1)
    obs0 = torch.arange(45, dtype=torch.float32).view(1, 45)
    obs1 = obs0 + 1000
    state0 = collector.initial(obs0)
    assert torch.allclose(state0[0, :-1], torch.zeros(4, 45))
    assert torch.equal(state0[0, -1], obs0[0])
    state1 = collector.append(obs1)
    assert torch.equal(state1[0, -2], obs0[0])
    assert torch.equal(state1[0, -1], obs1[0])
    state2 = collector.append(obs0, torch.tensor([True]))
    assert torch.allclose(state2[0, :-1], torch.zeros(4, 45))
    assert torch.equal(state2[0, -1], obs0[0])


def test_joint_order_round_trip_is_named_not_positional():
    model = ("a", "b", "c")
    mujoco = ("b", "c", "a")
    adapter = MODULE.JointOrderAdapter(mujoco_joint_names=mujoco, model_joint_names=model)
    x = np.asarray([10.0, 20.0, 30.0])
    assert np.array_equal(adapter.to_mujoco(x), np.asarray([20.0, 30.0, 10.0]))
    assert np.array_equal(adapter.to_model(adapter.to_mujoco(x)), x)


def test_source_model_strict_round_trip_and_interventions(tmp_path):
    torch.manual_seed(7)
    source = MODULE.SourceActorCriticMoECTS()
    checkpoint = tmp_path / "model_123.pt"
    torch.save({"model_state_dict": source.state_dict(), "iter": 123}, checkpoint)
    adapter = MODULE.UpstreamMoECTSAdapter.from_checkpoint(checkpoint)
    assert adapter.loaded.schema == "source_training"
    assert adapter.loaded.teacher_available
    obs = torch.randn(5, 45)
    history = torch.randn(5, 5, 45)
    components = adapter.forward_components(obs, history)
    interventions = adapter.intervention_actions(components, seed=99)
    assert tuple(components["gate"].shape) == (5, 8)
    assert tuple(components["expert_outputs"].shape) == (5, 8, 32)
    assert tuple(components["normalized_mixed_latent"].shape) == (5, 32)
    assert tuple(interventions["single_expert_action"].shape) == (5, 8, 12)
    # L2 normalization is applied after every weighted expert intervention.
    assert torch.allclose(
        torch.linalg.vector_norm(interventions["normalized_latent_uniform"], dim=-1),
        torch.ones(5),
        atol=1e-5,
    )


def test_local_deployment_bridge_is_not_labelled_source_training(tmp_path):
    torch.manual_seed(13)
    source = MODULE.SourceActorCriticMoECTS()
    bridge_state = {}
    for key, value in source.state_dict().items():
        if key.startswith("student_moe_encoder."):
            bridge_state["history_encoder." + key[len("student_moe_encoder."):]] = value
        elif key.startswith("actor.network."):
            bridge_state["actor." + key[len("actor.network."):]] = value
    checkpoint = tmp_path / "bridge.pt"
    torch.save(
        {
            "model_state_dict": bridge_state,
            "infos": {
                "provenance": "student/actor-only deployment bridge; teacher and critic unavailable"
            },
        },
        checkpoint,
    )
    loaded = MODULE.load_upstream_checkpoint(checkpoint)
    assert loaded.schema == "deployment_bridge"
    assert loaded.teacher_available is False
    assert loaded.critic_available is False
    assert loaded.provenance["deployment_bridge"] is True


def test_unrelated_torchscript_checkpoint_is_rejected_as_mixed_only():
    # A tiny unrelated scripted actor is enough to exercise the explicit
    # failure path; genuine deploy MoE artifacts are accepted as
    # student/actor-only bridges, but never treated as raw checkpoints with a
    # teacher/critic.
    module = torch.jit.script(torch.nn.Linear(2, 2))
    path = Path("/tmp/probe_upstream_jit_only.pt")
    module.save(path)
    try:
        with pytest.raises(TypeError, match="mixed latent/action"):
            MODULE.load_upstream_checkpoint(path)
    finally:
        path.unlink(missing_ok=True)


def test_metrics_are_deterministic_and_teacher_unavailable_is_explicit():
    rng = np.random.default_rng(3)
    n = 24
    arrays = {
        "gate": np.asarray(rng.dirichlet(np.ones(8), size=n), dtype=np.float32),
        "expert_outputs": rng.normal(size=(n, 8, 32)).astype(np.float32),
        "normalized_mixed_latent": rng.normal(size=(n, 32)).astype(np.float32),
        "learned_action": rng.normal(size=(n, 12)).astype(np.float32),
        "action_uniform": rng.normal(size=(n, 12)).astype(np.float32),
        "action_shuffled": rng.normal(size=(n, 12)).astype(np.float32),
        "action_top1": rng.normal(size=(n, 12)).astype(np.float32),
        "single_expert_action": rng.normal(size=(n, 8, 12)).astype(np.float32),
        "commands": rng.normal(size=(n, 3)).astype(np.float32),
        "terrain_id": np.tile(np.arange(2), n // 2),
        "run_id": np.repeat(np.arange(6), 4),
    }
    first = MODULE.compute_specialization_metrics(arrays, metadata={"teacher_available": False}, seed=11)
    second = MODULE.compute_specialization_metrics(arrays, metadata={"teacher_available": False}, seed=11)
    assert first == second
    assert first["teacher_oracle"]["available"] is False
    assert first["teacher_oracle"]["reason"]


def test_metrics_exclude_nonfinite_route_rows_from_specialization_statistics():
    rng = np.random.default_rng(31)
    n = 6
    arrays = {
        "gate": np.asarray(rng.dirichlet(np.ones(8), size=n), dtype=np.float32),
        "expert_outputs": rng.normal(size=(n, 8, 32)).astype(np.float32),
        "normalized_mixed_latent": rng.normal(size=(n, 32)).astype(np.float32),
        "learned_action": rng.normal(size=(n, 12)).astype(np.float32),
        "action_uniform": rng.normal(size=(n, 12)).astype(np.float32),
        "action_shuffled": rng.normal(size=(n, 12)).astype(np.float32),
        "action_top1": rng.normal(size=(n, 12)).astype(np.float32),
        "single_expert_action": rng.normal(size=(n, 8, 12)).astype(np.float32),
        "commands": rng.normal(size=(n, 3)).astype(np.float32),
        "terrain_id": np.zeros(n, dtype=np.int64),
        "command_id": np.zeros(n, dtype=np.int64),
        "run_id": np.arange(n),
    }
    for key in (
        "gate", "expert_outputs", "normalized_mixed_latent", "learned_action",
        "action_uniform", "action_shuffled", "action_top1", "single_expert_action",
        "commands",
    ):
        arrays[key][2] = np.nan
    result = MODULE.compute_specialization_metrics(arrays, metadata={"teacher_available": False}, seed=4)
    assert result["n_samples"] == n
    assert result["n_finite_samples"] == n - 1
    assert result["n_excluded_nonfinite"] == 1
    assert np.isfinite(result["gate"]["entropy_mean"])
    assert result["route_mode_counts"] == {"offline": n - 1}


def test_fixed_bank_shuffle_is_global_and_batch_size_invariant(tmp_path):
    torch.manual_seed(19)
    source = MODULE.SourceActorCriticMoECTS()
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_state_dict": source.state_dict()}, checkpoint)
    adapter = MODULE.UpstreamMoECTSAdapter.from_checkpoint(checkpoint)
    rng = np.random.default_rng(21)
    n = 13
    bank = {
        "obs": rng.normal(size=(n, 45)).astype(np.float32),
        "history": rng.normal(size=(n, 5, 45)).astype(np.float32),
    }
    first = MODULE.analyze_fixed_bank(adapter, bank, seed=23, batch_size=4)
    second = MODULE.analyze_fixed_bank(adapter, bank, seed=23, batch_size=7)
    expected = first["gate"][np.random.default_rng(23).permutation(n)]
    assert np.allclose(first["gate_shuffled"], expected, atol=1e-7, rtol=0.0)
    assert np.allclose(first["gate_shuffled"], second["gate_shuffled"], atol=1e-7, rtol=0.0)
    assert np.allclose(first["action_shuffled"], second["action_shuffled"], atol=1e-6, rtol=0.0)


def test_command_and_terrain_classifier_is_group_aware_and_class_stratified():
    # Two independent rollout groups per class are the minimum needed for a
    # train/test split that contains every class on both sides.
    rows = []
    labels = []
    groups = []
    for cls in range(3):
        for repeat in range(2):
            group = f"mode{repeat}:cmd{cls}"
            for sample in range(3):
                rows.append([float(cls), float(sample)])
                labels.append(cls)
                groups.append(group)
    result = MODULE._classification_probe(
        np.asarray(rows), np.asarray(labels), np.asarray(groups), seed=17, target="command_id"
    )
    assert result["available"]
    assert result["split"] == "deterministic_group_class_stratified"
    assert result["group_disjoint"]
    assert result["ordinary_accuracy"] == pytest.approx(1.0)
    assert result["balanced_accuracy"] == pytest.approx(1.0)
    assert result["majority_baseline"] == pytest.approx(1 / 3)
    assert result["class_counts"]["train"] == {"0": 3, "1": 3, "2": 3}
    assert result["class_counts"]["test"] == {"0": 3, "1": 3, "2": 3}
    assert np.asarray(result["confusion_matrix"]).shape == (3, 3)


def test_classifier_reports_unavailable_when_a_class_has_one_group():
    result = MODULE._classification_probe(
        np.zeros((4, 2)),
        np.asarray([0, 0, 1, 1]),
        np.asarray(["g0", "g0", "g1", "g1"]),
        seed=3,
        target="terrain_id",
    )
    assert result["available"] is False
    assert "at least two" in result["reason"]


def test_route_modes_are_canonical_and_shuffled_is_offline_only():
    assert MODULE._normalize_route_modes(("learned", "expert0", "fixed_expert_1")) == (
        "learned", "fixed_expert_0", "fixed_expert_1"
    )
    with pytest.raises(ValueError, match="offline-only"):
        MODULE._normalize_route_modes(("shuffled",))


def test_closed_loop_route_action_selects_fixed_expert_and_terminates_shuffled():
    torch.manual_seed(12)
    source = MODULE.SourceActorCriticMoECTS()
    checkpoint = MODULE.LoadedMoECTSCheckpoint(
        model=source,
        path="synthetic.pt",
        sha256="synthetic",
        iteration=-1,
        schema="source_training",
        teacher_available=True,
        critic_available=True,
    )
    adapter = MODULE.UpstreamMoECTSAdapter(checkpoint)
    obs = torch.randn(1, 45)
    history = torch.randn(1, 5, 45)
    comp = adapter.forward_components(obs, history)
    interventions = adapter.intervention_actions(comp, seed=8)
    assert torch.equal(
        adapter.route_action(interventions, "expert0"), interventions["action_fixed_expert_0"]
    )
    with pytest.raises(ValueError, match="offline-only"):
        adapter.route_action(interventions, "shuffled")


def test_true_closed_loop_routes_reset_history_and_emit_route_metadata(tmp_path):
    pytest.importorskip("mujoco")
    reference_root = Path(__file__).parents[1] / "go2_rl_gym"
    terrain = reference_root / "resources/robots/go2/flat.xml"
    if not terrain.is_file():
        pytest.skip("reference MuJoCo terrain checkout is not present")
    torch.manual_seed(22)
    source = MODULE.SourceActorCriticMoECTS()
    checkpoint = MODULE.LoadedMoECTSCheckpoint(
        model=source,
        path="synthetic.pt",
        sha256="synthetic",
        iteration=-1,
        schema="source_training",
        teacher_available=True,
        critic_available=True,
    )
    result = MODULE.run_closed_loop(
        MODULE.UpstreamMoECTSAdapter(checkpoint),
        reference_root=reference_root,
        out_dir=tmp_path,
        terrains=("flat",),
        commands=MODULE.PAPER_COMMANDS[:1],
        command_labels=MODULE.PAPER_COMMAND_LABELS[:1],
        duration_s=0.02,
        simulation_dt=0.002,
        control_decimation=1,
        seed=5,
        max_rollouts=1,
        route_modes=("learned", "uniform", "top1", "fixed_expert_0", "fixed_expert_1"),
    )
    assert result["route_modes"] == ["learned", "uniform", "top1", "fixed_expert_0", "fixed_expert_1"]
    assert set(result["closed_loop_metrics"]) == set(result["route_modes"])
    with np.load(tmp_path / "probe.npz", allow_pickle=False) as bank:
        modes = bank["route_mode"].astype(str)
        assert set(modes) == set(result["route_modes"])
        for mode in result["route_modes"]:
            first = np.flatnonzero(modes == mode)[0]
            history = bank["history"][first]
            assert np.allclose(history[:-1], 0.0)
            route_rows = np.flatnonzero(modes == mode)
            assert np.all(np.diff(bank["episode_step"][route_rows]) >= 0)
            assert bank["done"][route_rows[-1]] or result["closed_loop_metrics"][mode]["survival_duration_s_mean"] == pytest.approx(0.02, abs=0.002)
        metadata = json.loads(str(bank["metadata_json"].item()))
    assert metadata["termination_semantics"].startswith("terminate_on_fall")
    assert metadata["offline_only_modes"] == ["shuffled"]


def test_closed_loop_shuffled_intervention_is_global_not_batch_one_identity(tmp_path):
    pytest.importorskip("mujoco")
    reference_root = Path(__file__).parents[1] / "go2_rl_gym"
    if not (reference_root / "resources/robots/go2/flat.xml").is_file():
        pytest.skip("reference MuJoCo terrain checkout is not present")
    torch.manual_seed(41)
    source = MODULE.SourceActorCriticMoECTS()
    adapter = MODULE.UpstreamMoECTSAdapter(
        MODULE.LoadedMoECTSCheckpoint(
            model=source,
            path="synthetic.pt",
            sha256="synthetic",
            iteration=-1,
            schema="source_training",
            teacher_available=True,
            critic_available=True,
        )
    )
    result = MODULE.run_closed_loop(
        adapter,
        reference_root=reference_root,
        out_dir=tmp_path,
        terrains=("flat",),
        commands=MODULE.PAPER_COMMANDS[:2],
        command_labels=MODULE.PAPER_COMMAND_LABELS[:2],
        duration_s=0.02,
        simulation_dt=0.002,
        control_decimation=1,
        seed=5,
        max_rollouts=2,
        route_modes=("learned",),
    )
    assert result["rollouts"] == 2
    with np.load(tmp_path / "probe.npz", allow_pickle=False) as bank:
        # Seed 5 swaps the two recorded command rows; N=1 per-step shuffling
        # would incorrectly make this intervention identical to learned.
        assert not np.allclose(bank["action_shuffled"], bank["learned_action"])
        metadata = json.loads(str(bank["metadata_json"].item()))
    assert metadata["shuffled_intervention_seed"] == 5


def test_closed_loop_terminates_on_fall_without_post_terminal_rows(tmp_path, monkeypatch):
    mujoco = pytest.importorskip("mujoco")
    reference_root = Path(__file__).parents[1] / "go2_rl_gym"
    if not (reference_root / "resources/robots/go2/flat.xml").is_file():
        pytest.skip("reference MuJoCo terrain checkout is not present")
    torch.manual_seed(31)
    source = MODULE.SourceActorCriticMoECTS()
    adapter = MODULE.UpstreamMoECTSAdapter(
        MODULE.LoadedMoECTSCheckpoint(
            model=source,
            path="synthetic.pt",
            sha256="synthetic",
            iteration=-1,
            schema="source_training",
            teacher_available=True,
            critic_available=True,
        )
    )
    original_step = mujoco.mj_step

    def force_fall(model, data):
        original_step(model, data)
        data.qpos[2] = 0.10

    monkeypatch.setattr(mujoco, "mj_step", force_fall)
    result = MODULE.run_closed_loop(
        adapter,
        reference_root=reference_root,
        out_dir=tmp_path,
        terrains=("flat",),
        commands=MODULE.PAPER_COMMANDS[:1],
        command_labels=MODULE.PAPER_COMMAND_LABELS[:1],
        duration_s=0.2,
        simulation_dt=0.002,
        control_decimation=1,
        route_modes=("learned",),
        seed=2,
        max_rollouts=1,
    )
    assert result["rollout_summaries"][0]["fell"] is True
    assert result["rollout_summaries"][0]["n_steps"] == 1
    assert result["rollout_summaries"][0]["survival_duration_s"] == pytest.approx(0.002)
    with np.load(tmp_path / "probe.npz", allow_pickle=False) as bank:
        assert bank["done"].tolist() == [True]
        assert bank["elapsed_s"].tolist() == pytest.approx([0.002])
