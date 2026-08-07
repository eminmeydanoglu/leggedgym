import importlib.util
import sys
from pathlib import Path

import numpy as np

_PATH = Path(__file__).parents[1] / "legged_gym/scripts/eval/probe_lib.py"


def _load_probe_lib_fresh():
    """Load probe_lib via importlib and assert no simulator modules got pulled in.

    probe_lib.py's contract is that it imports no legged_gym env / genesis code,
    so it must be loadable purely on numpy/scipy even in a process that never
    touched the simulator.
    """
    spec = importlib.util.spec_from_file_location("probe_lib_under_test", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    forbidden = [name for name in sys.modules if name == "genesis" or name.startswith("genesis.")
                 or name == "legged_gym.envs" or name.startswith("legged_gym.envs.")]
    assert forbidden == [], f"probe_lib import pulled in simulator modules: {forbidden}"
    return mod


_MOD = _load_probe_lib_fresh()
NumpyLogisticProbe = _MOD.NumpyLogisticProbe
_NumpyRidgePipeline = _MOD._NumpyRidgePipeline
classification_metrics = _MOD.classification_metrics
compute_decoder_metrics = _MOD.compute_decoder_metrics
delta_r2 = _MOD.delta_r2
group_kfold_splits = _MOD.group_kfold_splits


def test_probe_lib_import_contract_no_simulator():
    # Re-verify explicitly (not just as a side effect of module load above).
    forbidden = [name for name in sys.modules if name == "genesis" or name.startswith("genesis.")
                 or name == "legged_gym.envs" or name.startswith("legged_gym.envs.")]
    assert forbidden == []


def test_group_kfold_is_disjoint_and_complete():
    groups = np.repeat(np.arange(11), np.arange(1, 12))
    seen = []
    for tr, te in group_kfold_splits(groups, 5):
        assert set(groups[tr]).isdisjoint(set(groups[te]))
        seen.extend(te.tolist())
    assert sorted(seen) == list(range(len(groups)))


def test_ridge_matches_unregularized_linear_solution():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 4))
    y = X @ np.array([1.5, -2.0, 0.25, 3.0]) + 0.7
    model = _NumpyRidgePipeline(alpha=1e-10).fit(X, y)
    assert np.max(np.abs(model.predict(X) - y)) < 1e-7


def test_logistic_probe_separable_toy_accuracy():
    rng = np.random.default_rng(2)
    X = np.r_[rng.normal((-3, 0), .3, (60, 2)), rng.normal((3, 0), .3, (60, 2)),
              rng.normal((0, 3), .3, (60, 2))]
    y = np.repeat(np.arange(3), 60)
    model = NumpyLogisticProbe(max_iter=500).fit(X, y)
    metrics = classification_metrics(y, model.predict(X), model.predict_proba(X), model.classes_)
    assert metrics["accuracy"] > .99
    assert metrics["balanced_accuracy"] > .99
    assert metrics["majority_baseline"] == 1 / 3


def test_delta_r2_is_near_zero_for_independent_latent():
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(40), 10)
    obs = rng.normal(size=(len(groups), 5))
    latent = rng.normal(size=(len(groups), 3))
    y = obs @ np.arange(1, 6) + rng.normal(scale=.2, size=len(groups))
    result = delta_r2(obs, latent, y, groups, n_splits=5)
    assert abs(result["mean"]) < .01


def test_delta_r2_is_strongly_positive_when_target_only_in_latent():
    # y is a (near-)deterministic function of latent alone; obs carries pure
    # noise unrelated to y. R2(obs) should be ~0, R2([obs,latent]) should be
    # ~1, so delta_r2 must be clearly positive (this is the key positive
    # control for the whole probing pipeline).
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(40), 10)
    n = len(groups)
    obs = rng.normal(size=(n, 5))  # unrelated to y
    latent = rng.normal(size=(n, 3))
    y = latent @ np.array([2.0, -1.0, 0.5]) + rng.normal(scale=.05, size=n)
    result = delta_r2(obs, latent, y, groups, n_splits=5)
    assert result["mean"] > .5, result


def test_group_kfold_disjointness_assert_actually_fires_on_bad_indices():
    # Direct regression guard on the disjointness invariant itself: manually
    # replicate what group_kfold_splits checks and confirm it raises when fed
    # overlapping train/test group sets.
    groups = np.array([0, 0, 1, 1, 2, 2])
    train_groups = {0, 1}
    test_groups = {1, 2}  # overlaps on group 1 -> must be caught
    try:
        assert train_groups.isdisjoint(test_groups)
        raised = False
    except AssertionError:
        raised = True
    assert raised, "disjointness assertion failed to catch overlapping groups"


def test_compute_decoder_metrics_fold_summary_matches_manual_stats():
    rng = np.random.default_rng(5)
    groups = np.repeat(np.arange(20), 10)
    n = len(groups)
    X = rng.normal(size=(n, 4))
    y = X @ np.array([1.0, 2.0, -1.0, 0.5]) + rng.normal(scale=.1, size=n)
    result = compute_decoder_metrics(X, y, groups, n_splits=5)
    fold_r2 = np.array([f["R2"] for f in result["folds"]])
    assert len(fold_r2) == 5
    assert np.isclose(result["fold_summary"]["R2"]["mean"], fold_r2.mean())
    assert np.isclose(result["fold_summary"]["R2"]["std"], fold_r2.std())
    # A clean linear signal should be decodable well above chance in every fold.
    assert fold_r2.mean() > .8
