import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "legged_gym/scripts/eval/plot_moe_latent_pca.py"
SPEC = importlib.util.spec_from_file_location("plot_moe_latent_pca", MODULE_PATH)
PCA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PCA)


def _bank(path, n=8):
    terrain_block = np.repeat([8, 0, 3, 5], n)
    terrain = np.tile(terrain_block, 6)
    command_id = np.repeat(np.arange(6), len(terrain_block))
    rng = np.random.default_rng(4)
    z = rng.normal(size=(len(terrain), 12)).astype(np.float32)
    commands = PCA.np.asarray([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0],
        [0, -1, 0], [0, 0, 1], [0, 0, -1],
    ], dtype=np.float32)[command_id]
    np.savez(path, z_s=z, terrain_id=terrain, command_id=command_id, command=commands)


def test_paper_pca_figures_and_metadata(tmp_path):
    samples = tmp_path / "samples.npz"
    _bank(samples)
    inputs = PCA._parse_inputs([f"MoE-CTS={samples}"])
    for kind in ("terrain", "command"):
        output = tmp_path / f"{kind}.png"
        meta = PCA.make_figure(inputs, kind, output, max_per_class=10, seed=3)
        assert output.is_file()
        assert output.with_suffix(".json").is_file()
        assert meta["shared_pca"] is True
        assert len(meta["explained_variance_ratio"]) == 2


def test_command_plot_rejects_incomplete_classes(tmp_path):
    path = tmp_path / "incomplete.npz"
    np.savez(path, z_s=np.ones((3, 4)), terrain_id=np.full(3, 8),
             command_id=np.zeros(3), command=np.tile([1, 0, 0], (3, 1)))
    inputs = PCA._parse_inputs([f"bad={path}"])
    try:
        PCA.make_figure(inputs, "command", tmp_path / "bad.png")
    except ValueError as exc:
        assert "missing command classes" in str(exc)
    else:
        raise AssertionError("incomplete command bank must be rejected")
