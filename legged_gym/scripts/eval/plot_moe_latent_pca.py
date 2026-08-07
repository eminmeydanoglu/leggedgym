"""Paper-style PCA plots for CTS/MoE-CTS student latent banks.

This module is deliberately simulator-free.  It consumes ``samples.npz``
files produced by ``probe_moe_latent.py`` and fits one shared PCA transform per
figure, so panels/models remain directly comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TERRAIN_GROUPS = {
    "Flat": (8,),
    "Wave": (0,),
    "Stairs": (3, 4),
    "Obstacle": (5,),
}
TERRAIN_COLORS = {
    "Flat": "#63a8e2", "Wave": "#79bc57", "Stairs": "#ad58dd",
    "Obstacle": "#df4f91",
}

# The dedicated paper-PCA collection profile in probe_moe_latent.py.
COMMAND_LABELS = {
    0: "Forward", 1: "Backward", 2: "Strafe Left", 3: "Strafe Right",
    4: "Turn Left", 5: "Turn Right",
}
COMMAND_COLORS = {
    "Forward": "#63a8e2", "Backward": "#6dbc51",
    "Strafe Left": "#b7b7b7", "Strafe Right": "#b35be0",
    "Turn Left": "#f0aa29", "Turn Right": "#df3b7d",
}


def _parse_inputs(values):
    out = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--input must be LABEL=PATH, got: {value}")
        label, path = value.split("=", 1)
        data = np.load(path)
        required = {"z_s", "terrain_id", "command_id", "command"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{path} is missing fields: {missing}")
        out.append((label, data))
    return out


def _balanced_indices(labels, max_per_class, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        if len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        selected.append(idx)
    return np.concatenate(selected) if selected else np.empty(0, dtype=int)


def _fit_pca(arrays):
    x = np.concatenate(arrays, axis=0).astype(np.float64)
    mean = x.mean(axis=0)
    _, singular, vt = np.linalg.svd(x - mean, full_matrices=False)
    explained = singular[:2] ** 2 / np.maximum((singular ** 2).sum(), 1e-12)
    return mean, vt[:2], explained


def _project(x, mean, components):
    return (x.astype(np.float64) - mean) @ components.T


def _terrain_labels(terrain_ids):
    labels = np.full(len(terrain_ids), "", dtype=object)
    for name, ids in TERRAIN_GROUPS.items():
        labels[np.isin(terrain_ids, ids)] = name
    return labels


def _forward_mask(data, atol=1e-4):
    target = np.asarray([1.0, 0.0, 0.0])
    return np.all(np.isclose(data["command"][:, :3], target, atol=atol), axis=1)


def _select(inputs, kind, max_per_class, seed):
    selections = []
    for panel, (title, data) in enumerate(inputs):
        if kind == "terrain":
            labels = _terrain_labels(data["terrain_id"])
            mask = _forward_mask(data) & (labels != "")
        else:
            labels = np.asarray([COMMAND_LABELS.get(int(x), "") for x in data["command_id"]], dtype=object)
            mask = labels != ""
        rows = np.flatnonzero(mask)
        if not len(rows):
            raise ValueError(f"{title}: no samples selected for {kind} PCA")
        local = _balanced_indices(labels[rows], max_per_class, seed + panel)
        rows = rows[local]
        present = set(labels[rows])
        expected = set(TERRAIN_GROUPS if kind == "terrain" else COMMAND_LABELS.values())
        missing = sorted(expected - present)
        if missing:
            raise ValueError(f"{title}: missing {kind} classes: {missing}")
        selections.append((title, data["z_s"][rows], labels[rows]))
    return selections


def make_figure(inputs, kind, output, max_per_class=2500, seed=1, dpi=180):
    selections = _select(inputs, kind, max_per_class, seed)
    mean, components, explained = _fit_pca([x for _, x, _ in selections])
    fig, axes = plt.subplots(1, len(selections), figsize=(5.2 * len(selections), 4.2), squeeze=False)
    order = list(TERRAIN_GROUPS) if kind == "terrain" else list(COMMAND_LABELS.values())
    colors = TERRAIN_COLORS if kind == "terrain" else COMMAND_COLORS
    for ax, (title, latent, labels) in zip(axes[0], selections):
        points = _project(latent, mean, components)
        for label in order:
            mask = labels == label
            ax.scatter(points[mask, 0], points[mask, 1], s=7, alpha=0.65,
                       c=colors[label], label=label, edgecolors="none")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(f"PCA Dimension 1 ({100 * explained[0]:.1f}%)")
        ax.set_ylabel(f"PCA Dimension 2 ({100 * explained[1]:.1f}%)")
        ax.legend(frameon=True, fontsize=8, markerscale=1.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    meta = {
        "kind": kind, "panels": [x[0] for x in selections],
        "max_per_class": max_per_class, "seed": seed,
        "shared_pca": True, "explained_variance_ratio": explained.tolist(),
        "terrain_group_ids": {k: list(v) for k, v in TERRAIN_GROUPS.items()},
    }
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--kind", choices=("terrain", "command"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_per_class", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    make_figure(_parse_inputs(args.input), args.kind, args.output,
                args.max_per_class, args.seed, args.dpi)


if __name__ == "__main__":
    main()
