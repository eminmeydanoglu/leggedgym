"""Overlay one or more sweep .npz files into degradation curves.

Each .npz is one method's sweep over the same axis. This draws method x OOD-point
curves for the metrics that matter, shading the in-distribution band so OOD
extrapolation is visually obvious.

Example:
    python legged_gym/scripts/eval/plot_sweep.py \
        logs/eval/friction_bench_nodr.npz logs/eval/friction_bench_mlp.npz \
        --out logs/eval/friction_curve.png
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (metric key, human title, y-label). fall_rate is the headline separator.
PANELS = [
    ("fall_rate", "Fall rate (lower better)", "fraction of episodes"),
    ("mean_ep_len", "Survival time (higher better)", "steps / episode"),
    ("tracking_lin_err", "Lin-vel tracking error", "|cmd - v| [m/s]"),
    ("mean_return", "Return (higher better)", "return / episode"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", nargs="+", help="sweep .npz files to overlay")
    p.add_argument("--out", type=str, default="logs/eval/sweep_curve.png")
    args = p.parse_args()

    runs = [np.load(f, allow_pickle=True) for f in args.npz]
    axis_name = str(runs[0]["axis"])
    unit = str(runs[0]["unit"])
    in_dist = runs[0]["in_dist"]
    grid = runs[0]["grid"]

    # Overlaying only makes sense when every curve shares the same x-axis. Guard
    # against silently plotting e.g. a friction sweep on top of an added_mass one,
    # or two runs on mismatched grids / in-dist bands.
    for f, r in zip(args.npz[1:], runs[1:]):
        if str(r["axis"]) != axis_name:
            raise SystemExit(f"axis mismatch: {f} is '{r['axis']}', expected '{axis_name}'")
        if str(r["unit"]) != unit:
            raise SystemExit(f"unit mismatch: {f} is '{r['unit']}', expected '{unit}'")
        if not np.array_equal(r["grid"], grid):
            raise SystemExit(f"grid mismatch: {f} has a different grid than {args.npz[0]}")
        if not np.array_equal(r["in_dist"], in_dist):
            raise SystemExit(f"in_dist mismatch: {f} has a different training band")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"OOD sweep: {axis_name}", fontsize=14, fontweight="bold")

    for ax, (key, title, ylabel) in zip(axes.flat, PANELS):
        # shade the in-distribution band (training range) -> everything else is OOD
        ax.axvspan(in_dist[0], in_dist[1], color="tab:green", alpha=0.08,
                   label="in-distribution" if ax is axes.flat[0] else None)
        for r in runs:
            grid = r["grid"]
            mean = r[f"{key}_mean"]
            p25, p75 = r[f"{key}_p25"], r[f"{key}_p75"]
            line, = ax.plot(grid, mean, marker="o", ms=4, label=str(r["label"]))
            ax.fill_between(grid, p25, p75, color=line.get_color(), alpha=0.15)
        ax.set_title(title)
        ax.set_xlabel(f"{axis_name} [{unit}]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes.flat[0].legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=130)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
