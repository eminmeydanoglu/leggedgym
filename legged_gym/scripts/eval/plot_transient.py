"""Overlay one or more transient .npz files into tracking-error time series.

Each .npz is one method's transient run (from `transient.py`) for the SAME scenario.
This draws the mean lin-vel tracking error (and base tilt) vs step, one line per
method, so the recovery/settling transient the sweep curves hide is visible
directly. For push_recovery the impulse step is marked; for step_response the
command-phase boundaries are marked.

Example:
    python legged_gym/scripts/eval/plot_transient.py \
        logs/eval/push_nodr.npz logs/eval/push_mlp.npz logs/eval/push_oracle.npz \
        --out logs/eval/push_recovery.png
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", nargs="+", help="transient .npz files to overlay")
    p.add_argument("--out", type=str, default="logs/eval/transient_curve.png")
    args = p.parse_args()

    runs = [np.load(f, allow_pickle=True) for f in args.npz]
    scenario = str(runs[0]["scenario"])

    # Overlaying only makes sense when every run is the SAME scenario (a push_recovery
    # error curve and a step_response one share no x-axis meaning).
    for f, r in zip(args.npz[1:], runs[1:]):
        if str(r["scenario"]) != scenario:
            raise SystemExit(f"scenario mismatch: {f} is '{r['scenario']}', expected '{scenario}'")

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle(f"Transient: {scenario}", fontsize=14, fontweight="bold")

    for r in runs:
        steps = np.arange(len(r["lin_err_ts"]))
        line, = axes[0].plot(steps, r["lin_err_ts"], label=str(r["label"]))
        axes[1].plot(steps, r["tilt_ts"], color=line.get_color(), label=str(r["label"]))

    # scenario-specific event markers
    if scenario == "push_recovery":
        push_step = int(runs[0]["push_step"])
        for ax in axes:
            ax.axvline(push_step, color="tab:red", ls="--", alpha=0.6,
                       label="impulse" if ax is axes[0] else None)
    else:  # step_response
        bounds = runs[0]["phase_bounds"]
        names = runs[0]["phase_names"]
        for ax in axes:
            for b, n in zip(bounds, names):
                ax.axvline(int(b), color="grey", ls=":", alpha=0.4)
        for b, n in zip(bounds, names):
            axes[0].text(int(b) + 1, axes[0].get_ylim()[1] * 0.95, str(n),
                         fontsize=7, rotation=90, va="top", color="grey")

    axes[0].set_ylabel("mean |cmd - v| [m/s]")
    axes[0].set_title("Lin-vel tracking error (lower / faster-settling better)")
    axes[1].set_ylabel("mean ||proj_grav_xy||")
    axes[1].set_title("Base tilt (lower better)")
    axes[1].set_xlabel("step")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=130)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
