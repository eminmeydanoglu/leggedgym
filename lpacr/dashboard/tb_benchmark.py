#!/usr/bin/env python3
"""Summarize TensorBoard training scalars for LP-ACRL vs Uniform head-to-head.

Used by the Curriculum Atlas dashboard (``GET /api/benchmark``).  Reads
``run_manifest.json`` + ``events.out.tfevents*`` under a logs root, pairs the
latest ``lp_acrl`` and ``uniform`` runs that share a seed, and emits a JSON
table with mean-of-last-window metrics and short Turkish commentary.

No network I/O.  Stdout is a single JSON object.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


# (label, tensorboard tag, higher_is_better|None for "similar is fine")
METRIC_SPEC: list[tuple[str, str, bool | None]] = [
    ("Mean reward", "Train/mean_reward", True),
    ("Episode length", "Train/mean_episode_length", True),
    ("Linear tracking reward", "Episode/rew_tracking_lin_vel", True),
    ("Angular tracking reward", "Episode/rew_tracking_ang_vel", True),
    ("Collision penalty", "Episode/rew_collision", True),  # less negative = better
    ("Vertical-motion penalty", "Episode/rew_lin_vel_z", True),
    ("Action-rate penalty", "Episode/rew_action_rate", True),
    ("Action-smoothness penalty", "Episode/rew_action_smoothness", True),
    ("Value loss", "Loss/value_function", None),
    ("Policy noise std", "Policy/mean_noise_std", None),
]


def _load_manifest(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "run_manifest.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _has_events(run_dir: Path) -> bool:
    return any(run_dir.glob("events.out.tfevents*"))


def _mean_window(values: list[float], window: int) -> float | None:
    if not values:
        return None
    chunk = values[-window:] if window > 0 else values
    return float(statistics.mean(chunk))


def _read_scalars(run_dir: Path, tags: list[str], window: int) -> dict[str, Any]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    out: dict[str, Any] = {"tags": {}, "last_step": None, "n_points": {}}
    last_step = -1
    for tag in tags:
        if tag not in available:
            out["tags"][tag] = None
            out["n_points"][tag] = 0
            continue
        events = ea.Scalars(tag)
        vals = [float(e.value) for e in events if math.isfinite(e.value)]
        steps = [int(e.step) for e in events]
        out["tags"][tag] = _mean_window(vals, window)
        out["n_points"][tag] = len(vals)
        if steps:
            last_step = max(last_step, steps[-1])
    out["last_step"] = last_step if last_step >= 0 else None
    return out


def discover_runs(logs_dir: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not logs_dir.is_dir():
        return runs
    for path in logs_dir.iterdir():
        if not path.is_dir():
            continue
        manifest = _load_manifest(path)
        if not manifest or not _has_events(path):
            continue
        alg = manifest.get("ued_curriculum_algorithm") or manifest.get("algorithm")
        seed = manifest.get("training_seed")
        if alg is None:
            # Fall back to directory name heuristics.
            name = path.name.lower()
            if "lp_acrl" in name or "lpacrl" in name:
                alg = "lp_acrl"
            elif "uniform" in name:
                alg = "uniform"
            else:
                continue
        try:
            seed_i = int(seed) if seed is not None else None
        except (TypeError, ValueError):
            seed_i = None
        runs.append(
            {
                "path": str(path.resolve()),
                "name": path.name,
                "algorithm": str(alg),
                "seed": seed_i,
                "task": manifest.get("task"),
                "mtime": path.stat().st_mtime,
            }
        )
    return runs


def pick_pair(
    runs: list[dict[str, Any]],
    *,
    seed: int | None,
    left_alg: str,
    right_alg: str,
    left_path: str | None,
    right_path: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if left_path and right_path:
        left = next((r for r in runs if r["path"] == str(Path(left_path).resolve())), None)
        right = next((r for r in runs if r["path"] == str(Path(right_path).resolve())), None)
        if left is None:
            left = {
                "path": str(Path(left_path).resolve()),
                "name": Path(left_path).name,
                "algorithm": left_alg,
                "seed": seed,
                "task": None,
                "mtime": 0,
            }
        if right is None:
            right = {
                "path": str(Path(right_path).resolve()),
                "name": Path(right_path).name,
                "algorithm": right_alg,
                "seed": seed,
                "task": None,
                "mtime": 0,
            }
        return left, right

    candidates = runs
    if seed is not None:
        seeded = [r for r in candidates if r["seed"] == seed]
        if seeded:
            candidates = seeded

    def latest(alg: str) -> dict[str, Any] | None:
        matched = [r for r in candidates if r["algorithm"] == alg]
        if not matched and seed is not None:
            # Retry without seed filter if exact seed missing.
            matched = [r for r in runs if r["algorithm"] == alg]
        if not matched:
            return None
        return max(matched, key=lambda r: r["mtime"])

    left = latest(left_alg)
    right = latest(right_alg)
    # Prefer same seed when both exist and no explicit seed was forced.
    if left and right and seed is None and left.get("seed") is not None:
        same = [
            r
            for r in runs
            if r["algorithm"] == right_alg and r.get("seed") == left["seed"]
        ]
        if same:
            right = max(same, key=lambda r: r["mtime"])
        left_same = [
            r
            for r in runs
            if r["algorithm"] == left_alg and r.get("seed") == right["seed"]
        ]
        if left_same and right:
            # Re-pick left for the chosen right seed if right was newer.
            left_for_seed = max(
                [r for r in runs if r["algorithm"] == left_alg and r.get("seed") == right["seed"]],
                key=lambda r: r["mtime"],
                default=left,
            )
            left = left_for_seed
    return left, right


def _fmt(value: float | None, digits: int = 4) -> str | None:
    if value is None or not math.isfinite(value):
        return None
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    if abs(value) >= 1:
        return f"{value:.3f}"
    return f"{value:.{digits}f}"


def _comment(
    label: str,
    left: float | None,
    right: float | None,
    higher_is_better: bool | None,
    left_name: str,
    right_name: str,
) -> str:
    if left is None or right is None:
        return "Veri yok"
    if not (math.isfinite(left) and math.isfinite(right)):
        return "Veri yok"
    if right == 0 and left == 0:
        return "İkisi de 0"
    # Relative gap vs |right| (or |left| if right≈0).
    base = abs(right) if abs(right) > 1e-12 else abs(left)
    rel = (left - right) / base if base > 0 else 0.0
    abs_rel = abs(rel)

    if higher_is_better is None:
        if abs_rel < 0.03:
            return "Benzer"
        if abs_rel < 0.10:
            return "Benzer, hafif fark"
        winner = left_name if (left < right) else right_name  # for loss-like, lower often better
        if "loss" in label.lower() or "noise" in label.lower():
            better = left_name if left < right else right_name
            return f"Fark var ({better} daha düşük)" if "loss" in label.lower() else "Benzer değil"
        return "Fark var"

    # Penalties are negative rewards: higher (less negative) is better when higher_is_better=True.
    left_better = left > right if higher_is_better else left < right
    winner = left_name if left_better else right_name
    loser = right_name if left_better else left_name

    if abs_rel < 0.02:
        return "Benzer"
    if abs_rel < 0.05:
        return f"{winner} hafif önde"

    pct = abs_rel * 100.0
    # For negative penalties, "higher" means less penalty — phrase as better/worse.
    if "penalty" in label.lower() or left < 0 or right < 0:
        if left_better:
            return f"{winner} daha iyi"
        return f"{winner} daha iyi" if not left_better else f"{loser} daha kötü"

    if left_better:
        return f"{winner} yaklaşık %{pct:.0f} yüksek"
    return f"{winner} yaklaşık %{pct:.0f} yüksek"


def build_table(
    left_vals: dict[str, float | None],
    right_vals: dict[str, float | None],
    left_label: str,
    right_label: str,
) -> list[dict[str, Any]]:
    rows = []
    for label, tag, hib in METRIC_SPEC:
        lv = left_vals.get(tag)
        rv = right_vals.get(tag)
        comment = _comment(label, lv, rv, hib, left_label, right_label)
        # Winner for styling
        winner = None
        if lv is not None and rv is not None and math.isfinite(lv) and math.isfinite(rv):
            if hib is True:
                if abs(lv - rv) / (abs(rv) if abs(rv) > 1e-12 else abs(lv) or 1) >= 0.02:
                    winner = "left" if lv > rv else "right"
            elif hib is False:
                if abs(lv - rv) / (abs(rv) if abs(rv) > 1e-12 else abs(lv) or 1) >= 0.02:
                    winner = "left" if lv < rv else "right"
        rows.append(
            {
                "metric": label,
                "tag": tag,
                "left": lv,
                "right": rv,
                "left_fmt": _fmt(lv),
                "right_fmt": _fmt(rv),
                "comment": comment,
                "winner": winner,
                "higher_is_better": hib,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=50, help="Mean of last N scalar points")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--left-alg", default="lp_acrl")
    parser.add_argument("--right-alg", default="uniform")
    parser.add_argument("--left-path", default=None)
    parser.add_argument("--right-path", default=None)
    parser.add_argument("--left-label", default="LP-ACRL")
    parser.add_argument("--right-label", default="Uniform")
    args = parser.parse_args(argv)

    logs_dir = args.logs_dir.expanduser().resolve()
    runs = discover_runs(logs_dir)
    left, right = pick_pair(
        runs,
        seed=args.seed,
        left_alg=args.left_alg,
        right_alg=args.right_alg,
        left_path=args.left_path,
        right_path=args.right_path,
    )

    tags = [tag for _, tag, _ in METRIC_SPEC]
    payload: dict[str, Any] = {
        "ok": True,
        "logs_dir": str(logs_dir),
        "window": args.window,
        "left_label": args.left_label,
        "right_label": args.right_label,
        "left": None,
        "right": None,
        "rows": [],
        "error": None,
    }

    if left is None or right is None:
        payload["ok"] = False
        payload["error"] = (
            f"Could not pair {args.left_alg} vs {args.right_alg} under {logs_dir} "
            f"(found {len(runs)} runs with events)."
        )
        print(json.dumps(payload))
        return 0

    try:
        left_sc = _read_scalars(Path(left["path"]), tags, args.window)
        right_sc = _read_scalars(Path(right["path"]), tags, args.window)
    except Exception as exc:  # noqa: BLE001 — surface to dashboard
        payload["ok"] = False
        payload["error"] = f"TensorBoard read failed: {exc}"
        print(json.dumps(payload))
        return 0

    payload["left"] = {
        **left,
        "last_step": left_sc["last_step"],
        "n_reward_points": left_sc["n_points"].get("Train/mean_reward", 0),
    }
    payload["right"] = {
        **right,
        "last_step": right_sc["last_step"],
        "n_reward_points": right_sc["n_points"].get("Train/mean_reward", 0),
    }
    payload["rows"] = build_table(
        left_sc["tags"],
        right_sc["tags"],
        args.left_label,
        args.right_label,
    )
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
