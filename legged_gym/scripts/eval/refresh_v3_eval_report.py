#!/usr/bin/env python3
"""Refresh the standalone V3 HTML report after an additional method finishes."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


METRICS = ("tracking_error", "fall_rate", "return_per_step", "achieved_speed_ratio")
METHOD = "HIM-fixed"


def as_float(value: str) -> float | None:
    return float(value) if value else None


def med(values: list[float | None]) -> float | None:
    values = [v for v in values if v is not None]
    return median(values) if values else None


def model_metrics(rows: list[dict[str, str]]) -> dict[str, float | None]:
    return {
        metric: med([as_float(row[metric]) for row in rows])
        for metric in METRICS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_dir", type=Path)
    args = parser.parse_args()
    root = args.campaign_dir
    report = root / "eval_report.html"
    raw_path = root / "tables" / "raw_cells.csv"
    score_path = root / "tables" / "scorecard_cells.csv"
    headline_path = root / "tables" / "headline.json"

    html = report.read_text(encoding="utf-8")
    match = re.search(r"const DATA = (.*?);\nconst MC", html, flags=re.S)
    if not match:
        raise RuntimeError(f"Could not find DATA payload in {report}")
    data = json.loads(match.group(1))

    with raw_path.open(newline="", encoding="utf-8") as handle:
        all_raw = list(csv.DictReader(handle))
    raw = [row for row in all_raw if row["model"] == METHOD]
    with score_path.open(newline="", encoding="utf-8") as handle:
        scores = [row for row in csv.DictReader(handle) if row["model"] == METHOD]
    headline = json.loads(headline_path.read_text(encoding="utf-8"))

    if not raw or not scores:
        raise RuntimeError(f"No {METHOD} rows found; report was not modified")

    if METHOD not in data["methods"]:
        data["methods"].append(METHOD)
    if METHOD not in data["models"]:
        oracle_i = data["models"].index(data["oracle"])
        data["models"].insert(oracle_i, METHOD)

    # Headline values are authoritatively produced by the campaign aggregator.
    headline_rows = [row for row in headline["headline"] if row["model"] == METHOD]
    static = {
        (row["training_seed"], row["model"]): row
        for row in all_raw
        if row["suite"] == "s0" and row["scenario"] == "static_id"
    }
    deltas = [
        as_float(static[(seed, METHOD)]["tracking_error"]) - as_float(static[(seed, data["baseline"])]["tracking_error"])
        for seed in {seed for seed, model in static if model == METHOD}
    ]
    fall_deltas = [
        as_float(static[(seed, METHOD)]["fall_rate"]) - as_float(static[(seed, data["baseline"])]["fall_rate"])
        for seed in {seed for seed, model in static if model == METHOD}
    ]
    id_entry = {
        "metric": "ID_delta_tracking", "model": METHOD,
        "n_training_seeds": len(deltas), "all_seeds_positive": None,
        "value_median": med(deltas), "value_min": min(deltas), "value_max": max(deltas),
        "fall_delta_median": med(fall_deltas),
    }
    entries = list(headline_rows) + [id_entry]
    data["headline_tiers"][METHOD] = {"strict": entries, "lenient": entries}

    # The report's donut counts one method's per-seed headline cells. The current
    # campaign emits the strict gate; retain the same counts for both UI toggles.
    counts = Counter(row["headline_status"] for row in scores)
    status = dict(counts)
    data["status_tiers"][METHOD] = {"strict": status, "lenient": status}
    eligible = sum(1 for row in scores if row["headline_include"] == "True")
    data["eligible_tiers"][METHOD] = {"strict": eligible, "lenient": eligible}

    # s0: two nominal scenarios, median over training seeds.
    for scenario in ("static_id", "dynamic_id_in_band"):
        rows = [row for row in raw if row["suite"] == "s0" and row["scenario"] == scenario]
        data["s0"][scenario][METHOD] = model_metrics(rows)

    # Per-cell gaps from scorecard rows; model measurements from raw rows.
    gap_rows: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in scores:
        gap_rows[(row["suite"], row["payload_name"], row["command_name"])].append(row)

    def attach_gap(cell: dict, suite: str, payload: str, command: str) -> None:
        rows = gap_rows[(suite, payload, command)]
        values = [as_float(row["headline_gap_closed"]) for row in rows]
        values = [value for value in values if value is not None]
        status_name = rows[0]["headline_status"] if rows else "no_oracle_headroom"
        cell.setdefault("gap", {})[METHOD] = {
            "gap_median": med(values),
            "gap_min": min(values) if values else None,
            "gap_max": max(values) if values else None,
            "eligible": len(values),
            "n": len(rows),
            "status": status_name,
            "headroom_abs": med([as_float(row["oracle_headroom_absolute"]) for row in rows]),
            "headroom_rel": med([as_float(row["oracle_headroom_relative"]) for row in rows]),
        }

    for cell in data["s1"]:
        rows = [row for row in raw if row["suite"] == "s1" and row["payload_name"] == cell["payload"] and row["command_name"] == cell["command"]]
        cell["models"][METHOD] = model_metrics(rows)
        attach_gap(cell, "s1", cell["payload"], cell["command"])
    for cell in data["s2"]:
        rows = [row for row in raw if row["suite"] == "s2" and row["payload_name"] == cell["switch"] and row["command_name"] == cell["command"]]
        cell["models"][METHOD] = model_metrics(rows)
        attach_gap(cell, "s2", cell["switch"], cell["command"])

    for family in ("kick", "terrain"):
        for point in data["s3"][family]:
            token = str(point["severity"]).replace(".", "_")
            prefix = f"/{family}_{token}"
            rows = [
                row for row in raw
                if row["suite"] == "s3" and prefix in row["path"]
            ]
            point["models"][METHOD] = model_metrics(rows)

    data["counts"] = {
        "raw_cells": sum(1 for _ in csv.DictReader(raw_path.open(encoding="utf-8"))),
        "scorecard_cells": sum(1 for _ in csv.DictReader(score_path.open(encoding="utf-8"))),
        "seed_scores": sum(1 for _ in csv.DictReader((root / "tables" / "scorecard_seed_scores.csv").open(encoding="utf-8"))),
    }
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    refreshed = html[: match.start(1)] + payload + html[match.end(1) :]
    refreshed = refreshed.replace(
        "const MC = {MLP:'#6b7a8d', 'Superset-Oracle':'#7c3aed', SysID:'#0c8a45', RMA:'#d97706', DreamWaQ:'#2563eb'};",
        "const MC = {MLP:'#6b7a8d', 'Superset-Oracle':'#7c3aed', SysID:'#0c8a45', RMA:'#d97706', DreamWaQ:'#2563eb', 'HIM-fixed':'#db2777'};",
    ).replace(
        "SysID:'Yöntem · SysID', RMA:'Yöntem · RMA', DreamWaQ:'Yöntem · DreamWaQ'};",
        "SysID:'Yöntem · SysID', RMA:'Yöntem · RMA', DreamWaQ:'Yöntem · DreamWaQ', 'HIM-fixed':'Yöntem · HIM-fixed'};",
    )
    report.write_text(refreshed, encoding="utf-8")
    print(f"refreshed {report} with {METHOD}")


if __name__ == "__main__":
    main()
