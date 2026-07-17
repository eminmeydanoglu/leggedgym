#!/usr/bin/env python3
"""Build the compact per-seed data sidecar used by the standalone V3 report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_dir", type=Path)
    args = parser.parse_args()

    source = args.campaign_dir / "tables" / "raw_cells.csv"
    destination = args.campaign_dir / "tables" / "eval_report_seed_data.js"
    data: dict[str, dict[str, dict[str, dict[str, float]]]] = {}

    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            suite = row["suite"]
            if suite not in {"s0", "s1", "s2"}:
                continue
            error = row["post_switch_tracking_error"] if suite == "s2" else row["tracking_error"]
            if not error:
                continue
            data.setdefault(suite, {}).setdefault(row["scenario"], {}).setdefault(
                row["model"], {}
            )[row["training_seed"]] = float(error)

    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    destination.write_text(f"window.SEED_DATA={payload};\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
