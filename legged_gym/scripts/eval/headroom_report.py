#!/usr/bin/env python3
"""Render a self-contained MLP-to-Oracle headroom report.

This module deliberately consumes a *normalised* comparison table instead of
campaign artifacts.  It does not decide whether an oracle is safe or whether a
world belongs in scope; those decisions must have already been made by the
evaluation protocol.  Its only derived tracking quantities are transparent:

``absolute headroom = MLP tracking error - Oracle tracking error``
``method gap closed = (MLP tracking error - method tracking error) / headroom``

Tracking error is lower-is-better.  Fall rate and achieved-speed ratio are
displayed in a separate survival table, never folded into the headroom value.

Input JSON (UTF-8) shape::

  {
    "experiment": {
      "name": "V4 fixed-terrain comparison",
      "contract": "Same terrain, command bank, evaluation seeds and budget.",
      "seed_count": 3,
      "tracking_metric": "Mean planar tracking error (m/s)",
      "limitations": ["... optional, protocol-specific caveat ..."]
    },
    "worlds": [{
      "world": "Stairs / L7 / 1.0 m/s", "id_ood": "OOD",
      "include": true, "exclusion_reason": "",
      "tracking_error": {"MLP": 0.82, "Oracle": 0.50,
                         "DreamWaQ": 0.61, "HIM": 0.58},
      "fall_rate": {"MLP": 0.04, "Oracle": 0.01,
                    "DreamWaQ": 0.02, "HIM": 0.01},
      "achieved_speed_ratio": {"MLP": 0.88, "Oracle": 0.97,
                                "DreamWaQ": 0.94, "HIM": 0.96},
      "seed_consistency": {
        "DreamWaQ": {"better_seeds": 3, "total_seeds": 3},
        "HIM": {"better_seeds": 2, "total_seeds": 3}
      }
    }]
  }

CSV is one world per row.  Required columns are ``world,id_ood,include``, and
``tracking_error_mlp,tracking_error_oracle``.  Optional method columns use
``tracking_error_dreamwaq``, ``tracking_error_him``, ``fall_rate_<method>``,
``achieved_speed_ratio_<method>``, and ``seed_consistency_<method>`` (e.g.
``3/3``).  CSV experiment metadata is supplied with CLI options.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


METHODS = ("DreamWaQ", "HIM")
MODEL_LABELS = {"MLP": "MLP", "Oracle": "Oracle", "DreamWaQ": "DreamWaQ", "HIM": "HIM"}
METHOD_KEYS = {"DreamWaQ": "dreamwaq", "HIM": "him"}
EXCLUSION_LABELS = {
    "mlp_fall_rate_gt_0.05": "MLP düşme oranı %5’in üzerinde",
    "oracle_fall_rate_gt_0.05": "Oracle düşme oranı %5’in üzerinde",
    "oracle_achieved_speed_ratio_lt_0.90": "Oracle komut hızının %90’ına ulaşamadı",
    "absolute_tracking_headroom_lt_0.10": "MLP–Oracle tracking farkı 0.10’un altında",
    "secondary_tier_not_primary_headline": "Birleşik stress hücresi; ana ID/OOD sonucuna dahil değil",
}


@dataclass(frozen=True)
class Experiment:
    name: str
    contract: str
    seed_count: int
    tracking_metric: str
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class World:
    name: str
    id_ood: str
    include: bool
    exclusion_reason: str
    tracking: Mapping[str, float | None]
    fall_rate: Mapping[str, float | None]
    speed_ratio: Mapping[str, float | None]
    consistency: Mapping[str, str]
    gap_closed: Mapping[str, float | None]
    method_note: Mapping[str, str]


def _number(value: Any, *, field: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number or empty; received {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite; received {value!r}")
    return number


def _mapping_numbers(value: Any, *, field: str) -> dict[str, float | None]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return {str(key): _number(item, field=f"{field}.{key}") for key, item in value.items()}


def _consistency(value: Any, *, field: str, seed_count: int) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, str):
        return value.strip() or "—"
    if isinstance(value, Mapping):
        better = value.get("better_seeds")
        total = value.get("total_seeds", seed_count)
        try:
            better_int, total_int = int(better), int(total)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} requires better_seeds and total_seeds") from exc
        if total_int < 1 or better_int < 0 or better_int > total_int:
            raise ValueError(f"{field} has an invalid seed count {better_int}/{total_int}")
        return f"{better_int}/{total_int}"
    raise ValueError(f"{field} must be 'k/n' text or an object")


def _parse_world(payload: Mapping[str, Any], *, seed_count: int, index: int) -> World:
    where = f"worlds[{index}]"
    name = str(payload.get("world", "")).strip()
    if not name:
        raise ValueError(f"{where}.world must be a non-empty string")
    split = str(payload.get("id_ood", "")).strip().upper()
    if split not in {"ID", "OOD"}:
        raise ValueError(f"{where}.id_ood must be ID or OOD")
    raw_include = payload.get("include", True)
    include = raw_include if isinstance(raw_include, bool) else str(raw_include).strip().lower() in {"1", "true", "yes", "evet"}
    tracking = _mapping_numbers(payload.get("tracking_error"), field=f"{where}.tracking_error")
    if "MLP" not in tracking or "Oracle" not in tracking:
        raise ValueError(f"{where}.tracking_error must contain MLP and Oracle")
    raw_consistency = payload.get("seed_consistency", {})
    if not isinstance(raw_consistency, Mapping):
        raise ValueError(f"{where}.seed_consistency must be an object")
    raw_gap = payload.get("gap_closed", {})
    if not isinstance(raw_gap, Mapping):
        raise ValueError(f"{where}.gap_closed must be an object")
    raw_note = payload.get("method_note", {})
    if not isinstance(raw_note, Mapping):
        raise ValueError(f"{where}.method_note must be an object")
    return World(
        name=name, id_ood=split, include=include,
        exclusion_reason=str(payload.get("exclusion_reason", "")).strip(),
        tracking=tracking,
        fall_rate=_mapping_numbers(payload.get("fall_rate"), field=f"{where}.fall_rate"),
        speed_ratio=_mapping_numbers(payload.get("achieved_speed_ratio"), field=f"{where}.achieved_speed_ratio"),
        consistency={method: _consistency(raw_consistency.get(method), field=f"{where}.seed_consistency.{method}", seed_count=seed_count) for method in METHODS},
        gap_closed=_mapping_numbers(raw_gap, field=f"{where}.gap_closed"),
        method_note={str(key): str(value) for key, value in raw_note.items()},
    )


def normalise_json(payload: Mapping[str, Any]) -> tuple[Experiment, list[World]]:
    if not isinstance(payload, Mapping):
        raise ValueError("JSON root must be an object")
    experiment_raw = payload.get("experiment")
    worlds_raw = payload.get("worlds")
    if not isinstance(experiment_raw, Mapping):
        raise ValueError("experiment must be an object")
    if not isinstance(worlds_raw, list):
        raise ValueError("worlds must be an array")
    name = str(experiment_raw.get("name", "")).strip()
    contract = str(experiment_raw.get("contract", "")).strip()
    if not name or not contract:
        raise ValueError("experiment.name and experiment.contract are required")
    try:
        seed_count = int(experiment_raw.get("seed_count"))
    except (TypeError, ValueError) as exc:
        raise ValueError("experiment.seed_count must be a positive integer") from exc
    if seed_count < 1:
        raise ValueError("experiment.seed_count must be a positive integer")
    limitations_raw = experiment_raw.get("limitations", [])
    if not isinstance(limitations_raw, list) or not all(isinstance(item, str) and item.strip() for item in limitations_raw):
        raise ValueError("experiment.limitations must be an array of non-empty strings")
    worlds = [_parse_world(item, seed_count=seed_count, index=index) for index, item in enumerate(worlds_raw) if isinstance(item, Mapping)]
    if len(worlds) != len(worlds_raw):
        raise ValueError("each worlds item must be an object")
    if len({world.name for world in worlds}) != len(worlds):
        raise ValueError("world names must be unique")
    return Experiment(name, contract, seed_count, str(experiment_raw.get("tracking_metric", "Planar tracking error")).strip() or "Planar tracking error", tuple(item.strip() for item in limitations_raw)), worlds


def _csv_metric(row: Mapping[str, str], metric: str) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for model, slug in (("MLP", "mlp"), ("Oracle", "oracle"), ("DreamWaQ", "dreamwaq"), ("HIM", "him")):
        values[model] = _number(row.get(f"{metric}_{slug}"), field=f"{metric}_{slug}")
    return values


def normalise_csv(path: Path, *, name: str, contract: str, seed_count: int, tracking_metric: str, limitations: Sequence[str]) -> tuple[Experiment, list[World]]:
    if not name.strip() or not contract.strip() or seed_count < 1:
        raise ValueError("CSV mode requires --name, --contract, and a positive --seed-count")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not rows[0]:
        raise ValueError("CSV must contain a header and at least one world")
    required = {"world", "id_ood", "include", "tracking_error_mlp", "tracking_error_oracle"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"CSV missing required column(s): {', '.join(sorted(missing))}")
    worlds: list[World] = []
    for index, row in enumerate(rows):
        payload: dict[str, Any] = {
            "world": row.get("world"), "id_ood": row.get("id_ood"), "include": row.get("include"),
            "exclusion_reason": row.get("exclusion_reason", ""),
            "tracking_error": _csv_metric(row, "tracking_error"),
            "fall_rate": _csv_metric(row, "fall_rate"),
            "achieved_speed_ratio": _csv_metric(row, "achieved_speed_ratio"),
            "seed_consistency": {method: row.get(f"seed_consistency_{slug}", "") for method, slug in METHOD_KEYS.items()},
        }
        worlds.append(_parse_world(payload, seed_count=seed_count, index=index))
    if len({world.name for world in worlds}) != len(worlds):
        raise ValueError("world names must be unique")
    return Experiment(name.strip(), contract.strip(), seed_count, tracking_metric.strip() or "Planar tracking error", tuple(item.strip() for item in limitations if item.strip())), worlds


def load_input(path: Path, **csv_metadata: Any) -> tuple[Experiment, list[World]]:
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as handle:
            return normalise_json(json.load(handle))
    if path.suffix.lower() == ".csv":
        return normalise_csv(path, **csv_metadata)
    raise ValueError("input must be a .json or .csv file")


def _valid_headroom(world: World) -> tuple[bool, str]:
    if not world.include:
        return False, world.exclusion_reason or "Protokol dışında bırakıldı"
    mlp, oracle = world.tracking.get("MLP"), world.tracking.get("Oracle")
    if mlp is None or oracle is None:
        return False, "MLP veya Oracle tracking error eksik"
    if mlp <= oracle:
        return False, "MLP→Oracle tracking headroom pozitif değil"
    return True, ""


def _gap_closed(world: World, method: str) -> float | None:
    if method in world.gap_closed:
        # V4 aggregation stores the unitless fraction; the presentation
        # contract, labels, and rail all use percent.
        value = world.gap_closed[method]
        return None if value is None else 100.0 * value
    mlp, oracle, value = world.tracking.get("MLP"), world.tracking.get("Oracle"), world.tracking.get(method)
    if mlp is None or oracle is None or value is None or mlp <= oracle:
        return None
    return 100.0 * (mlp - value) / (mlp - oracle)


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}%"


def _escaped(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _exclusion_text(value: str) -> str:
    reasons = []
    for reason in value.split(";"):
        reason = reason.strip()
        if reason and reason not in reasons:
            reasons.append(reason)
    labels = []
    for reason in reasons:
        label = EXCLUSION_LABELS.get(reason)
        if label is None:
            match = re.fullmatch(r"(mlp|oracle)_fall_rate_gt_(\d+\.\d+)", reason)
            if match:
                label = f"{match.group(1).upper()} düşme oranı %{100.0 * float(match.group(2)):g}’in üzerinde"
            else:
                match = re.fullmatch(r"oracle_achieved_speed_ratio_lt_(\d+\.\d+)", reason)
                if match:
                    label = f"Oracle komut hızının %{100.0 * float(match.group(1)):g}’ına ulaşamadı"
                else:
                    match = re.fullmatch(r"(absolute|relative)_tracking_headroom_lt_(\d+\.\d+)", reason)
                    if match:
                        adjective = "mutlak" if match.group(1) == "absolute" else "göreli"
                        label = f"MLP–Oracle {adjective} tracking headroom eşiğin altında ({match.group(2)})"
        labels.append(label or reason.replace("_", " "))
    return "; ".join(labels)


def _rail(world: World) -> str:
    rail_points = []
    for method, class_name in (("DreamWaQ", "dw"), ("HIM", "him")):
        closed = _gap_closed(world, method)
        if closed is None:
            continue
        clamped = min(100.0, max(0.0, closed))
        # The visible line occupies 10–90% of the rail.  Calculate that
        # translation here rather than relying on unsupported CSS multiplication.
        visual_position = 10.0 + 0.8 * clamped
        rail_points.append(
            f'<span class="rail-marker {class_name}" style="--position:{visual_position:.4f}%" '
            f'aria-label="{_escaped(method)}: {_pct(closed)} boşluk kapandı"><b>{_escaped(method)}</b><i>{_pct(closed)}</i></span>'
        )
    return '<div class="gap-rail" role="img" aria-label="MLP ile Oracle arasındaki tracking headroom konumları">' \
           '<span class="rail-end mlp">MLP</span><span class="rail-line"></span><span class="rail-end oracle">Oracle</span>' \
           + "".join(rail_points) + "</div>"


def _tracking_rows(worlds: Iterable[World]) -> str:
    rows = []
    for world in worlds:
        headroom = world.tracking["MLP"] - world.tracking["Oracle"]  # valid rows only
        dream_note = world.method_note.get("DreamWaQ", f"{_pct(_gap_closed(world, 'DreamWaQ'))} kapandı")
        him_note = world.method_note.get("HIM", f"{_pct(_gap_closed(world, 'HIM'))} kapandı")
        cells = [
            f'<th scope="row"><span class="world-name">{_escaped(world.name)}</span><small>{_escaped(world.id_ood)}</small></th>',
            f'<td>{_fmt(world.tracking.get("MLP"))}</td>', f'<td>{_fmt(world.tracking.get("Oracle"))}</td>',
            f'<td class="headroom">{_fmt(headroom)}</td>',
            f'<td>{_fmt(world.tracking.get("DreamWaQ"))}<small>{_escaped(dream_note)}</small></td>',
            f'<td>{_fmt(world.tracking.get("HIM"))}<small>{_escaped(him_note)}</small></td>',
            f'<td class="consistency"><span>DW { _escaped(world.consistency["DreamWaQ"]) }</span><span>HIM { _escaped(world.consistency["HIM"]) }</span></td>',
            f'<td class="rail-cell">{_rail(world)}</td>',
        ]
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(rows) or '<tr><td colspan="8">Geçerli MLP→Oracle tracking headroom dünyası yok.</td></tr>'


def _survival_rows(worlds: Iterable[World]) -> str:
    rows = []
    for world in worlds:
        values = []
        for model in ("MLP", "Oracle", "DreamWaQ", "HIM"):
            fall = world.fall_rate.get(model)
            speed = world.speed_ratio.get(model)
            values.append(f'<td><b>{_fmt(fall, 3)}</b><small>fall rate<br>{_fmt(speed, 3)} speed ratio</small></td>')
        rows.append(f'<tr><th scope="row">{_escaped(world.name)}<small>{_escaped(world.id_ood)}</small></th>{"".join(values)}</tr>')
    return "\n".join(rows) or '<tr><td colspan="5">Gösterilecek geçerli dünya yok.</td></tr>'


def _excluded_rows(worlds: Iterable[World]) -> str:
    rows = []
    for world in worlds:
        valid, reason = _valid_headroom(world)
        if valid:
            continue
        mlp, oracle = world.tracking.get("MLP"), world.tracking.get("Oracle")
        absolute = None if mlp is None or oracle is None else mlp - oracle
        relative = None if absolute is None or mlp is None or mlp <= 0 else absolute / mlp
        reason_keys = {item.strip() for item in reason.split(";") if item.strip()}
        has_headroom_gate = any("tracking_headroom" in item for item in reason_keys)
        speed_limited = any("achieved_speed_ratio" in item for item in reason_keys)
        fall_limited = any("fall_rate" in item for item in reason_keys)
        if not has_headroom_gate and speed_limited and not fall_limited:
            status = "Headroom var; Oracle hız-limitli"
        elif not has_headroom_gate and fall_limited:
            status = "Headroom var; Oracle düşme-limitli"
        else:
            status = "Headroom eşiğin altında"
        rows.append(
            f'<tr><th scope="row">{_escaped(world.name)}<small>{_escaped(world.id_ood)}</small></th>'
            f'<td><b>{_escaped(status)}</b><small>{_escaped(_exclusion_text(reason))}</small></td>'
            f'<td>{_fmt(mlp)}</td><td>{_fmt(oracle)}</td><td class="headroom">{_fmt(absolute)}</td>'
            f'<td>{_pct(None if relative is None else 100.0 * relative)}</td>'
            f'<td>{_fmt(world.speed_ratio.get("Oracle"))}</td><td>{_fmt(world.fall_rate.get("Oracle"))}</td></tr>'
        )
    return "\n".join(rows) or '<tr><td colspan="8">Dışlanan dünya yok.</td></tr>'


def render_report(experiment: Experiment, worlds: Sequence[World]) -> str:
    valid = [world for world in worlds if _valid_headroom(world)[0]]
    excluded = [world for world in worlds if not _valid_headroom(world)[0]]
    base_limitations = (
        "Boşluk kapanma yüzdesi seed-bazlı GapClosed değerlerinin medyanıdır; error kolonları ayrı, açıklayıcı seed medyanlarıdır. Survival-gated yöntem için yüzde gösterilmez.",
        "Yüzde 0–100 aralığı dışına taşabilir. Cetvel işareti okunabilirlik için uçta tutulur; metin gerçek yüzdeyi korur.",
        "Seed tutarlılığı güven aralığı veya istatistiksel anlamlılık iddiası değildir; yalnızca belirtilen seedlerde MLP’ye göre iyileşme sayısıdır.",
    )
    limitations = list(experiment.limitations) + list(base_limitations)
    title = _escaped(experiment.name)
    return f'''<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} — Headroom raporu</title>
  <style>
    :root {{ --ink:#162330; --muted:#617181; --paper:#f5f8fa; --panel:#ffffff; --line:#cbd6df; --rule:#8ea0af; --mlp:#4b6274; --oracle:#1f3f59; --dw:#26708d; --him:#7a516d; --soft-dw:#dceef4; --soft-him:#eee4ec; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); background:var(--paper); font:15px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ max-width:1540px; margin:auto; padding:34px clamp(18px,4vw,58px) 70px; }}
    .kicker {{ margin:0 0 9px; color:var(--muted); font:600 11px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing:.12em; text-transform:uppercase; }}
    h1 {{ max-width:1000px; margin:0; font:650 clamp(28px,4vw,48px)/1.05 Georgia, "Times New Roman", serif; letter-spacing:-.035em; }}
    .lede {{ max-width:860px; margin:15px 0 0; color:#3e505f; font-size:17px; }}
    .contract {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:0; margin:30px 0 38px; border-top:2px solid var(--ink); border-bottom:1px solid var(--rule); }}
    .contract div {{ min-height:88px; padding:13px 17px 15px 0; border-right:1px solid var(--line); }} .contract div+div {{ padding-left:17px; }} .contract div:last-child {{ border-right:0; }}
    dt {{ color:var(--muted); font:600 10px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing:.09em; text-transform:uppercase; }} dd {{ margin:7px 0 0; font-weight:560; }}
    section {{ margin-top:48px; }} .section-head {{ display:flex; gap:18px; align-items:baseline; justify-content:space-between; border-bottom:1px solid var(--rule); padding-bottom:10px; margin-bottom:15px; }} h2 {{ margin:0; font:650 23px/1.15 Georgia, "Times New Roman", serif; letter-spacing:-.02em; }} .section-head p {{ margin:0; color:var(--muted); font-size:13px; text-align:right; }}
    .table-wrap {{ overflow-x:auto; background:var(--panel); border:1px solid var(--line); }} table {{ border-collapse:collapse; width:100%; min-width:1130px; }} caption {{ text-align:left; padding:13px 15px; color:#425462; font-size:13px; border-bottom:1px solid var(--line); }} th,td {{ padding:11px 12px; border-bottom:1px solid #dfe6eb; vertical-align:middle; text-align:right; white-space:nowrap; }} thead th {{ color:#405462; background:#f3f6f8; font:650 10px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing:.065em; text-transform:uppercase; }} th:first-child, tbody th {{ text-align:left; }} tbody tr:last-child > * {{ border-bottom:0; }} tbody tr:hover {{ background:#f8fbfc; }} small {{ display:block; margin-top:2px; color:var(--muted); font-size:11px; line-height:1.25; font-weight:450; white-space:normal; }} .world-name {{ display:block; max-width:220px; white-space:normal; line-height:1.25; }} .headroom {{ font-weight:700; }} .consistency span {{ display:block; font:600 11px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .rail-cell {{ min-width:280px; padding-right:18px; }} .gap-rail {{ position:relative; height:48px; min-width:260px; }} .rail-line {{ position:absolute; left:10%; right:10%; top:22px; height:2px; background:var(--rule); }} .rail-end {{ position:absolute; top:11px; z-index:2; padding:0 3px; background:var(--panel); font:700 10px/1 ui-monospace, SFMono-Regular, Menlo, monospace; }} .rail-end.mlp {{ left:0; color:var(--mlp); }} .rail-end.oracle {{ right:0; color:var(--oracle); }} .rail-marker {{ position:absolute; z-index:3; left:var(--position); top:5px; transform:translateX(-50%); display:grid; place-items:center; min-width:18px; text-align:center; }} .rail-marker::before {{ content:""; width:11px; height:11px; border:2px solid var(--panel); border-radius:50%; background:var(--dw); box-shadow:0 0 0 1px var(--dw); }} .rail-marker.him::before {{ background:var(--him); box-shadow:0 0 0 1px var(--him); }} .rail-marker b {{ position:absolute; top:19px; color:var(--ink); font:650 9px/1 ui-monospace, SFMono-Regular, Menlo, monospace; }} .rail-marker i {{ position:absolute; top:30px; color:var(--muted); font:10px/1 ui-monospace, SFMono-Regular, Menlo, monospace; font-style:normal; }}
    .survival td {{ min-width:132px; }} .survival td b {{ font:650 15px/1 ui-monospace, SFMono-Regular, Menlo, monospace; }} .exclusions {{ min-width:760px; }} .limitations {{ margin:0; padding:0; list-style:none; border-top:1px solid var(--line); }} .limitations li {{ padding:13px 0 13px 20px; border-bottom:1px solid var(--line); position:relative; color:#3e505f; }} .limitations li::before {{ content:"—"; position:absolute; left:0; color:var(--muted); }} footer {{ margin-top:38px; padding-top:12px; border-top:1px solid var(--rule); color:var(--muted); font:11px/1.35 ui-monospace, SFMono-Regular, Menlo, monospace; }}
    :focus-visible {{ outline:3px solid #5c9bb5; outline-offset:3px; }}
    @media (max-width:760px) {{ main {{ padding-top:25px; }} .contract {{ grid-template-columns:1fr 1fr; }} .contract div:nth-child(2) {{ border-right:0; }} .contract div:nth-child(n+3) {{ border-top:1px solid var(--line); }} .section-head {{ display:block; }} .section-head p {{ margin-top:6px; text-align:left; }} }}
    @page {{ margin:13mm; }}
    @media print {{ body {{ background:#fff; font-size:10pt; }} main {{ max-width:none; padding:0; }} .table-wrap {{ overflow:visible; }} table {{ min-width:0; font-size:8pt; }} th,td {{ padding:5px; white-space:normal; }} .rail-cell {{ min-width:180px; }} .gap-rail {{ min-width:170px; }} section {{ break-inside:avoid; margin-top:25px; }} .contract {{ margin-bottom:22px; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <p class="kicker">Headroom ölçümü / yalnızca tracking karşılaştırması</p>
    <h1>{title}</h1>
    <p class="lede">Geçerli dünyalarda MLP ile Oracle arasındaki ölçülebilir tracking boşluğunu ve DreamWaQ/HIM’in bu boşluğun ne kadarını kapattığını gösterir.</p>
    <dl class="contract">
      <div><dt>Deney sözleşmesi</dt><dd>{_escaped(experiment.contract)}</dd></div>
      <div><dt>Training seed</dt><dd>{experiment.seed_count}</dd></div>
      <div><dt>Tracking metriği</dt><dd>{_escaped(experiment.tracking_metric)}</dd></div>
      <div><dt>Geçerli dünya</dt><dd>{len(valid)} / {len(worlds)} <small>{len(excluded)} dışlandı</small></dd></div>
    </dl>
  </header>
  <section aria-labelledby="tracking-title">
    <div class="section-head"><h2 id="tracking-title">Tracking headroom</h2><p>Alt değer daha iyi. Yüzde, MLP→Oracle boşluğunun kapanan payıdır.</p></div>
    <div class="table-wrap"><table><caption>Yalnızca pozitif MLP→Oracle tracking headroom ile protokole dahil edilmiş dünyalar.</caption><thead><tr><th>Dünya</th><th>MLP</th><th>Oracle</th><th>Mutlak headroom</th><th>DreamWaQ</th><th>HIM</th><th>Seed tutarlılığı</th><th>MLP → Oracle cetveli</th></tr></thead><tbody>{_tracking_rows(valid)}</tbody></table></div>
  </section>
  <section aria-labelledby="survival-title">
    <div class="section-head"><h2 id="survival-title">Survival ve komut gerçekleşmesi</h2><p>Bu değerler tracking skoruna birleştirilmez.</p></div>
    <div class="table-wrap survival"><table><caption>Her hücre: düşme oranı, ardından ulaşılan hız oranı. Eksik ölçüm “—” olarak kalır.</caption><thead><tr><th>Dünya</th><th>MLP</th><th>Oracle</th><th>DreamWaQ</th><th>HIM</th></tr></thead><tbody>{_survival_rows(valid)}</tbody></table></div>
  </section>
  <section aria-labelledby="excluded-title">
    <div class="section-head"><h2 id="excluded-title">Dışlanan dünyalar</h2><p>Başlık sonucu yalnızca geçerli tracking headroom üzerinde yorumlanır.</p></div>
    <div class="table-wrap exclusions"><table><caption>Her satır iki training seed'in medyanıdır. Headroom sütunları, strict headline gate'inden bağımsız olarak MLP−Oracle farkını gösterir.</caption><thead><tr><th>Dünya</th><th>Durum ve neden</th><th>MLP hata</th><th>Oracle hata</th><th>Mutlak headroom</th><th>Göreli headroom</th><th>Oracle speed ratio</th><th>Oracle fall rate</th></tr></thead><tbody>{_excluded_rows(worlds)}</tbody></table></div>
  </section>
  <section aria-labelledby="limitations-title"><div class="section-head"><h2 id="limitations-title">Sınırlılıklar</h2></div><ul class="limitations">{"".join(f"<li>{_escaped(item)}</li>" for item in limitations)}</ul></section>
  <footer>Üretici: legged_gym.scripts.eval.headroom_report · Girdi: normalize edilmiş JSON/CSV · Rapor tek dosyalı ve çevrimdışı okunabilir.</footer>
</main>
</body>
</html>'''


def write_report(input_path: Path, output_path: Path, **csv_metadata: Any) -> Path:
    experiment, worlds = load_input(input_path, **csv_metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_report(experiment, worlds), encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize edilmiş MLP–Oracle headroom girdisinden tek dosyalı HTML raporu üretir.")
    parser.add_argument("--input", required=True, type=Path, help=".json veya .csv normalize girdi")
    parser.add_argument("--output", required=True, type=Path, help="yazılacak .html yolu")
    parser.add_argument("--name", default="", help="CSV deney adı")
    parser.add_argument("--contract", default="", help="CSV deney sözleşmesi")
    parser.add_argument("--seed-count", type=int, default=0, help="CSV training seed sayısı")
    parser.add_argument("--tracking-metric", default="Planar tracking error", help="CSV tracking metriği etiketi")
    parser.add_argument("--limitation", action="append", default=[], help="CSV için tekrarlanabilir sınırlılık")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = write_report(args.input, args.output, name=args.name, contract=args.contract, seed_count=args.seed_count, tracking_metric=args.tracking_metric, limitations=args.limitation)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"headroom-report: {exc}") from exc
    print(f"headroom report written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
