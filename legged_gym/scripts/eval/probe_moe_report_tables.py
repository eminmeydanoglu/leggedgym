"""Render the MoE-CTS latent-probing report tables from probe_metrics.json.

The report (`logs/eval/MOE_LATENT_PROBING_RAPOR.md`) quotes six checkpoints x two
latents x three columns. Transcribing that by hand is how numbers drift, so the
tables are generated instead.

Every table shows the same three quantities side by side:

    R2(obs45)        the observation-only baseline probe
    R2(obs45 + z)    the same probe with the latent concatenated
    dR2              what the latent added

so a reader can see whether a small delta sits on top of a strong baseline or a
useless one -- a distinction the delta alone hides.

Usage:
    python legged_gym/scripts/eval/probe_moe_report_tables.py            # all sections
    python legged_gym/scripts/eval/probe_moe_report_tables.py --section physics
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
EVAL = os.path.join(ROOT, "logs", "eval")

# iteration -> probe_metrics.json. Phase A's `analysis/` directory is the
# pre-fix pipeline (broken fold split, diluted targets) and is deliberately not
# referenced here; `analysis_fixed/` supersedes it. Phase B was produced after
# the fix, so its `analysis/` is already correct.
CHECKPOINTS: list[tuple[int, str]] = [
    (500, "moe_latent_probe_phaseB/analysis/model_500"),
    (2500, "moe_latent_probe_phaseB/analysis/model_2500"),
    (7500, "moe_latent_probe_phaseB/analysis/model_7500"),
    (12500, "moe_latent_probe_phaseB/analysis/model_12500"),
    (20500, "moe_latent_probe_phaseA/analysis_fixed/best_tracking"),  # best_tracking.pt
    (23500, "moe_latent_probe_phaseA/analysis_fixed/model_23500"),
]

# Targets the pipeline sweeps one axis at a time and scores only inside that
# axis's own block, hence the per-target row counts.
PHYSICS = [
    ("friction", "sürtünme katsayısı"),
    ("added_mass", "gövdeye eklenen kütle (kg)"),
    ("com_x", "ağırlık merkezi kayması, ileri eksen (m)"),
    ("com_y", "ağırlık merkezi kayması, yanal eksen (m)"),
    ("com_z", "ağırlık merkezi kayması, dikey eksen (m)"),
    ("pd_gain_scale", "PD kazanç çarpanı"),
]

STATE = [
    ("base_lin_vel_x", "gövde ileri hızı (m/s)"),
    ("base_lin_vel_z", "gövde dikey hızı (m/s)"),
    ("base_ang_vel_z", "gövde yaw hızı (rad/s)"),
    ("torque_norm", "12 eklem torkunun L2 normu (N·m)"),
    ("dof_acc_norm", "12 eklem ivmesinin L2 normu (rad/s²)"),
    ("terrain_level", "zemin zorluk seviyesi (müfredat indeksi)"),
    ("gait_phase", "yürüyüş fazı"),
]

# (target, description, n_classes). Chance level for BALANCED accuracy is 1/K --
# not the `majority_baseline` field in the json, which is a PLAIN-accuracy
# baseline and is not comparable to a balanced-accuracy score.
CLASSIFICATION = [
    ("terrain_id", "zemin tipi", 7),
    ("contact_pattern", "ayak temas deseni", 16),
    ("command_id", "komut hücresi kimliği", 6),
]

INTERVENE = [
    ("intervene_friction_hi", "sürtünme yüksek"),
    ("intervene_friction_lo", "sürtünme düşük"),
    ("intervene_pdgain_hi", "pd_gain yüksek"),
]

MODE_ORDER = ["student", "teacher_true", "shuffled_matched", "wrong_regime"]


def load(rel: str) -> dict[str, Any]:
    with open(os.path.join(EVAL, rel, "probe_metrics.json")) as fh:
        return json.load(fh)


def r2(doc: dict, feature: str, target: str) -> float | None:
    """Group-CV fold-mean R^2 for `feature -> target`."""
    try:
        v = doc["features"][feature]["regression"][target]["fold_summary"]["R2"]["mean"]
    except KeyError:
        return None
    return None if v is None or math.isnan(v) else float(v)


def stored_delta(doc: dict, feature: str, target: str) -> tuple[float, float] | None:
    """Pipeline's own dR2: mean over per-fold (R2_concat - R2_obs) differences."""
    try:
        d = doc["features"][feature]["regression"][target]["delta_r2_vs_obs45"]
    except KeyError:
        return None
    if d.get("mean") is None or math.isnan(d["mean"]):
        return None
    return float(d["mean"]), float(d.get("std", float("nan")))


def bal_acc(doc: dict, feature: str, target: str) -> float | None:
    try:
        v = doc["features"][feature]["classification"][target]["fold_summary"]["balanced_accuracy"]["mean"]
    except KeyError:
        return None
    return None if v is None or math.isnan(v) else float(v)


def stored_delta_bal(doc: dict, feature: str, target: str) -> tuple[float, float] | None:
    try:
        d = doc["features"][feature]["classification"][target]["delta_balanced_accuracy_vs_obs45"]
    except KeyError:
        return None
    if d.get("mean") is None or math.isnan(d["mean"]):
        return None
    return float(d["mean"]), float(d.get("std", float("nan")))


def n_rows(doc: dict, target: str) -> int | None:
    try:
        return doc["features"]["z_s"]["regression"][target].get("n_rows")
    except KeyError:
        return None


def fmt(v: float | None, nd: int = 3) -> str:
    return "—" if v is None else f"{v:+.{nd}f}"


def fmt_pm(pair: tuple[float, float] | None, nd: int = 3) -> str:
    if pair is None:
        return "—"
    m, s = pair
    return f"**{m:+.{nd}f}** ± {s:.{nd}f}" if not math.isnan(s) else f"**{m:+.{nd}f}**"


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #

def section_regression(docs, targets, *, derived_delta: bool, heading: str, note: str) -> str:
    out = [heading, "", note, ""]
    for target, desc in targets:
        rows = []
        any_data = False
        for it, doc in docs:
            base = r2(doc, "obs45", target)
            cells = [f"| {it:,} ".replace(",", ".")]
            cells.append(f"| {fmt(base)} ")
            for latent in ("z_s", "z_t"):
                concat = r2(doc, f"obs45+{latent}", target)
                delta = stored_delta(doc, latent, target)
                if delta is None and derived_delta and concat is not None and base is not None:
                    delta = (concat - base, float("nan"))
                if concat is not None or delta is not None:
                    any_data = True
                cells.append(f"| {fmt(concat)} | {fmt_pm(delta)} ")
            rows.append("".join(cells) + "|")

        nr = n_rows(docs[-1][1], target)
        scope = f" · puanlanan satır: {nr:,}".replace(",", ".") if nr else ""
        out.append(f"**`{target}`** — {desc}{scope}")
        out.append("")
        if not any_data:
            out += ["> Bu hedefte probe puanlanamadı (aşağıdaki nota bakın).", ""]
            continue
        out.append("| iter | R² gözlem | R² gözlem+z_s | ΔR² (z_s) | R² gözlem+z_t | ΔR² (z_t) |")
        out.append("|---:|---:|---:|---:|---:|---:|")
        out += rows
        out.append("")
    return "\n".join(out)


def section_classification(docs) -> str:
    out = [
        "Bütün hücreler **dengeli doğruluk** (her sınıfın recall'unun ortalaması). "
        "Şans seviyesi `1/sınıf sayısı`; sınıf dengesizliğinden etkilenmez. "
        "JSON'daki `majority_baseline` alanı **düz** doğruluk tabanıdır ve bu sütunlarla "
        "kıyaslanamaz, o yüzden buraya alınmadı.",
        "",
    ]
    for target, desc, k in CLASSIFICATION:
        out.append(f"**`{target}`** — {desc}, {k} sınıf · şans seviyesi **{1.0 / k:.3f}**")
        out.append("")
        out.append("| iter | gözlem | gözlem+z_s | Δ (z_s) | gözlem+z_t | Δ (z_t) | `g` tek başına |")
        out.append("|---:|---:|---:|---:|---:|---:|---:|")
        for it, doc in docs:
            row = [f"| {it:,}".replace(",", "."), f"| {fmt(bal_acc(doc, 'obs45', target))}"]
            for latent in ("z_s", "z_t"):
                row.append(f"| {fmt(bal_acc(doc, f'obs45+{latent}', target))}")
                row.append(f"| {fmt_pm(stored_delta_bal(doc, latent, target))}")
            row.append(f"| {fmt(bal_acc(doc, 'g', target))}")
            out.append(" ".join(row) + " |")
        out.append("")
    return "\n".join(out)


def section_controls(docs) -> str:
    out = ["| iter | karıştırılmış etiket ΔR² | eğitilmemiş encoder ΔR² | kapı |", "|---:|---:|---:|:--|"]
    for it, doc in docs:
        c = doc.get("controls", {})
        thr = c.get("threshold", 0.05)
        vals = []
        for key in ("shuffled_label_friction", "random_init_latent_friction"):
            d = c.get(key, {}).get("delta_r2_vs_obs45", {})
            vals.append((d.get("mean"), d.get("std")))
        gate = "PASS" if all(v[0] is not None and v[0] <= thr for v in vals) else "FAIL"
        cells = " ".join(
            f"| {m:+.4f} ± {s:.4f}" if m is not None else "| —" for m, s in vals
        )
        out.append(f"| {it:,}".replace(",", ".") + f" {cells} | **{gate}** |")
    return "\n".join(out)


def section_ood(docs) -> str:
    out = ["| hedef | dağılım-içi R² (z_s) | görülmemiş seviyede R² (z_s) | n |", "|---|---:|---:|---:|"]
    doc = docs[-1][1]
    for target, desc in PHYSICS:
        try:
            node = doc["features"]["z_s"]["regression"][target]
            ood = node.get("ood_unseen_level", {})
        except KeyError:
            continue
        idv = r2(doc, "z_s", target)
        ov, n = ood.get("R2"), ood.get("n")
        if ov is None or (isinstance(ov, float) and math.isnan(ov)):
            continue
        out.append(f"| `{target}` | {fmt(idv)} | {fmt(float(ov))} | {n:,}".replace(",", ".") + " |")
    return "\n".join(out)


def section_intervention() -> str:
    """Recomputed from the raw intervention banks, not from probe_metrics.json."""
    base = os.path.join(EVAL, "moe_latent_probe_phaseA")
    out = []
    for metric, label, nd in [
        ("tracking_lin_err", "Doğrusal hız takip hatası (m/s, düşük = iyi)", 3),
        ("fall", "Düşme oranı (düşmüş adım payı, düşük = iyi)", 5),
    ]:
        out.append(f"**{label}**")
        out.append("")
        header = "| mod | " + " | ".join(d for _, d in INTERVENE) + " | satır/hücre |"
        out.append(header)
        out.append("|---|" + "---:|" * (len(INTERVENE) + 1))
        for mode in MODE_ORDER:
            cells, counts = [], []
            for cell, _ in INTERVENE:
                path = os.path.join(base, cell, "intervene.npz")
                if not os.path.exists(path):
                    cells.append("—")
                    continue
                z = np.load(path, allow_pickle=True)
                sel = z["mode"] == mode
                counts.append(int(sel.sum()))
                cells.append(f"{float(z[metric][sel].astype(np.float64).mean()):.{nd}f}" if sel.any() else "—")
            n = f"{counts[0]:,}".replace(",", ".") if counts else "—"
            out.append(f"| `{mode}` | " + " | ".join(cells) + f" | {n} |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--section",
        default="all",
        choices=["all", "physics", "state", "classification", "controls", "ood", "intervention"],
    )
    args = ap.parse_args()

    docs = [(it, load(rel)) for it, rel in CHECKPOINTS]
    want = args.section

    if want in ("all", "controls"):
        print("### Sızıntı kontrolleri\n")
        print(section_controls(docs), "\n")

    if want in ("all", "physics"):
        print(section_regression(
            docs, PHYSICS, derived_delta=False,
            heading="### Gizli fizik parametreleri",
            note="ΔR², hattın kendi hesabı: her fold'da `R²(gözlem+z) − R²(gözlem)` alınır, "
                 "sonra fold'lar ortalanır. ± değeri fold'lar arası standart sapmadır.",
        ))

    if want in ("all", "state"):
        print(section_regression(
            docs, STATE, derived_delta=True,
            heading="### Durum ve dinamik büyüklükler",
            note="⚠️ Hat bu hedefler için fold bazlı ΔR² kaydetmiyor. Buradaki ΔR² "
                 "fold ortalamalarının farkıdır (`R²(gözlem+z) − R²(gözlem)`), bu yüzden "
                 "fold sapması verilemiyor. Fizik tablosundaki ± ile aynı türden bir sayı değildir.",
        ))

    if want in ("all", "classification"):
        print("### Sınıflandırma hedefleri\n")
        print(section_classification(docs))

    if want in ("all", "ood"):
        print("### Dağılım dışı ekstrapolasyon (iter 23.500)\n")
        print(section_ood(docs), "\n")

    if want in ("all", "intervention"):
        print("### Nedensel müdahale\n")
        print(section_intervention())


if __name__ == "__main__":
    main()
