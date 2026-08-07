"""Metrics for the brake probe: what actually differs between Genesis and MuJoCo.

Consumes one or more ``brake_probe.py`` npz files and prints a comparison.  All
metrics live here rather than in the recorder so a definition can be revised
without re-running rollouts -- which matters, because the obvious definition is
wrong twice over:

* **Peak foot height is not the metric.**  A nominal single-rollout smoke had a
  HIGHER peak foot height in Genesis while the interesting event was twice as
  frequent in MuJoCo.  What the eye reads as "the rear legs came up" is both
  rear feet leaving the ground TOGETHER, which peak height does not measure.

* **The event has a nonzero baseline.**  A trot at 1.5-2 m/s already has flight
  phases; both rear feet are legitimately airborne together for part of every
  stride.  So the post-stop rate is meaningless on its own.  Every event metric
  here is reported next to the SAME metric measured on the pre-trigger cruise
  window, and the headline number is the excess of one over the other.

Two calibrations keep the backends comparable:

* **Foot height is calibrated per backend.**  ``feet_pos`` is the foot body
  frame origin, and that origin sits at a different height above the ground in
  the URDF Genesis loads than in the MJCF MuJoCo compiles (measured: ~0.021 m
  vs ~0.017 m resting).  Clearance is therefore measured against each backend's
  own observed resting height, not against z = 0.

* **Feet and joints are resolved by NAME.**  Link and DOF tables are built from
  different assets per backend; index 2 is not guaranteed to be the same leg.

Usage::

    python legged_gym/scripts/eval/brake_analyze.py \
        logs/eval/brake/genesis_student.npz logs/eval/brake/mujoco_student.npz

    # per-cell breakdown instead of the pooled summary
    python legged_gym/scripts/eval/brake_analyze.py --by vx ramp *.npz
"""

import argparse
import os

import numpy as np


# --------------------------------------------------------------------------- #
# Thresholds.  Defaults are the ones the protocol was specified with; each is a
# CLI flag because "is the conclusion threshold-dependent?" is a question the
# analysis has to be able to answer.
# --------------------------------------------------------------------------- #
DEF_CONTACT_N = 5.0      # |contact force| below this counts as "not in contact"
DEF_CLEARANCE_M = 0.03   # foot must ALSO be this far above its resting height
DEF_MIN_STEPS = 2        # ... for at least this many control steps (40 ms @ 50 Hz)
DEF_WINDOW_S = 1.0       # post-trigger measurement window


def quat_to_pitch_roll(q):
    """(..., 4) xyzw -> (pitch, roll) in radians.

    xyzw is the codebase convention (``simulator.base_quat``); pitch uses the
    clamped-asin form so a near-vertical base does not produce NaN.
    """
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    return pitch, roll


def runs_of_true(mask):
    """Lengths of every maximal run of True in a 1-D boolean array."""
    if not mask.any():
        return np.zeros(0, dtype=int)
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges[1::2] - edges[0::2]


def leg_groups(feet_names):
    """Indices of the rear and front foot pairs, by name.

    Go2 links are ``FL_foot`` / ``FR_foot`` / ``RL_foot`` / ``RR_foot``; the
    leading letter is the one that matters and the order is NOT assumed.
    """
    names = [str(n) for n in feet_names]
    rear = [i for i, n in enumerate(names) if n.startswith("R")]
    front = [i for i, n in enumerate(names) if n.startswith("F")]
    if len(rear) != 2 or len(front) != 2:
        raise SystemExit(f"cannot identify front/rear pairs from feet names {names}")
    return rear, front


def dof_group(dof_names, prefixes):
    return [i for i, n in enumerate(map(str, dof_names))
            if any(str(n).startswith(p) for p in prefixes)]


class BrakeRun:
    """One npz: derived per-condition metrics, plus the pre-trigger baselines."""

    def __init__(self, path, cfg):
        d = np.load(path, allow_pickle=True)
        self.path = path
        self.d = d
        self.cfg = cfg
        self.backend = str(d["backend"])
        self.arm = str(d["arm"])
        self.dt = float(d["dt"])
        self.trig = d["trigger_step"]
        self.vx = d["cond_vx"]
        self.ramp = d["cond_ramp"]
        self.phase = d["cond_phase"]
        self.n = len(self.trig)
        self.win = max(1, int(round(cfg.window_s / self.dt)))
        self.label = f"{self.backend}/{self.arm}"

        self.rear, self.front = leg_groups(d["feet_names"])
        self._prepare()
        self._measure()

    # -- shared derived signals ------------------------------------------- #
    def _prepare(self):
        d = self.d
        forces = d["feet_contact_forces"]                       # (T, N, 4, 3)
        self.fmag = np.linalg.norm(forces, axis=-1)             # (T, N, 4)
        self.fz = np.abs(forces[..., 2])
        self.ftan = np.linalg.norm(forces[..., :2], axis=-1)
        self.footz = d["feet_pos"][..., 2]                      # (T, N, 4)

        # Resting foot height, per backend, from the pre-trigger cruise stretch:
        # the 2nd percentile over every foot and every steady step is the stance
        # height, i.e. the offset between this asset's foot frame and the ground.
        pre = self._pre_mask()
        stance = self.footz[pre]                                # (K, 4) flattened
        self.rest_z = np.percentile(stance, 2.0, axis=0)        # per foot
        self.clearance = self.footz - self.rest_z[None, None, :]

        self.pitch, self.roll = quat_to_pitch_roll(d["base_quat"])
        self.pitch_rate = d["base_ang_vel"][..., 1]
        self.basez = d["base_pos"][..., 2]
        self.vxb = d["base_lin_vel"][..., 0]

        # "Off the ground" = force below threshold AND the foot actually lifted.
        # The force test alone fires on a lightly-loaded stance foot; the height
        # test alone fires on a foot skimming the ground under load.
        self.airborne = (self.fmag < self.cfg.contact_n) & \
                        (self.clearance > self.cfg.clearance_m)

    def _pre_mask(self):
        """(T, N) boolean: the steady-cruise steps before each env's trigger.

        The first few post-warmup steps are dropped so the baseline is measured
        on a settled stride, not on the tail of the episode-clock reset.
        """
        T = self.d["base_quat"].shape[0]
        t = np.arange(T)[:, None]
        return (t >= self.cfg.baseline_skip) & (t < self.trig[None, :])

    def _post_mask(self):
        """(T, N) boolean: ``window_s`` of steps starting at each env's trigger."""
        T = self.d["base_quat"].shape[0]
        t = np.arange(T)[:, None]
        return (t >= self.trig[None, :]) & (t < (self.trig + self.win)[None, :])

    # -- metrics ----------------------------------------------------------- #
    def _pair_event(self, pair, mask):
        """Per-condition (event_rate, duty, longest) for a simultaneous-pair lift.

        ``event_rate`` counts CONDITIONS in which the pair was simultaneously
        airborne for at least ``min_steps`` consecutive steps -- a run-length
        test, not an instant test, so a one-frame force dropout during a normal
        stance cannot register as a leg lift.
        ``duty`` is the fraction of the window with both feet airborne;
        ``longest`` is the longest such run in seconds.
        """
        both = self.airborne[:, :, pair].all(axis=2)            # (T, N)
        ev = np.zeros(self.n)
        duty = np.zeros(self.n)
        longest = np.zeros(self.n)
        for i in range(self.n):
            seg = both[mask[:, i], i]
            if seg.size == 0:
                continue
            duty[i] = seg.mean()
            r = runs_of_true(seg)
            if r.size:
                longest[i] = r.max() * self.dt
                ev[i] = float(r.max() >= self.cfg.min_steps)
        return ev, duty, longest

    def _window_reduce(self, arr, mask, fn):
        return np.asarray([fn(arr[mask[:, i], i]) if mask[:, i].any() else np.nan
                           for i in range(self.n)])

    def _measure(self):
        d = self.d
        post, pre = self._post_mask(), self._pre_mask()
        m = {}

        for tag, pair in (("rear", self.rear), ("front", self.front)):
            ev, duty, longest = self._pair_event(pair, post)
            m[f"{tag}_lift_rate"] = ev
            m[f"{tag}_lift_duty"] = duty
            m[f"{tag}_lift_longest_s"] = longest
            bev, bduty, _ = self._pair_event(pair, pre)
            m[f"{tag}_lift_rate_base"] = bev
            m[f"{tag}_lift_duty_base"] = bduty

        # Full flight (all four off) separates "the rear end came up" from
        # "the whole robot was in a flight phase of the gait".
        allfour = self.airborne.all(axis=2)
        m["flight_duty"] = self._window_reduce(allfour.astype(float), post, np.mean)
        m["flight_duty_base"] = self._window_reduce(allfour.astype(float), pre, np.mean)

        m["peak_pitch_deg"] = np.degrees(self._window_reduce(self.pitch, post, lambda a: np.abs(a).max()))
        m["peak_pitch_deg_base"] = np.degrees(self._window_reduce(self.pitch, pre, lambda a: np.abs(a).max()))
        m["peak_pitch_rate"] = self._window_reduce(self.pitch_rate, post, lambda a: np.abs(a).max())
        m["peak_pitch_rate_base"] = self._window_reduce(self.pitch_rate, pre, lambda a: np.abs(a).max())

        base_h = self._window_reduce(self.basez, pre, np.mean)
        m["base_h_overshoot"] = self._window_reduce(self.basez, post, np.max) - base_h
        m["base_h_undershoot"] = base_h - self._window_reduce(self.basez, post, np.min)

        m["rear_clearance_max"] = self._window_reduce(
            self.clearance[:, :, self.rear].max(axis=2), post, np.max)
        m["rear_clearance_max_base"] = self._window_reduce(
            self.clearance[:, :, self.rear].max(axis=2), pre, np.max)

        # Impulses: where the braking load is actually taken.  Normal (z) and
        # tangential (xy) are separated because a plant/contact-model difference
        # shows up as a different tangential/normal split, not just a bigger force.
        for tag, pair in (("rear", self.rear), ("front", self.front)):
            m[f"{tag}_impulse_n"] = self._window_reduce(
                self.fz[:, :, pair].sum(axis=2), post, np.sum) * self.dt
            m[f"{tag}_impulse_t"] = self._window_reduce(
                self.ftan[:, :, pair].sum(axis=2), post, np.sum) * self.dt

        # Rear joint extrema, by name.
        dof_names = d["dof_names"]
        if dof_names.size:
            for joint in ("hip", "thigh", "calf"):
                idx = dof_group(dof_names, ("RL_" + joint, "RR_" + joint))
                if not idx:
                    continue
                q = d["dof_pos"][:, :, idx]
                m[f"rear_{joint}_max"] = self._window_reduce(q.max(axis=2), post, np.max)
                m[f"rear_{joint}_min"] = self._window_reduce(q.min(axis=2), post, np.min)

        # Control effort.
        tl = d["torque_limits"]
        sat = (np.abs(d["torques"]) >= self.cfg.sat_frac * tl[None, None, :]).any(axis=2)
        m["torque_sat_duty"] = self._window_reduce(sat.astype(float), post, np.mean)
        arate = np.abs(np.diff(d["actions"], axis=0)).mean(axis=2)
        arate = np.concatenate([arate[:1], arate], axis=0)      # keep length T
        m["action_rate"] = self._window_reduce(arate, post, np.mean)

        m["settle_s"] = self._settling()
        m["fell"] = d["fell"][-1]
        m["cruise_vx"] = self._window_reduce(self.vxb, pre, np.mean)

        self.m = m

    def _settling(self):
        """Seconds from trigger until |vx| stays below ``stop_tol``.

        Right-censored at the end of the recorded window: an env that never
        settles is reported at the window length, so the mean is a lower bound
        and never silently drops the worst cases.
        """
        T = self.vxb.shape[0]
        out = np.full(self.n, np.nan)
        for i in range(self.n):
            g = int(self.trig[i])
            seg = np.abs(self.vxb[g:, i])
            below = seg < self.cfg.stop_tol
            stays = np.flip(np.minimum.accumulate(np.flip(below)))
            out[i] = (np.argmax(stays) if stays.any() else len(seg)) * self.dt
        return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
# (key, label, format, "excess" baseline key or None)
ROWS = [
    ("rear_lift_rate", "rear pair lift rate", "{:.2f}", "rear_lift_rate_base"),
    ("rear_lift_duty", "rear pair airborne duty", "{:.3f}", "rear_lift_duty_base"),
    ("rear_lift_longest_s", "longest rear lift [s]", "{:.3f}", None),
    ("front_lift_rate", "front pair lift rate", "{:.2f}", "front_lift_rate_base"),
    ("flight_duty", "all-four flight duty", "{:.3f}", "flight_duty_base"),
    ("rear_clearance_max", "max rear clearance [m]", "{:.3f}", "rear_clearance_max_base"),
    ("peak_pitch_deg", "peak |pitch| [deg]", "{:.2f}", "peak_pitch_deg_base"),
    ("peak_pitch_rate", "peak |pitch rate| [rad/s]", "{:.2f}", "peak_pitch_rate_base"),
    ("base_h_overshoot", "base height overshoot [m]", "{:.3f}", None),
    ("base_h_undershoot", "base height undershoot [m]", "{:.3f}", None),
    ("rear_impulse_n", "rear normal impulse [Ns]", "{:.1f}", None),
    ("rear_impulse_t", "rear tangential impulse [Ns]", "{:.1f}", None),
    ("front_impulse_n", "front normal impulse [Ns]", "{:.1f}", None),
    ("front_impulse_t", "front tangential impulse [Ns]", "{:.1f}", None),
    ("rear_thigh_max", "rear thigh max [rad]", "{:.2f}", None),
    ("rear_calf_min", "rear calf min [rad]", "{:.2f}", None),
    ("torque_sat_duty", "torque saturation duty", "{:.3f}", None),
    ("action_rate", "action rate", "{:.3f}", None),
    ("settle_s", "settling time [s]", "{:.2f}", None),
    ("cruise_vx", "cruise vx [m/s]", "{:.2f}", None),
    ("fell", "fall rate", "{:.2f}", None),
]


def summarize(runs, sel=None, title=""):
    """Print one table: rows = metrics, columns = runs."""
    print()
    print(f"### {title}" if title else "###")
    width = max(28, *(len(r.label) for r in runs)) if runs else 28
    head = "  ".join(f"{r.label:>18}" for r in runs)
    print(f"{'metric':<{width}}  {head}")
    print("-" * (width + 2 + len(head)))
    for key, label, fmt, base in ROWS:
        if not all(key in r.m for r in runs):
            continue
        cells = []
        for r in runs:
            idx = np.ones(r.n, dtype=bool) if sel is None else sel(r)
            v = np.nanmean(r.m[key][idx])
            cell = fmt.format(v)
            if base is not None and base in r.m:
                cell += f" ({fmt.format(v - np.nanmean(r.m[base][idx]))})"
            cells.append(f"{cell:>18}")
        print(f"{label:<{width}}  " + "  ".join(cells))
    n = [int((np.ones(r.n, dtype=bool) if sel is None else sel(r)).sum()) for r in runs]
    print(f"{'conditions':<{width}}  " + "  ".join(f"{x:>18}" for x in n))


def main():
    p = argparse.ArgumentParser(description="Brake probe metrics")
    p.add_argument("npz", nargs="+", help="brake_probe.py outputs to compare")
    p.add_argument("--window_s", type=float, default=DEF_WINDOW_S)
    p.add_argument("--contact_n", type=float, default=DEF_CONTACT_N)
    p.add_argument("--clearance_m", type=float, default=DEF_CLEARANCE_M)
    p.add_argument("--min_steps", type=int, default=DEF_MIN_STEPS)
    p.add_argument("--baseline_skip", type=int, default=5,
                   help="pre-trigger steps skipped before the baseline window")
    p.add_argument("--stop_tol", type=float, default=0.15,
                   help="|vx| below this counts as stopped [m/s]")
    p.add_argument("--sat_frac", type=float, default=0.95,
                   help="fraction of the torque limit that counts as saturated")
    p.add_argument("--by", nargs="*", default=[], choices=["vx", "ramp"],
                   help="also break the summary down by these grid axes")
    cfg = p.parse_args()

    runs = [BrakeRun(path, cfg) for path in cfg.npz]
    for r in runs:
        print(f"[{os.path.basename(r.path)}] backend={r.backend} arm={r.arm} "
              f"n={r.n} dt={r.dt} rest_z={np.round(r.rest_z, 4).tolist()}")

    print("\nNumbers in parentheses are the EXCESS over the same metric measured "
          f"on the pre-trigger cruise window\n(a trot already has flight phases). "
          f"Window = {cfg.window_s} s after the stop trigger.")
    summarize(runs, None, "pooled over the whole grid")

    if "vx" in cfg.by:
        for v in sorted(set(np.round(runs[0].vx, 3))):
            summarize(runs, lambda r, v=v: np.isclose(r.vx, v), f"vx = {v} m/s")
    if "ramp" in cfg.by:
        for v in sorted(set(np.round(runs[0].ramp, 3))):
            summarize(runs, lambda r, v=v: np.isclose(r.ramp, v), f"ramp = {v} s")
    print()


if __name__ == "__main__":
    main()
