"""Probe (NOT a pytest test): WHO contacts the rear calf links (RL_calf /
RR_calf) when Genesis reports nonzero net contact force on them while the
go2_moects robot just stands / jitters on flat ground?

Motivation: tests/_measure_collision_force_noise.py found that 6 of 8
penalized links read exactly 0.0 N always, but the two rear calves show
frequent nonzero forces (p99.9 = 26 N, max 142 N) with zero env resets.
Deciding the collision-reward threshold requires knowing whether those are
  (a) genuine ground contacts (the reward SHOULD fire), or
  (b) self-collision artifacts (PhysX reference would never see them).

Method: rebuild the flat env, stand (zero actions) then jitter, and on every
step where a rear calf reads > 0.5 N, pull `robot.get_contacts()` and record
for each rear-calf contact: partner link name, force magnitude, contact
position z, and whether the same-side foot is loaded at that moment.

Run:
    SIMULATOR=genesis .venv/bin/python tests/_probe_rear_calf_contacts.py
"""

import json
import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from _measure_collision_force_noise import make_flat_env  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
OUT_JSON = os.path.join(ROOT, "tmp", "rear_calf_contact_probe.json")

SETTLE = 100
PROBE_STEPS_STAND = 400
PROBE_STEPS_JITTER = 400
FORCE_EVENT_MIN = 0.5  # [N] rear-calf net force that counts as an event

# penalized_bodies_force_norm column order (from measurement run):
# [FL_thigh, FR_thigh, RL_thigh, RR_thigh, FL_calf, FR_calf, RL_calf, RR_calf]
REAR_CALF_COLS = {"RL_calf": 6, "RR_calf": 7}
# feet_force_norm column order: [FL_foot, FR_foot, RL_foot, RR_foot]
REAR_FOOT_COLS = {"RL_calf": 2, "RR_calf": 3}


def probe(env, action_fn, n_steps, agg):
    sim = env.simulator
    robot = sim._robot
    link_start = robot.link_start
    all_links = sim._scene.rigid_solver.links
    rear_global = {name: (sim.penalized_contact_indices[col] + link_start)
                   for name, col in REAR_CALF_COLS.items()}
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    for step in range(n_steps):
        actions = action_fn(step, actions)
        env.step(actions)
        pen = env.penalized_bodies_force_norm            # (N, 8) on GPU
        rear = pen[:, [REAR_CALF_COLS["RL_calf"], REAR_CALF_COLS["RR_calf"]]]
        event_envs = (rear.max(dim=1).values > FORCE_EVENT_MIN).nonzero(
            as_tuple=False).flatten()
        agg["steps_total"] += 1
        if len(event_envs) == 0:
            continue
        agg["steps_with_event"] += 1
        agg["env_ids"].update(int(i) for i in event_envs.cpu())
        contacts = robot.get_contacts()
        link_a = contacts["link_a"]        # (N, C)
        link_b = contacts["link_b"]
        force = contacts["force_b"]        # (N, C, 3) force on geom B
        pos = contacts["position"]         # (N, C, 3)
        valid = contacts["valid_mask"]     # (N, C)
        feet = env.feet_force_norm         # (N, 4)
        for env_id in event_envs.cpu().tolist():
            for name, gidx in rear_global.items():
                if rear[env_id, 0 if name == "RL_calf" else 1] <= FORCE_EVENT_MIN:
                    continue
                hit = valid[env_id] & ((link_a[env_id] == gidx)
                                       | (link_b[env_id] == gidx))
                idxs = hit.nonzero(as_tuple=False).flatten()
                agg["events"][name] += 1
                foot_loaded = bool(feet[env_id, REAR_FOOT_COLS[name]] > 1.0)
                agg["foot_loaded_during_event"][name] += int(foot_loaded)
                if len(idxs) == 0:
                    agg["no_contact_record_found"][name] += 1
                    continue
                for c in idxs.cpu().tolist():
                    a, b = int(link_a[env_id, c]), int(link_b[env_id, c])
                    partner = b if a == gidx else a
                    partner_name = all_links[partner].name
                    fvec = force[env_id, c]
                    if partner == gidx:      # degenerate same-link guard
                        continue
                    # force applied to the calf geom specifically
                    calf_f = fvec if b == gidx else -fvec
                    fnorm = float(calf_f.norm())
                    agg["partner_counts"][name][partner_name] += 1
                    agg["partner_force_sum"][name][partner_name] += fnorm
                    agg["pos_z"].append(float(pos[env_id, c, 2]))
                    agg["force_norms"].append(fnorm)
                    agg["force_z_frac"].append(
                        float(calf_f[2] / (calf_f.norm() + 1e-9)))
    return agg


def main():
    env = make_flat_env()
    agg = {
        "steps_total": 0, "steps_with_event": 0, "env_ids": set(),
        "events": Counter(), "foot_loaded_during_event": Counter(),
        "no_contact_record_found": Counter(),
        "partner_counts": {n: Counter() for n in REAR_CALF_COLS},
        "partner_force_sum": {n: Counter() for n in REAR_CALF_COLS},
        "pos_z": [], "force_norms": [], "force_z_frac": [],
    }
    zero = lambda step, prev: torch.zeros_like(prev)

    def jitter(step, prev):
        if step % 10 == 0:
            return torch.rand_like(prev) * 0.4 - 0.2
        return prev

    env.reset()
    probe(env, zero, SETTLE, agg)          # settle (aggregated, harmless)
    for phase, fn in (("stand", zero), ("jitter", jitter)):
        before = agg["steps_total"]
        probe(env, fn, PROBE_STEPS_STAND if phase == "stand"
              else PROBE_STEPS_JITTER, agg)
        print(f"phase {phase}: {agg['steps_total'] - before} steps probed, "
              f"events so far: {dict(agg['events'])}")

    out = {
        "force_event_min_N": FORCE_EVENT_MIN,
        "steps_total": agg["steps_total"],
        "steps_with_event": agg["steps_with_event"],
        "num_envs_with_events": len(agg["env_ids"]),
        "events_per_link": dict(agg["events"]),
        "foot_loaded_during_event": dict(agg["foot_loaded_during_event"]),
        "no_contact_record_found": dict(agg["no_contact_record_found"]),
        "partner_counts": {k: dict(v) for k, v in agg["partner_counts"].items()},
        "partner_force_sum_N": {k: dict(v)
                                for k, v in agg["partner_force_sum"].items()},
    }
    for key, vals in (("pos_z", agg["pos_z"]), ("force_norms", agg["force_norms"]),
                      ("force_z_frac", agg["force_z_frac"])):
        if vals:
            t = torch.tensor(vals, dtype=torch.float64)
            out[key] = {"n": len(vals), "mean": float(t.mean()),
                        "p10": float(torch.quantile(t, 0.10)),
                        "p50": float(torch.quantile(t, 0.50)),
                        "p90": float(torch.quantile(t, 0.90)),
                        "min": float(t.min()), "max": float(t.max())}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nsaved -> {OUT_JSON}")
    if hasattr(env, "destroy"):
        env.destroy()


if __name__ == "__main__":
    sys.exit(main())
