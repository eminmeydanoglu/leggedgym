#!/usr/bin/env python
"""Coverage-regime sweep for the stateless (Eq. 7) LP-ACRL curriculum.

Decides the frozen ``V5_BETA`` and whether the optional epsilon exploration
floor is needed at the V5 sampling budget.  Reproduces the exact softmax the
production ``advance`` runs (``_FiniteEpisodeCurriculum._softmax`` + the epsilon
mix) on the worst-case single-hot learning-progress vector: one cell with
LP=Delta, the remaining ``n-1`` cells imputed LP=0.

A cell is *observable* in a stage only if its sampling probability yields at
least a few completed episodes.  With ~``stage_episodes`` completions spread over
the task space, the coldest cell needs ``min_prob * stage_episodes >= COVER_MIN``
episodes to keep producing a return (and thus stay eligible for a fresh LP next
stage).  We report, per beta, the coldest-cell probability, its expected
episodes/stage, and the effective sample size (ESS); the smallest beta that
keeps coverage while still concentrating is the one to freeze.

Run:  .venv/bin/python tests/verify_coverage_regime.py
"""
from __future__ import annotations

import argparse

import numpy as np

from legged_gym.utils.ued import TaskSpace
from legged_gym.utils.ued.episode_curriculum import _FiniteEpisodeCurriculum

# A cell needs at least this many completed episodes/stage to be re-observed.
COVER_MIN = 3.0


def sweep(n: int, stage_episodes: int, deltas, betas, epsilon: float) -> None:
    print(
        f"task cells n={n}   stage_episodes~={stage_episodes}   "
        f"coverage floor=1/{stage_episodes}={1.0 / stage_episodes:.2e}   "
        f"epsilon={epsilon}\n"
        f"worst case: 1 hot cell (LP=Delta), {n - 1} imputed LP=0\n"
    )
    header = f"{'Delta':>6} {'beta':>5} {'min_prob':>11} {'cold_eps/stage':>15} {'ESS':>8} {'max_prob':>9}  coverage"
    for delta in deltas:
        print(header)
        for beta in betas:
            progress = np.zeros(n, dtype=np.float64)
            progress[0] = float(delta)  # lp_acrl score == progress (signed)
            weights = _FiniteEpisodeCurriculum._softmax(progress, float(beta))
            if epsilon > 0.0:
                weights = (1.0 - epsilon) * weights + epsilon / n
            min_prob = float(weights.min())
            max_prob = float(weights.max())
            ess = float(1.0 / np.sum(np.square(weights)))
            cold_eps = min_prob * stage_episodes
            ok = "OK" if cold_eps >= COVER_MIN else "STARVED"
            print(
                f"{delta:>6} {beta:>5} {min_prob:>11.3e} {cold_eps:>15.2f} "
                f"{ess:>8.2f} {max_prob:>9.3f}  {ok}"
            )
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--stage-episodes", type=int, default=8192,
                    help="completed episodes per stage (see go2_v5_config.py:44-51)")
    ap.add_argument("--epsilon", type=float, default=0.0)
    args = ap.parse_args()

    n = TaskSpace().size
    deltas = (15, 20, 25, 30)
    betas = (1, 2, 3, 4, 5, 8)
    sweep(n, args.stage_episodes, deltas, betas, args.epsilon)
    print(
        f"Decision rule: smallest beta with cold_eps/stage >= {COVER_MIN} across the\n"
        f"plausible Delta range AND ESS not collapsed to ~1 -> freeze as V5_BETA.\n"
        f"If no beta satisfies both, re-run with --epsilon 0.05 and pin the floor."
    )


if __name__ == "__main__":
    main()
