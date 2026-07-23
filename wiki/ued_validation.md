---
name: ued_validation
desc: Frozen V5 UED validation-bank and offline SPNTE checkpoint-selection contract.
created: 2026-07-23T09:02:39Z
updated: 2026-07-23T09:02:39Z
---

# ued_validation

`configs/eval/v5_ued.yaml` freezes a 4,032-replica bank: 84 terrain/velocity
cells with 48 deterministic in-bin commands each. Its canonical fingerprint,
terrain geometry hashes, validation seed `31001`, held-out seed `41001`, and
linear SPNTE support scale are stored with every result.

`legged_gym.scripts.eval.ued_validation` is simulator-neutral. A terrain
caller supplies one measurement per precomputed replica, including the command
and geometry hash it actually used. It rejects duplicate, missing, mismatched,
or out-of-range measurements before equally weighting the 84 cell means.

`legged_gym.scripts.eval.select_checkpoint` considers only existing
`model_<iteration>.pt` files with a matching complete-bank JSON artifact. It
selects minimum macro SPNTE; only scores within `1e-6` use, in order, worst-10%
CVaR, fall rate, success rate, and earlier iteration. It writes `best_spnte.pt`
without replacing `best_tracking.pt`, retains checkpoint/resume metadata, and
keeps assignment distribution distinct from PPO transition occupancy.
