---
name: ued_genesis
desc: Genesis adapter and static teleport-grid contract for UED task assignments.
created: 2026-07-23T09:03:58Z
updated: 2026-07-23T09:03:58Z
---

# ued_genesis

The UED training terrain is a deterministic static 4x6 Genesis grid.  Its
physical tiles use the shared taxonomy geometry, while its logical task support
is five four-level terrain families plus flat at level zero: 21 terrain
configurations times four velocity bins equals 84 task IDs.  The flat tiles at
higher rows exist only to retain `[level, type]` origin addressing and are not
sampled by `TaskSpace`.

`GenesisUEDAdapter.assign` changes only per-environment terrain type, level,
origin, task/revision, and velocity-bin tensors.  It never rebuilds the
heightfield.  Reset must observe the completed old assignment before sampling
and applying the new one; commands are then sampled inside the active bin and
the root is teleported to the selected origin.

Standstill rollouts remain usable by PPO but are marked invalid for curriculum
observation.  Existing tasks retain their legacy batch-wide standstill draw and
command curriculum unless their new opt-in flags change those behaviors.
