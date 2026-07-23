---
name: ued_teacher
desc: Pure NumPy UED teacher interface and frozen behavior.
created: 2026-07-23T08:29:29Z
updated: 2026-07-23T08:29:29Z
---

The UED teacher core lives in `legged_gym/utils/ued` and is intentionally
independent of simulator, tensor, policy, and PPO code.

`TaskSpace` owns the canonical 84 moving tasks: five terrain types with four
levels, one single-level flat type, and four forward-velocity bins. Integer task
IDs are stable; `TaskSpec` is immutable; `decode_batch` returns arrays for
terrain type, terrain level, velocity-bin index, and bin bounds. Its fingerprint
covers terrain labels, velocity edges, and supplied deterministic-builder
parameters.

`UniformEpisodeCurriculum`, `LPACRLEpisodeCurriculum`, and
`ALPEpisodeCurriculum` implement the shared `EpisodeCurriculum` protocol:
`sample(count, *, global_control_steps)`, `observe(outcomes)`,
`advance(global_control_steps)`, `probabilities()`, `diagnostics()`,
`state_dict()`, and `load_state_dict(state)`. Their hot-path payloads are
`TaskAssignmentBatch` and `EpisodeOutcomeBatch`; assignments preserve task ID,
sampler revision, stage, drawn probability, and source label (`bootstrap`, `lp`,
or `alp`).

Stages are delimited only by global control steps. Only valid outcomes affect
per-task returns. The first observed stage establishes the baseline; later
adjacent observations yield signed LP for LP-ACRL or absolute LP for ALP. Cells
without a valid adjacent measurement retain their prior probability instead of
receiving an invented return. Softmax uses float64 max-shifting and rejects
non-finite values. Each teacher owns a PCG64 generator, and schema-v1 state
persists all arrays, counters, provenance occupancy, fingerprints, and RNG
state for deterministic continuation.

The Genesis adapter is the next consumer of this contract. It must create
outcomes from the old assignment before sampling a replacement and must not
place standstill episodes into this moving-task teacher.

Task identity is exact: scalar task coordinates and IDs must be integers, and
the TaskSpace snapshots finite JSON-compatible builder parameters when it is
created. Its fingerprint therefore cannot be changed by later caller mutation.
Public batches likewise reject lossy revision, length, and validity coercions.

Softmax subtracts the float64 maximum before scaling by beta, so finite extreme
learning-progress inputs remain a defined distribution. Checkpoint loading
validates scalar and vector schema types, probability normalization, source
provenance, and PCG64 state before applying any state; a rejected checkpoint
does not partially mutate a teacher.

The UED module sources import only standard-library and NumPy dependencies.
Importing it through `legged_gym.utils.ued` still executes the repository's
top-level `legged_gym` package initializer, which requires a configured
simulator on Python 3.11. Topic 03 runs inside Genesis, but standalone tooling
must account for that package-level boundary.
