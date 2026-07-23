---
name: spnte_eval
desc: First-fall SPNTE contract for V3 and V4 evaluation artifacts.
created: 2026-07-23T08:27:33Z
updated: 2026-07-23T08:27:33Z
---

# spnte_eval

`MetricAccumulator` now reports `spnte_lin`, `spnte_yaw`, and `first_fall_step` alongside the existing metrics. It keeps `auto_reset=True`: legacy tracking, returns, and fall metrics continue consuming the full stream, while SPNTE freezes each environment at its first `done & ~time_out` and fills its remaining fixed rollout horizon with maximum error.

V3 fixed-rollout and payload-switch artifacts persist `spnte_v_scale` and `spnte_yaw_scale`. The switch path feeds the same accumulator with x-only tracking error, so a reset episode cannot improve a dynamic-payload score. The linear scale is the maximum absolute endpoint of the active `lin_vel_x` command support, and yaw is derived symmetrically. A zero-only support uses float32 epsilon only to keep the all-zero-command calculation finite.

V3/V4 raw aggregation and `scorecard.md` display SPNTE as an append-only cross-check. Existing tracking/fall scorecard calculations are intentionally unchanged; future V5 checkpoint selection can consume the persisted SPNTE fields without changing historical V4 results.
## Validation

Evaluation imports require `SIMULATOR=genesis`. Under that documented setting, the focused SPNTE tests and V3/V4 evaluation regressions pass. The V2 resume fixture currently omits `OnPolicyRunner.device`, although its unchanged checkpoint loader uses that attribute for `map_location`; the same three fixture failures reproduce from the unchanged `main` archive and are outside the SPNTE path.
