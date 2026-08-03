# Terrain-conditioned MoE gating telemetry (go2_moects TELEMETRY-ONLY feature).
#
# Answers "does the student MoE encoder's gating network route differently
# depending on terrain type?" by bucketing a snapshot of gating weights
# (StudentMoEEncoder.forward_with_weights) by the env's semantic terrain id
# (WtyCurriculumMixin.wty_terrain_ids) and comparing each terrain's mean gate
# profile to the global mean profile via Jensen-Shannon divergence.
#
# Pure math only -- no simulator, no nn.Module -- so it is unit-testable on
# synthetic (weights, terrain_ids) without a GPU or Genesis. The runner
# (rsl_rl/runners/moe_cts_runner.py) is the only caller that wires this to
# live env/policy state, and it does so entirely under torch.no_grad(): this
# module must never be a site of RNG consumption or autograd-graph
# construction, since the caller's contract (bit-identical training with the
# feature on vs off) depends on that.

from typing import Dict, Optional, Sequence

import torch

# Semantic terrain id order (vendored go2_rl_gym terrain.py / make_moe_terrain,
# see legged_gym/utils/terrain.py:793-794 and
# legged_gym/envs/go2/go2_moects/wty_curriculum_mixin.py wty_terrain_ids).
# Index i is the semantic id written into wty_terrain_ids for column type i.
TERRAIN_NAMES = (
    "wave", "slope", "rough_slope", "stairs_up", "stairs_down",
    "obstacles", "stepping_stones", "gap", "flat",
)


def _js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """Jensen-Shannon divergence between two discrete distributions, in nats.

    p, q: 1-D tensors over the same simplex (nonnegative, same length; do not
    need to be exactly normalized, but are expected to sum to ~1 -- gating
    weights and their per-terrain/global means always do).

    Uses torch.xlogy so p_i == 0 (or q_i == 0) contributes exactly 0 to its
    term instead of raising/NaN-ing on log(0) -- the standard 0*log(0) := 0
    convention for entropy-like sums. m = 0.5*(p+q) is strictly positive
    everywhere p or q is positive, so log(m) is only ever evaluated where the
    xlogy weight is nonzero.
    """
    m = 0.5 * (p + q)
    kl_p = (torch.xlogy(p, p) - torch.xlogy(p, m)).sum()
    kl_q = (torch.xlogy(q, q) - torch.xlogy(q, m)).sum()
    js = 0.5 * (kl_p + kl_q)
    # JS divergence is mathematically >= 0; clamp away fp-noise negatives
    # (e.g. -1e-9) so the metric never prints as a small negative number.
    return float(js.clamp(min=0.0))


def compute_terrain_gate_stats(
    weights: torch.Tensor,
    terrain_ids: torch.Tensor,
    terrain_names: Sequence[str] = TERRAIN_NAMES,
) -> Optional[Dict]:
    """Bucket gating weights by terrain id and summarize per-terrain routing.

    Args:
        weights: (N, E) softmax gating weights, one row per sample (rows sum
            to ~1). Any grad-tracking is irrelevant -- this function never
            calls .backward() and detaches defensively -- but callers should
            still run the forward pass that produced these under
            torch.no_grad() to avoid building an unused autograd graph.
        terrain_ids: (N,) integer tensor, semantic terrain id per sample.
            Samples whose id falls outside [0, len(terrain_names)) are
            dropped (defensive: never raises on unexpected ids).
        terrain_names: id -> name lookup, default TERRAIN_NAMES.

    Returns:
        None if there are no valid samples (nothing to report -- caller
        no-ops). Otherwise a dict:
          {
            "per_terrain": {
                name: {"count": int, "mean_gate": (E,) tensor,
                       "entropy": float, "max_weight": float},
                ...  # only terrain types with count > 0
            },
            "global_mean_gate": (E,) tensor,   # mean over ALL valid samples
            "specialization": float,           # env-share-weighted mean JS
                                                # divergence (nats) of each
                                                # populated terrain's profile
                                                # vs the global profile
          }
    """
    if weights is None or terrain_ids is None:
        return None
    if weights.numel() == 0 or terrain_ids.numel() == 0:
        return None
    if weights.shape[0] != terrain_ids.shape[0]:
        return None

    weights = weights.detach()
    terrain_ids = terrain_ids.detach().to(torch.long)
    num_types = len(terrain_names)

    valid = (terrain_ids >= 0) & (terrain_ids < num_types)
    if not bool(valid.any()):
        return None
    if not bool(valid.all()):
        weights = weights[valid]
        terrain_ids = terrain_ids[valid]

    device = weights.device
    dtype = weights.dtype
    num_samples, num_experts = weights.shape

    counts = torch.bincount(terrain_ids, minlength=num_types).to(dtype=dtype, device=device)
    gate_sums = torch.zeros(num_types, num_experts, dtype=dtype, device=device)
    gate_sums.index_add_(0, terrain_ids, weights)

    per_sample_entropy = -torch.xlogy(weights, weights).sum(dim=-1)
    entropy_sums = torch.zeros(num_types, dtype=dtype, device=device)
    entropy_sums.index_add_(0, terrain_ids, per_sample_entropy)

    per_sample_max = weights.max(dim=-1).values
    max_sums = torch.zeros(num_types, dtype=dtype, device=device)
    max_sums.index_add_(0, terrain_ids, per_sample_max)

    global_mean_gate = weights.mean(dim=0)
    total = float(num_samples)

    per_terrain: Dict[str, Dict] = {}
    specialization = 0.0
    for tid in range(num_types):
        c = float(counts[tid].item())
        if c <= 0.0:
            continue
        mean_gate = gate_sums[tid] / c
        name = terrain_names[tid]
        per_terrain[name] = {
            "count": int(c),
            "mean_gate": mean_gate,
            "entropy": float((entropy_sums[tid] / c).item()),
            "max_weight": float((max_sums[tid] / c).item()),
        }
        js = _js_divergence(mean_gate, global_mean_gate)
        specialization += (c / total) * js

    if not per_terrain:
        return None

    return {
        "per_terrain": per_terrain,
        "global_mean_gate": global_mean_gate,
        "specialization": float(specialization),
    }
