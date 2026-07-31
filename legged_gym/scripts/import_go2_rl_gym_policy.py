"""Convert a published go2_rl_gym deploy policy into a go2_moects checkpoint.

The go2_rl_gym release ships TorchScript *deploy* policies (deploy/pre_train/go2/
*.pt), not training checkpoints: a traced module carrying only the two parts the
robot needs at inference time -- the student MoE encoder and the actor -- plus an
internal history ring buffer and an (Identity) observation normalizer. There is
no teacher encoder, no critic and no optimizer state in them.

That is exactly the subset ``ActorCriticMoECTS.act_student`` uses, so the policy
can be replayed here. This script rewrites the traced parameter names into our
module layout and emits a normal host checkpoint (model_state_dict / iter / ...)
that play.py and the eval harness load through the usual resume path. The
teacher encoder and critic stay at their fresh initialization -- they are never
touched by the student inference path -- so the result is for PLAYBACK AND
EVALUATION ONLY, never for resuming training.

Two export layouts ship in the release and both are handled (see _FLAT_MAP /
_NESTED_MAP below); ``--verify`` re-runs the traced module's own submodules on
random inputs and asserts our ``act_student`` reproduces them. Both published
MoE policies match to 0.0 exactly.

Joint order IS handled. go2_rl_gym trained in IsaacGym with the URDF order
(FL, FR, RL, RR -- see deploy/deploy_mujoco/configs/go2.yaml model_joint_names),
while this repo's go2 asset lists dofs in real-robot order (FR, FL, RR, RL).
Feeding our observations to the reference weights unpermuted swaps the left and
right legs, which reads as a stiff, stumbling gait that collapses on spawn. The
conversion therefore permutes the affected weight columns/rows so the resulting
checkpoint is native to OUR dof order; ``--no-dof-permute`` disables it.

What is still NOT checked here, because it lives outside the weights: the
environment must present observations the way go2_rl_gym did -- same 45-D
layout and scales, same 5-frame oldest-first history, same gains and action
scale. The go2_moects config was ported for that parity.

Usage:
    python legged_gym/scripts/import_go2_rl_gym_policy.py \
        --src /path/to/go2_rl_gym/deploy/pre_train/go2/go2_moe_cts_137k_0.6739.pt \
        --run wty_137k --verify

Then replay it with the normal flags, e.g.
    python legged_gym/scripts/play.py --task=go2_moects --load_run wty_137k
"""

import argparse
import os

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.go2.go2_moects.go2_moects_config import Go2MoECTSCfg, Go2MoECTSCfgPPO
from rsl_rl.modules import ActorCriticMoECTS


# The release ships two export layouts. Both describe the same network; they
# differ only in how the traced module happened to nest its submodules.
#
# "flat" (go2_moe_cts_137k, the paper default) comes from the moe_ng variant,
# which spells the expert stack out as experts_backbone / experts_hidden /
# experts_out and exports the actor as a bare nn.Sequential.
_FLAT_MAP = {
    "student_moe_encoder.experts_backbone.0": "history_encoder.moe.experts.backbone.network.0",
    "student_moe_encoder.experts_backbone.2": "history_encoder.moe.experts.backbone.network.2",
    "student_moe_encoder.experts_hidden.0":   "history_encoder.moe.experts.backbone.network.4",
    "student_moe_encoder.experts_out":        "history_encoder.moe.experts.experts",
    "student_moe_encoder.gating_network.0":   "history_encoder.moe.gating_network.0.network.0",
    "student_moe_encoder.gating_network.2":   "history_encoder.moe.gating_network.0.network.2",
    "student_moe_encoder.gating_network.4":   "history_encoder.moe.gating_network.0.network.4",
    "actor.0": "actor.0", "actor.2": "actor.2", "actor.4": "actor.4", "actor.6": "actor.6",
}
# "nested" (go2_moe_cts_high_slope_thre_164k) comes from the plain moe_cts
# variant -- the exact module tree moe_utils.py was ported from, so every
# encoder suffix is already ours; only the two roots differ.
_NESTED_MAP = {
    **{f"student_moe_encoder.moe.{s}": f"history_encoder.moe.{s}" for s in (
        "experts.backbone.network.0", "experts.backbone.network.2",
        "experts.backbone.network.4", "experts.experts",
        "gating_network.0.network.0", "gating_network.0.network.2",
        "gating_network.0.network.4")},
    **{f"actor.network.{i}": f"actor.{i}" for i in (0, 2, 4, 6)},
}


# go2_rl_gym's IsaacGym dof order, copied from its deploy config
# (deploy/deploy_mujoco/configs/go2.yaml, model_joint_names).
_REFERENCE_DOF_NAMES = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)
# Offsets of the per-dof blocks inside one 45-D observation frame
# [ang_vel(3), gravity(3), commands(3), dof_pos(12), dof_vel(12), actions(12)].
_OBS_DOF_BLOCK_OFFSETS = (9, 21, 33)


def dof_permutation():
    """``perm[j] = reference index of our dof j`` (None when already aligned).

    Our 12-D vectors are in ``cfg.asset.dof_names`` order, the reference weights
    were trained in ``_REFERENCE_DOF_NAMES`` order. A Linear that consumed
    reference-ordered inputs accepts ours after ``W[:, perm]``; one that emitted
    reference-ordered outputs emits ours after ``W[perm, :]``. Gathering a
    reference-ordered *vector* out of one of ours instead needs the inverse
    (``inverse_permutation``) -- for go2 the two happen to coincide, but the
    call sites below spell out which one they mean rather than relying on that.
    """
    ours = list(Go2MoECTSCfg.asset.dof_names)
    if len(ours) != len(_REFERENCE_DOF_NAMES):
        raise ValueError(f"expected 12 dofs, got {len(ours)}")
    missing = set(ours) ^ set(_REFERENCE_DOF_NAMES)
    if missing:
        raise ValueError(f"dof name sets differ; unmatched: {sorted(missing)}")
    perm = [_REFERENCE_DOF_NAMES.index(name) for name in ours]
    return None if perm == list(range(len(perm))) else perm


def inverse_permutation(perm):
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def _permute_input_blocks(weight, perm, offsets):
    """Gather ``weight[..., perm]`` inside each 12-wide dof block.

    Used both on weights (last dim = input columns) and, in ``verify``, on
    observation vectors (last dim = features).
    """
    out = weight.clone()
    idx = torch.as_tensor(perm, device=weight.device)
    for off in offsets:
        out[:, off:off + len(perm)] = weight[:, off + idx]
    return out


def _frame_offsets(latent_dim, num_frames, frame_width=45):
    """Dof-block column offsets of a ``latent_dim``-prefixed frame stack."""
    return tuple(latent_dim + f * frame_width + off
                 for f in range(num_frames)
                 for off in _OBS_DOF_BLOCK_OFFSETS)


def apply_dof_permutation(converted, perm):
    """Rewrite the converted tensors from reference dof order into ours.

    Only three places touch dof-ordered data: the actor's first layer (its obs
    slice), the actor's last layer (it emits per-dof actions), and the two
    encoder entry layers (experts backbone + gating, both reading the flattened
    history). Everything in between is order-agnostic.
    """
    latent_dim = Go2MoECTSCfg.env.num_latent_dims
    num_frames = Go2MoECTSCfg.env.frame_stack

    # Actor input is cat([latent, obs]) -- one frame, offset past the latent.
    actor_in = "actor.0.weight"
    converted[actor_in] = _permute_input_blocks(
        converted[actor_in], perm, _frame_offsets(latent_dim, 1))

    # Actor output rows are the 12 actions.
    idx = torch.as_tensor(perm, device=converted["actor.6.weight"].device)
    converted["actor.6.weight"] = converted["actor.6.weight"][idx]
    converted["actor.6.bias"] = converted["actor.6.bias"][idx]

    # Encoder inputs are the flattened 5-frame history (no latent prefix).
    for key in ("history_encoder.moe.experts.backbone.network.0.weight",
                "history_encoder.moe.gating_network.0.network.0.weight"):
        converted[key] = _permute_input_blocks(
            converted[key], perm, _frame_offsets(0, num_frames))
    return converted


def _select_param_map(traced_sd):
    """Pick the layout the traced module was exported with."""
    if "student_moe_encoder.experts_backbone.0.weight" in traced_sd:
        return _FLAT_MAP, "flat (moe_ng export)"
    if "student_moe_encoder.moe.experts.backbone.network.0.weight" in traced_sd:
        return _NESTED_MAP, "nested (moe_cts export)"
    roots = sorted({k.split(".")[0] for k in traced_sd})
    raise ValueError(
        f"Unrecognised policy layout (top-level modules: {roots}). This script "
        "converts go2_rl_gym MoE-CTS deploy policies; the plain CTS baseline "
        "(go2_cts_*.pt, 'student_encoder' MLP) belongs to the go2_cts task, not "
        "go2_moects.")


def build_actor_critic(device="cpu"):
    """Fresh ActorCriticMoECTS with the go2_moects task dimensions."""
    env_cfg, ppo_cfg = Go2MoECTSCfg.env, Go2MoECTSCfgPPO.policy
    return ActorCriticMoECTS(
        env_cfg.num_observations,
        env_cfg.num_actions,
        env_cfg.num_privileged_obs,
        env_cfg.num_history_obs,
        env_cfg.num_latent_dims,
        env_cfg.num_critic_obs,
        actor_hidden_dims=ppo_cfg.actor_hidden_dims,
        critic_hidden_dims=ppo_cfg.critic_hidden_dims,
        privilege_encoder_hidden_dims=ppo_cfg.privilege_encoder_hidden_dims,
        expert_num=ppo_cfg.expert_num,
        student_encoder_hidden_dims=ppo_cfg.student_encoder_hidden_dims,
        norm_type=ppo_cfg.norm_type,
        init_noise_std=ppo_cfg.init_noise_std,
    ).to(device)


def convert(src_path, device="cpu", permute_dofs=True):
    """Return (actor_critic, traced_module, layout, perm) with weights loaded."""
    traced = torch.jit.load(src_path, map_location=device)
    traced_sd = traced.state_dict()

    # The reference exporter emits an Identity normalizer when training used no
    # empirical normalization. Anything else would silently shift every input.
    if any(k.startswith("normalizer.") for k in traced_sd):
        raise ValueError(
            "Source policy carries a non-Identity observation normalizer; the "
            "go2_moects env applies no such normalization. Port it first.")

    param_map, layout = _select_param_map(traced_sd)
    print(f"source layout: {layout}")

    if layout.startswith("flat"):
        # moe_ng can feed the gating network a command-masked history while the
        # experts see the full one. Equal input widths mean the mask was
        # all-True, i.e. it collapses to the plain MoE we ported.
        gating_in = traced_sd["student_moe_encoder.gating_network.0.weight"].shape[1]
        experts_in = traced_sd["student_moe_encoder.experts_backbone.0.weight"].shape[1]
        if gating_in != experts_in:
            raise ValueError(
                f"Source policy is a real 'no goal' variant (gating input {gating_in} "
                f"!= experts input {experts_in}): its gating network sees a "
                "command-masked history, which the ported MoE does not model.")

    actor_critic = build_actor_critic(device)
    our_sd = actor_critic.state_dict()

    converted = {}
    for src_prefix, dst_prefix in param_map.items():
        for suffix in (".weight", ".bias"):
            src_key, dst_key = src_prefix + suffix, dst_prefix + suffix
            if src_key not in traced_sd:
                raise KeyError(f"Source policy is missing '{src_key}'")
            if dst_key not in our_sd:
                raise KeyError(f"ActorCriticMoECTS has no '{dst_key}'")
            src_t, dst_t = traced_sd[src_key], our_sd[dst_key]
            if src_t.shape != dst_t.shape:
                raise ValueError(
                    f"Shape mismatch {src_key} {tuple(src_t.shape)} -> "
                    f"{dst_key} {tuple(dst_t.shape)}")
            converted[dst_key] = src_t.clone()

    unmapped = set(traced_sd) - {k for p in param_map for k in
                                 (p + ".weight", p + ".bias")}
    if unmapped:
        raise ValueError(f"Unmapped parameters in source policy: {sorted(unmapped)}")

    perm = dof_permutation() if permute_dofs else None
    if perm is not None:
        print(f"dof permutation (ours -> reference index): {perm}")
        apply_dof_permutation(converted, perm)
    elif permute_dofs:
        print("dof order already matches the reference; no permutation needed")
    else:
        print("WARNING: --no-dof-permute: weights keep the reference dof order")

    # Overlay onto the fresh state so the teacher encoder / critic / std keep
    # valid (random) tensors and load_state_dict stays strict.
    our_sd.update(converted)
    actor_critic.load_state_dict(our_sd, strict=True)
    return actor_critic, traced, layout, perm


def verify(actor_critic, traced, layout, perm=None, device="cpu", batch=16, tol=1e-4):
    """Compare our act_student against the traced module's own submodules.

    With a dof permutation applied, equality is up to relabelling: our network
    fed OUR-order inputs must reproduce the reference fed the same data in
    REFERENCE order, with its output mapped back to our order.
    """
    num_obs = Go2MoECTSCfg.env.num_observations
    num_hist = Go2MoECTSCfg.env.num_history_obs
    num_frames = Go2MoECTSCfg.env.frame_stack
    torch.manual_seed(0)
    obs = torch.randn(batch, num_obs, device=device)
    history = torch.randn(batch, num_hist, device=device)

    ref_obs, ref_history = obs, history
    if perm is not None:
        idx = torch.as_tensor(perm, device=device)
        # Build the reference-ordered view of the same data: our dof j lives at
        # reference slot perm[j], so gathering needs the inverse of perm.
        to_ref = inverse_permutation(perm)
        ref_obs = _permute_input_blocks(obs, to_ref, _frame_offsets(0, 1))
        ref_history = _permute_input_blocks(
            history, to_ref, _frame_offsets(0, num_frames))

    actor_critic.eval()
    with torch.no_grad():
        ours = actor_critic.act_student(obs, history)
        # Reference forward, spelled out: gating and experts both see the full
        # history (mask all-True), latent first in the actor concat.
        if layout.startswith("flat"):
            # moe_ng signature: (history, history_no_goal); mask is all-True here.
            ref_latent, _ = traced.student_moe_encoder(ref_history, ref_history)
        else:
            ref_latent, _ = traced.student_moe_encoder(ref_history)
        theirs = traced.actor(torch.cat([ref_latent, ref_obs], dim=1))
        if perm is not None:
            theirs = theirs[:, idx]

    # Judge on RELATIVE error. Reordering a matmul's inputs changes its float32
    # accumulation order, so a correct permutation still drifts by ~1e-5 on a
    # 225-wide input -- and by a different amount on different hardware. A wiring
    # error is not a near miss: swapping legs moves outputs by O(1), i.e. a
    # relative error near 1, so the two failure modes are orders of magnitude
    # apart and this threshold never has to split hairs.
    max_err = (ours - theirs).abs().max().item()
    scale = max(theirs.abs().max().item(), 1e-6)
    rel_err = max_err / scale
    ok = rel_err < tol
    print(f"over {batch} random inputs: max |ours - reference| = {max_err:.3e}, "
          f"max |reference| = {scale:.3e}, relative = {rel_err:.3e} "
          f"({'MATCH' if ok else 'MISMATCH'})")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True,
                        help="go2_rl_gym deploy policy (.pt, TorchScript)")
    parser.add_argument("--run", default="go2_rl_gym_import",
                        help="load_run directory name under logs/go2_moects")
    parser.add_argument("--iter", type=int, default=0,
                        help="iteration number baked into the checkpoint filename")
    parser.add_argument("--verify", action="store_true",
                        help="numerically compare against the source policy")
    parser.add_argument("--no-dof-permute", dest="dof_permute", action="store_false",
                        help="keep the reference (IsaacGym FL/FR/RL/RR) joint "
                             "order instead of remapping to cfg.asset.dof_names")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    actor_critic, traced, layout, perm = convert(
        args.src, args.device, permute_dofs=args.dof_permute)
    if args.verify and not verify(actor_critic, traced, layout, perm, args.device):
        raise SystemExit("verification failed; refusing to write the checkpoint")

    out_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs",
                           Go2MoECTSCfgPPO.runner.experiment_name, args.run)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"model_{args.iter}.pt")
    torch.save({
        "model_state_dict": actor_critic.state_dict(),
        "optimizer_state_dict": None,
        "iter": args.iter,
        "iteration_semantics": "completed_updates_v2",
        "infos": {
            "source": os.path.abspath(args.src),
            "provenance": "go2_rl_gym TorchScript deploy policy (actor + student "
                          "MoE encoder only); teacher encoder and critic are "
                          "randomly initialized -- PLAYBACK/EVAL ONLY, do not resume "
                          "training from this checkpoint",
            "dof_permutation": perm,
        },
    }, out_path)
    print(f"wrote {out_path}")
    print(f"replay with: python legged_gym/scripts/play.py --task=go2_moects "
          f"--load_run {args.run} --checkpoint {args.iter}")


if __name__ == "__main__":
    main()
