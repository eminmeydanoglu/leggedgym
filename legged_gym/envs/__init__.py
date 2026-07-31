# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import *
from .base.legged_robot import LeggedRobot
from .base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
# k1
from legged_gym.envs.k1.k1 import K1Robot
from legged_gym.envs.k1.k1_config import K1Cfg, K1CfgPPO
# k1 motion visualization
from legged_gym.envs.k1.k1_motion_vis.k1_motion_vis import K1MotionVis
from legged_gym.envs.k1.k1_motion_vis.k1_motion_vis_config import K1MotionVisCfg
# k1 DeepMimic
from legged_gym.envs.k1.k1_deepmimic.k1_deepmimic import K1DeepMimic
from legged_gym.envs.k1.k1_deepmimic.k1_deepmimic_config import K1DeepMimicCfg, K1DeepMimicCfgPPO
# k1 AMP
from legged_gym.envs.k1.k1_amp.k1_amp import K1AMP
from legged_gym.envs.k1.k1_amp.k1_amp_config import K1AMPCfg, K1AMPCfgPPO
# k1 cts amp
from legged_gym.envs.k1.k1_cts_amp.k1_cts_amp import K1_CTS_AMP
from legged_gym.envs.k1.k1_cts_amp.k1_cts_amp_config import K1_CTS_AMPCfg, K1_CTS_AMPCfgPPO
# g1
from legged_gym.envs.g1.g1 import G1Robot
from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
# g1 motion visualization
from legged_gym.envs.g1.g1_motion_vis.g1_motion_vis import G1MotionVis
from legged_gym.envs.g1.g1_motion_vis.g1_motion_vis_config import G1MotionVisCfg
# g1 DeepMimic
from legged_gym.envs.g1.g1_deepmimic.g1_deepmimic import G1DeepMimic
from legged_gym.envs.g1.g1_deepmimic.g1_deepmimic_config import G1DeepMimicCfg, G1DeepMimicCfgPPO
# go2
from legged_gym.envs.go2.go2 import GO2
from legged_gym.envs.go2.go2_config import GO2Cfg, GO2CfgPPO
# go2 walk these ways
from legged_gym.envs.go2.go2_wtw.go2_wtw import GO2WTW
from legged_gym.envs.go2.go2_wtw.go2_wtw_config import GO2WTWCfg, GO2WTWCfgPPO
# go2_ts(teacher-student)
from legged_gym.envs.go2.go2_ts.go2_ts import Go2TS
from legged_gym.envs.go2.go2_ts.go2_ts_config import Go2TSCfg, Go2TSCfgPPO
# go2_ee(explicit estimator)
from legged_gym.envs.go2.go2_ee.go2_ee import Go2EE
from legged_gym.envs.go2.go2_ee.go2_ee_config import Go2EECfg, Go2EECfgPPO
# go2_cts(concurrent teacher-student)
from legged_gym.envs.go2.go2_cts.go2_cts import Go2CTS
from legged_gym.envs.go2.go2_cts.go2_cts_config import Go2CTSCfg, Go2CTSCfgPPO
# go2_dreamwaq
from legged_gym.envs.go2.go2_dreamwaq.go2_dreamwaq import Go2Dreamwaq
from legged_gym.envs.go2.go2_dreamwaq.go2_dreamwaq_config import Go2DreamwaqCfg, Go2DreamwaqCfgPPO
# go2_cat(constraint-as-termination)
from legged_gym.envs.go2.go2_cat.go2_cat import Go2CaT
from legged_gym.envs.go2.go2_cat.go2_cat_config import Go2CaTCfg, Go2CaTCfgPPO
# go2_ts_depth
from legged_gym.envs.go2.go2_ts_depth.go2_ts_depth import Go2TSDepth
from legged_gym.envs.go2.go2_ts_depth.go2_ts_depth_config import Go2TSDepthCfg, Go2TSDepthCfgPPO
# go2_nav
from legged_gym.envs.go2.go2_nav.go2_nav import GO2Nav
from legged_gym.envs.go2.go2_nav.go2_nav_config import GO2NavCfg, GO2NavCfgPPO
# go2_moects (go2_rl_gym MoE-CTS port: shared wty curriculum substrate,
# MoE-CTS and HIM arms swappable by task name only)
from legged_gym.envs.go2.go2_moects.go2_moects import Go2MoECTS
from legged_gym.envs.go2.go2_moects.go2_moects_him import Go2MoECTSHIM
from legged_gym.envs.go2.go2_moects.go2_moects_config import (
    Go2MoECTSCfg, Go2MoECTSCfgPPO, Go2MoECTSHIMCfg, Go2MoECTSHIMCfgPPO)

# tron1_pf
from legged_gym.envs.tron1pf.tron1pf import TRON1PF
from legged_gym.envs.tron1pf.tron1pf_config import TRON1PFCfg, TRON1PFCfgPPO
# tron_pf_ee
from legged_gym.envs.tron1pf.tron1pf_ee.tron1pf_ee import TRON1PF_EE
from legged_gym.envs.tron1pf.tron1pf_ee.tron1pf_ee_config import TRON1PF_EECfg, TRON1PF_EECfgPPO
# tron1_sf
from legged_gym.envs.tron1sf.tron1sf import TRON1SF
from legged_gym.envs.tron1sf.tron1sf_config import TRON1SFCfg, TRON1SFCfgPPO
# bipedal_walker
from legged_gym.envs.bipedal_walker.bipedal_walker_config import BipedalWalkerCfg, BipedalWalkerCfgPPO
from legged_gym.envs.bipedal_walker.bipedal_walker import BipedalWalker
# # go2_sysid
# from legged_gym.envs.go2.go2_sysid.go2_sysid import GO2SysID
# from legged_gym.envs.go2.go2_sysid.go2_sysid_config import GO2SysIDCfg


from legged_gym.utils.task_registry import task_registry

task_registry.register("k1", K1Robot, K1Cfg(), K1CfgPPO())
task_registry.register("k1_deepmimic", K1DeepMimic, K1DeepMimicCfg(), K1DeepMimicCfgPPO())
task_registry.register("k1_motion_vis", K1MotionVis, K1MotionVisCfg(), LeggedRobotCfgPPO()) # for motion visualization, not for training
task_registry.register("k1_amp", K1AMP, K1AMPCfg(), K1AMPCfgPPO())
task_registry.register("k1_cts_amp", K1_CTS_AMP, K1_CTS_AMPCfg(), K1_CTS_AMPCfgPPO()) # unvalidated
task_registry.register("g1", G1Robot, G1RoughCfg(), G1RoughCfgPPO())
task_registry.register("g1_deepmimic", G1DeepMimic, G1DeepMimicCfg(), G1DeepMimicCfgPPO())
task_registry.register("g1_motion_vis", G1MotionVis, G1MotionVisCfg(), LeggedRobotCfgPPO()) # for motion visualization, not for training
task_registry.register( "go2", GO2, GO2Cfg(), GO2CfgPPO())
task_registry.register( "go2_wtw", GO2WTW, GO2WTWCfg(), GO2WTWCfgPPO())
task_registry.register( "go2_ts", Go2TS, Go2TSCfg(), Go2TSCfgPPO())
task_registry.register( "go2_ee", Go2EE, Go2EECfg(), Go2EECfgPPO())
task_registry.register( "go2_cts", Go2CTS, Go2CTSCfg(), Go2CTSCfgPPO())
task_registry.register( "go2_dreamwaq", Go2Dreamwaq, Go2DreamwaqCfg(), Go2DreamwaqCfgPPO())
task_registry.register( "go2_cat", Go2CaT, Go2CaTCfg(), Go2CaTCfgPPO())
task_registry.register( "go2_ts_depth", Go2TSDepth, Go2TSDepthCfg(), Go2TSDepthCfgPPO()) # unvalidated
task_registry.register( "go2_nav", GO2Nav, GO2NavCfg(), GO2NavCfgPPO())
task_registry.register( "go2_moects", Go2MoECTS, Go2MoECTSCfg(), Go2MoECTSCfgPPO())
task_registry.register( "go2_moects_him", Go2MoECTSHIM, Go2MoECTSHIMCfg(), Go2MoECTSHIMCfgPPO())
task_registry.register( "tron1pf", TRON1PF, TRON1PFCfg(), TRON1PFCfgPPO())
task_registry.register( "tron1pf_ee", TRON1PF_EE, TRON1PF_EECfg(), TRON1PF_EECfgPPO())
task_registry.register( "tron1sf", TRON1SF, TRON1SFCfg(), TRON1SFCfgPPO())
# task_registry.register( "go2_sysid", GO2SysID, GO2SysIDCfg(), GO2CfgPPO())
# task_registry.register( "bipedal_walker", BipedalWalker, BipedalWalkerCfg(), BipedalWalkerCfgPPO())
# ---------------------------------------------------------------------------
# go2_benchmark: adaptation study family (frozen flat substrate + frozen DR,
# all inherit Go2BenchmarkCommonCfg). No-DR MLP (floor) / DR-MLP (baseline) /
# Oracle (ceiling). Added additively; existing "go2" (simple_rl) is untouched.
# ---------------------------------------------------------------------------
from legged_gym.envs.go2.go2_bench_mlp.go2_bench_mlp import Go2BenchMlp
from legged_gym.envs.go2.go2_bench_mlp.go2_bench_mlp_config import Go2BenchMlpCfg, Go2BenchCfgPPO
from legged_gym.envs.go2.go2_bench_nodr.go2_bench_nodr import Go2BenchNoDR
from legged_gym.envs.go2.go2_bench_nodr.go2_bench_nodr_config import Go2BenchNoDRCfg, Go2BenchNoDRCfgPPO
from legged_gym.envs.go2.go2_bench_oracle.go2_bench_oracle import Go2BenchOracle
from legged_gym.envs.go2.go2_bench_oracle.go2_bench_oracle_config import Go2BenchOracleCfg, Go2BenchOracleCfgPPO
from legged_gym.envs.go2.go2_bench_oracle_id.go2_bench_oracle_id import Go2BenchOracleID
from legged_gym.envs.go2.go2_bench_oracle_id.go2_bench_oracle_id_config import Go2BenchOracleIDCfg, Go2BenchOracleIDCfgPPO
# oracle_id + true base_lin_vel(3): oracle ceiling of what RMA/DreamWaQ/SysID estimate
from legged_gym.envs.go2.go2_bench_oracle_id_vel.go2_bench_oracle_id_vel import Go2BenchOracleIDVel
from legged_gym.envs.go2.go2_bench_oracle_id_vel.go2_bench_oracle_id_vel_config import Go2BenchOracleIDVelCfg, Go2BenchOracleIDVelCfgPPO
from legged_gym.envs.go2.go2_bench_mlp_wide.go2_bench_mlp_wide import Go2BenchMlpWide
from legged_gym.envs.go2.go2_bench_mlp_wide.go2_bench_mlp_wide_config import Go2BenchMlpWideCfg, Go2BenchMlpWideCfgPPO
# --- Wave-2 rich-P cell (pd_gain + ctrl_delay added to P); isolated from the
#     frozen 5-task factorial above, trained separately. ---
from legged_gym.envs.go2.go2_bench_mlp_rich.go2_bench_mlp_rich_config import Go2BenchMlpRichCfg, Go2BenchMlpRichCfgPPO
from legged_gym.envs.go2.go2_bench_oracle_rich.go2_bench_oracle_rich import Go2BenchOracleRich
from legged_gym.envs.go2.go2_bench_oracle_rich.go2_bench_oracle_rich_config import Go2BenchOracleRichCfg, Go2BenchOracleRichCfgPPO
# --- Phase 2: implicit/explicit adaptation methods rebound onto the frozen flat
#     benchmark substrate (Go2BenchmarkCommonCfg). RMA (implicit z), DreamWaQ
#     (VAE latent + explicit lin_vel), Explicit SysID (regress base_lin_vel + P). ---
from legged_gym.envs.go2.go2_bench_rma.go2_bench_rma import Go2BenchRMA
from legged_gym.envs.go2.go2_bench_rma.go2_bench_rma_config import Go2BenchRMACfg, Go2BenchRMACfgPPO
from legged_gym.envs.go2.go2_bench_dreamwaq.go2_bench_dreamwaq import Go2BenchDreamwaq
from legged_gym.envs.go2.go2_bench_dreamwaq.go2_bench_dreamwaq_config import Go2BenchDreamwaqCfg, Go2BenchDreamwaqCfgPPO
from legged_gym.envs.go2.go2_bench_sysid.go2_bench_sysid import Go2BenchSysID
from legged_gym.envs.go2.go2_bench_sysid.go2_bench_sysid_config import Go2BenchSysIDCfg, Go2BenchSysIDCfgPPO
# HIM (Hybrid Internal Model): explicit lin_vel + implicit SwAV latent. Unlike
# RMA/DreamWaQ it keeps the standard single-tensor contract, so it retains the
# fair command_schedule + Eval-V2 machinery from OnPolicyRunner.
from legged_gym.envs.go2.go2_bench_him.go2_bench_him import Go2BenchHIM
from legged_gym.envs.go2.go2_bench_him.go2_bench_him_config import Go2BenchHIMCfg, Go2BenchHIMCfgPPO
# ---------------------------------------------------------------------------
# V3 dynamic-physics campaign.  Separate task names preserve the frozen Wave-1
# family and make manifests/checkpoints unambiguous.
# ---------------------------------------------------------------------------
from legged_gym.envs.go2.go2_v3_mlp import Go2V3Mlp, Go2V3MlpCfg, Go2V3MlpCfgPPO
from legged_gym.envs.go2.go2_v3_sysid import Go2V3SysID, Go2V3SysIDCfg, Go2V3SysIDCfgPPO
from legged_gym.envs.go2.go2_v3_rma import Go2V3RMA, Go2V3RMACfg, Go2V3RMACfgPPO
from legged_gym.envs.go2.go2_v3_dreamwaq import (
    Go2V3Dreamwaq,
    Go2V3DreamwaqCfg,
    Go2V3DreamwaqCfgPPO,
)
from legged_gym.envs.go2.go2_v3_him_fixed import (
    Go2V3HIMFixed,
    Go2V3HIMFixedCfg,
    Go2V3HIMFixedCfgPPO,
)
from legged_gym.envs.go2.go2_v3_superset_oracle import (
    Go2V3SupersetOracle,
    Go2V3SupersetOracleCfg,
    Go2V3SupersetOracleCfgPPO,
)
from legged_gym.envs.go2.go2_v4_config import (
    Go2V4MlpCfg, Go2V4MlpCfgPPO,
    Go2V4SysIDCfg, Go2V4SysIDCfgPPO,
    Go2V4RMACfg, Go2V4RMACfgPPO,
    Go2V4DreamwaqCfg, Go2V4DreamwaqCfgPPO,
    Go2V4HIMFixedCfg, Go2V4HIMFixedCfgPPO,
    Go2V4SupersetOracleCfg, Go2V4SupersetOracleCfgPPO,
)
# ---------------------------------------------------------------------------
# V5 UED: same frozen V4 physics/reward/DR/actor-critic/budget substrate
# (§2), reused Go2V3Mlp env implementation; only the episode-task sampler
# (curriculum.algorithm) differs across the four arms (§3, §14.5).
# ---------------------------------------------------------------------------
from legged_gym.envs.go2.go2_v5_config import (
    Go2V5HandcraftedCfg, Go2V5HandcraftedCfgPPO,
    Go2V5UniformCfg, Go2V5UniformCfgPPO,
    Go2V5LPACRLCfg, Go2V5LPACRLCfgPPO,
    Go2V5ALPCfg, Go2V5ALPCfgPPO,
)
from legged_gym.envs.go2.go2_v6_frontier_config import (
    Go2V6FrontierCfg, Go2V6FrontierCfgPPO,
    Go2V6FrontierSupersetOracleCfg, Go2V6FrontierSupersetOracleCfgPPO,
    Go2V6FrontierDreamwaqCfg, Go2V6FrontierDreamwaqCfgPPO,
    Go2V6FrontierHIMFixedCfg, Go2V6FrontierHIMFixedCfgPPO,
)
from legged_gym.envs.go2.go2_v7_lpacrl_him_config import (
    Go2V7FlatLPACRLHIMCfg,
    Go2V7FlatLPACRLHIMCfgPPO,
    Go2V7LPACRLHIMCfg,
    Go2V7LPACRLHIMCfgPPO,
    Go2V7UniformHIMCfg,
    Go2V7UniformHIMCfgPPO,
)

task_registry.register("go2_bench_mlp",    Go2BenchMlp,    Go2BenchMlpCfg(),    Go2BenchCfgPPO())
task_registry.register("go2_bench_nodr",   Go2BenchNoDR,   Go2BenchNoDRCfg(),   Go2BenchNoDRCfgPPO())
task_registry.register("go2_bench_oracle", Go2BenchOracle, Go2BenchOracleCfg(), Go2BenchOracleCfgPPO())
task_registry.register("go2_bench_oracle_id", Go2BenchOracleID, Go2BenchOracleIDCfg(), Go2BenchOracleIDCfgPPO())
# oracle_id + true base_lin_vel appended to the obs (matched to oracle_id otherwise)
task_registry.register("go2_bench_oracle_id_vel", Go2BenchOracleIDVel, Go2BenchOracleIDVelCfg(), Go2BenchOracleIDVelCfgPPO())
task_registry.register("go2_bench_mlp_wide", Go2BenchMlpWide, Go2BenchMlpWideCfg(), Go2BenchMlpWideCfgPPO())
# mlp_rich reuses the plain Go2BenchMlp env (no P); only its config differs.
task_registry.register("go2_bench_mlp_rich", Go2BenchMlp, Go2BenchMlpRichCfg(), Go2BenchMlpRichCfgPPO())
task_registry.register("go2_bench_oracle_rich", Go2BenchOracleRich, Go2BenchOracleRichCfg(), Go2BenchOracleRichCfgPPO())
# Phase 2: implicit/explicit adaptation methods on the frozen flat substrate
task_registry.register("go2_bench_rma",      Go2BenchRMA,      Go2BenchRMACfg(),      Go2BenchRMACfgPPO())
task_registry.register("go2_bench_dreamwaq", Go2BenchDreamwaq, Go2BenchDreamwaqCfg(), Go2BenchDreamwaqCfgPPO())
task_registry.register("go2_bench_sysid",    Go2BenchSysID,    Go2BenchSysIDCfg(),    Go2BenchSysIDCfgPPO())
task_registry.register("go2_bench_him",      Go2BenchHIM,      Go2BenchHIMCfg(),      Go2BenchHIMCfgPPO())
task_registry.register("go2_v3_mlp", Go2V3Mlp, Go2V3MlpCfg(), Go2V3MlpCfgPPO())
task_registry.register("go2_v3_sysid", Go2V3SysID, Go2V3SysIDCfg(), Go2V3SysIDCfgPPO())
task_registry.register("go2_v3_rma", Go2V3RMA, Go2V3RMACfg(), Go2V3RMACfgPPO())
task_registry.register(
    "go2_v3_dreamwaq",
    Go2V3Dreamwaq,
    Go2V3DreamwaqCfg(),
    Go2V3DreamwaqCfgPPO(),
)
task_registry.register(
    "go2_v3_him_fixed",
    Go2V3HIMFixed,
    Go2V3HIMFixedCfg(),
    Go2V3HIMFixedCfgPPO(),
)
task_registry.register(
    "go2_v3_superset_oracle",
    Go2V3SupersetOracle,
    Go2V3SupersetOracleCfg(),
    Go2V3SupersetOracleCfgPPO(),
)
# V4 reuses V3 method implementations but pins all runs to the same medium
# terrain row.  Only its oracle receives the privileged 17x11 terrain map.
task_registry.register("go2_v4_mlp", Go2V3Mlp, Go2V4MlpCfg(), Go2V4MlpCfgPPO())
task_registry.register("go2_v4_sysid", Go2V3SysID, Go2V4SysIDCfg(), Go2V4SysIDCfgPPO())
task_registry.register("go2_v4_rma", Go2V3RMA, Go2V4RMACfg(), Go2V4RMACfgPPO())
task_registry.register("go2_v4_dreamwaq", Go2V3Dreamwaq, Go2V4DreamwaqCfg(), Go2V4DreamwaqCfgPPO())
task_registry.register("go2_v4_him_fixed", Go2V3HIMFixed, Go2V4HIMFixedCfg(), Go2V4HIMFixedCfgPPO())
task_registry.register(
    "go2_v4_superset_oracle",
    Go2V3SupersetOracle,
    Go2V4SupersetOracleCfg(),
    Go2V4SupersetOracleCfgPPO(),
)
# V5 UED arms: same Go2V3Mlp env implementation (45D noisy proprio / 48D
# privileged critic, no height map) as go2_v4_mlp; only the cfg (episode-task
# sampler vs. legacy V4 curriculum) differs across arms.
task_registry.register("go2_v5_handcrafted", Go2V3Mlp, Go2V5HandcraftedCfg(), Go2V5HandcraftedCfgPPO())
task_registry.register("go2_v5_uniform", Go2V3Mlp, Go2V5UniformCfg(), Go2V5UniformCfgPPO())
task_registry.register("go2_v5_lpacrl", Go2V3Mlp, Go2V5LPACRLCfg(), Go2V5LPACRLCfgPPO())
task_registry.register("go2_v5_alp", Go2V3Mlp, Go2V5ALPCfg(), Go2V5ALPCfgPPO())
# V6 frontier arms: shared frontier curriculum substrate (10x10 V4 terrain +
# success-gated sampler); each arm reuses the matching V3 env implementation
# and only swaps the observation / PPO contract via config.
task_registry.register(
    "go2_v6_frontier",
    Go2V3Mlp,
    Go2V6FrontierCfg(),
    Go2V6FrontierCfgPPO(),
)
task_registry.register(
    "go2_v6_frontier_mlp",
    Go2V3Mlp,
    Go2V6FrontierCfg(),
    Go2V6FrontierCfgPPO(),
)
task_registry.register(
    "go2_v6_frontier_oracle",
    Go2V3SupersetOracle,
    Go2V6FrontierSupersetOracleCfg(),
    Go2V6FrontierSupersetOracleCfgPPO(),
)
task_registry.register(
    "go2_v6_frontier_dreamwaq",
    Go2V3Dreamwaq,
    Go2V6FrontierDreamwaqCfg(),
    Go2V6FrontierDreamwaqCfgPPO(),
)
task_registry.register(
    "go2_v6_frontier_him",
    Go2V3HIMFixed,
    Go2V6FrontierHIMFixedCfg(),
    Go2V6FrontierHIMFixedCfgPPO(),
)
task_registry.register(
    "go2_v7_lpacrl_him",
    Go2V3HIMFixed,
    Go2V7LPACRLHIMCfg(),
    Go2V7LPACRLHIMCfgPPO(),
)
task_registry.register(
    "go2_v7_uniform_him",
    Go2V3HIMFixed,
    Go2V7UniformHIMCfg(),
    Go2V7UniformHIMCfgPPO(),
)
task_registry.register(
    "go2_v7_flat_lpacrl_him",
    Go2V3HIMFixed,
    Go2V7FlatLPACRLHIMCfg(),
    Go2V7FlatLPACRLHIMCfgPPO(),
)
