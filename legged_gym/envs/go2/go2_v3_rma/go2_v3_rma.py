"""V3 dynamic-physics variant of the deployed RMA student task."""

import torch

from legged_gym.envs.go2.go2_bench_rma.go2_bench_rma import Go2BenchRMA
from legged_gym.envs.go2.go2_v3_physics import Go2V3PhysicsResampleMixin


class Go2V3RMA(Go2V3PhysicsResampleMixin, Go2BenchRMA):
    """RMA with the V3 dynamic physics and an opt-in V4 teacher terrain map."""

    def compute_observations(self):
        # Keep V3's exact actor/history/critic contract.  V4 only extends the
        # privileged teacher input, so the student must still recover terrain
        # effects from its proprioceptive history rather than receiving a map.
        super().compute_observations()
        if getattr(self.cfg.env, "rma_teacher_height_map", False):
            height_map = torch.clip(
                self.simulator.base_pos[:, 2].unsqueeze(1) - 0.5
                - self.simulator.measured_heights,
                -1.0,
                1.0,
            ) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat(
                (self.privileged_obs_buf, height_map), dim=-1
            )

        expected = self.num_privileged_obs
        actual = self.privileged_obs_buf.shape[1]
        if actual != expected:
            raise RuntimeError(
                f"RMA teacher input mismatch: expected {expected}, got {actual}"
            )
