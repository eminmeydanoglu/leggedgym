# Curriculum, command-resampling and reward machinery ported from go2_rl_gym
# (wty-yy, vendored IsaacGym fork of unitree_rl_gym, RSS 2026 MoE locomotion
# paper) onto the host LeggedRobot contract. Shared by both go2_moects arms
# (MoE-CTS and HIM): the mixin only WRAPS ``compute_observations`` (control-rate
# last_dof_vel bookkeeping), so each arm keeps its own observation contract.
#
# Vendored sources (go2_rl_gym/legged_gym/envs/base/legged_robot.py unless
# noted): check_termination (same-step contact termination; REPLACEMENT
# override, see the method docstring), _reset_dofs (multiplicative
# default*U(0.5,1.5); REPLACEMENT override), _reward_dof_acc +
# _wty_last_dof_vel control-rate tracking (post_physics_step tail; the host
# simulator's own last_dof_vel is per-SUBSTEP and feeds PD damping -- do not
# touch it), _update_terrain_curriculum,
# _get_env_origins (round-robin part),
# _update_env_command_ranges, _resample_commands, update_reward_curriculum /
# get_current_scale + compute_reward ramping, _get_dynamic_sigma +
# _reward_tracking_{lin,ang}_vel, _get_base_height + _reward_correct_base_height
# + _reward_feet_regulation, and _reward_hip_to_default from
# go2_rl_gym/legged_gym/envs/go2/go2_env.py.
#
# Buffer-name mapping (vendored -> host): root_states -> simulator.base_pos,
# contact_forces[feet] -> feet_force_norm, rigid_body_states[feet] ->
# simulator.feet_pos / simulator.feet_vel, torques -> simulator.torques,
# terrain / terrain_levels / terrain_types -> simulator._terrain /
# simulator.terrain_levels / simulator.terrain_types. Buffers the host does not
# track (commands_resampling_step, commands_xy_accumulation, max_move_distance,
# stop_heading, last_is_limit_vel, env_command_ranges) are maintained by the
# mixin itself in _init_buffers.

from itertools import product

import torch

from legged_gym.utils.math_utils import wrap_to_pi, quat_apply, torch_rand_float


def _sample_disjoint_intervals(env_ids, limit_bound, cfg_min, cfg_max, device):
    """Sample uniformly from [cfg_min, -limit_bound] U [limit_bound, cfg_max].

    Ported from go2_rl_gym/legged_gym/utils/isaacgym_utils.py.
    """
    width_neg = torch.nn.functional.relu(-limit_bound - cfg_min)
    width_pos = torch.nn.functional.relu(cfg_max - limit_bound)

    total_width = width_neg + width_pos + 1e-6  # epsilon against division by zero
    u = torch.rand(len(env_ids), device=device) * total_width

    samples = torch.where(
        u < width_neg,
        cfg_min + u,
        cfg_max - width_pos + (u - width_neg)
    )
    return samples


def _sample_single_interval(env_ids, cfg_min, cfg_max, device):
    """Sample uniformly from [cfg_min, cfg_max].

    Ported from go2_rl_gym/legged_gym/utils/isaacgym_utils.py.
    """
    r = torch.rand(len(env_ids), device=device)
    samples = cfg_min + r * (cfg_max - cfg_min)
    return samples


class WtyCurriculumMixin:
    """go2_rl_gym terrain-curriculum / command / reward machinery for host envs.

    Cooperative overrides (every method calls ``super()`` where a host method
    of the same name exists) with ONE exception: ``check_termination`` is a
    replacement override (no super() call) that restores the vendored
    same-step base-contact termination -- see its docstring. Place the mixin
    first in the MRO:

        class Go2MoECTS(WtyCurriculumMixin, LeggedRobotCTS): ...
        class Go2MoECTSHIM(WtyCurriculumMixin, Go2BenchHIM): ...

    The mixin is active only on a ``moe_grid`` terrain (the Phase-2 builder
    records ``cols2id``/``name2cols`` on the Terrain instance). On any other
    terrain it degrades to global command ranges, default tracking sigma and no
    terrain curriculum -- the defensive fallback required by the port plan.
    """

    # ------------------------------------------------------------------
    # Config / buffer setup
    # ------------------------------------------------------------------

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        # Iteration-based curricula count in PPO iterations; the vendored repo
        # hardcodes 24 (its runner's num_steps_per_env). Keep
        # cfg.env.wty_steps_per_iteration in sync with
        # CfgPPO.runner.num_steps_per_env.
        self.num_steps_per_iter = getattr(self.cfg.env, "wty_steps_per_iteration", 24)
        # Keep the CTS teacher/student split proportional when num_envs is
        # overridden (e.g. --num_envs on small GPUs): cfg.env.num_teacher is a
        # class-level constant computed for the cfg's own num_envs. Uses
        # cfg.env.num_envs (self.num_envs is not set yet at _parse_cfg time).
        # No-op for the HIM arm (no num_teacher attribute).
        if hasattr(self, "num_teacher"):
            self.num_teacher = int(self.cfg.env.num_envs
                                   * getattr(self.cfg.env, "teacher_env_ratio", 0.75))
        # [vendored _parse_cfg]: budget-fraction command range curriculum,
        # sorted descending so reached entries can be popped from the back.
        command_range_curriculum = getattr(self.cfg.commands, "command_range_curriculum", None)
        if command_range_curriculum:
            self.cfg.commands.command_range_curriculum = sorted(
                command_range_curriculum, key=lambda x: x["ratio"], reverse=True)
        # Planned total PPO iterations the ratio-based curricula
        # (zero_command / command_range / curriculum_rewards) resolve against.
        # The runner overrides this at learn() time with the effective
        # max_iterations via set_wty_total_iterations.
        self._wty_total_iterations = getattr(
            self.cfg.env, "curriculum_total_iterations", 30000)
        # Same-step base-contact termination threshold [N] (consumed by
        # check_termination below). Vendored go2_rl_gym hardcodes 1.0; the
        # moects cfgs set env.base_contact_terminate_threshold = 2.5 (see the
        # config comment for the rationale and the watch metric).
        self.base_contact_terminate_threshold = getattr(
            self.cfg.env, "base_contact_terminate_threshold", 2.5)

    def set_wty_total_iterations(self, total_iterations):
        """Override the planned total PPO iterations the ratio-based
        curricula (zero_command / command_range / curriculum_rewards) resolve
        against. Called by the runner at learn() time with the effective
        max_iterations (incl. any --max_iterations CLI override).

        This is also the resync point for a resumed run: it runs after
        runner.load() restored common_step_counter, and it is the first moment
        the ratio-based curricula can be evaluated against the *effective*
        budget, so the derived state is rebuilt here rather than at load time.
        """
        assert total_iterations and total_iterations > 0, (
            f"total_iterations must be positive, got {total_iterations}")
        self._wty_total_iterations = int(total_iterations)
        self._wty_resync_curriculum()

    def _wty_resync_curriculum(self):
        """Rebuild every curriculum quantity derived from progress.

        On a fresh run progress is 0 and this is a no-op restatement of the
        initial values. On a resumed run common_step_counter has just been
        restored to e.g. iteration 15000, and without this the command ranges
        would stay at the narrowest band until each env's first resample and
        the reward ramps would sit at start_value until the next
        num_steps_per_iter boundary.
        """
        self._apply_command_range_curriculum()
        self._update_reward_curriculum(force_update=True)
        self.zero_command_proba = self._get_current_scale(
            self.cfg.commands.zero_command_curriculum)

    def _wty_progress(self):
        """Training progress as a fraction of the planned budget: current
        PPO iteration / planned total iterations (0..1+).
        """
        current_iter = self.common_step_counter // self.num_steps_per_iter
        return current_iter / self._wty_total_iterations

    def _init_buffers(self):
        super()._init_buffers()
        # --- termination telemetry state (check_termination / reset_idx) ---
        # per-env flag for the reset reason of the current step; consumed by
        # reset_idx for the Episode/termination_* metrics. Recomputed every
        # check_termination call, so no manual clearing is needed.
        self.terminated_by_base_contact = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        # --- training telemetry accumulators (Episode/tracking_*,
        # Episode/torque_sat_*): per-env per-step sums / running max, emitted
        # and zeroed per finished episode by _write_episode_telemetry
        # (episode_sums-style window means). The terrain promote/demote counts
        # are python counters filled by _update_terrain_curriculum and reset
        # on each emit.
        self._tele_lin_vel_err_sum = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self._tele_ang_vel_err_sum = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self._tele_torque_sat_sum = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self._tele_torque_sat_max = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self._tele_terrain_promotions = 0
        self._tele_terrain_demotions = 0
        # --- vendored control-rate dof_vel tracking (dof_acc reward + obs
        # feature): go2_rl_gym updates last_dof_vel ONCE PER CONTROL STEP, at
        # the very end of post_physics_step (ref legged_robot.py:145), so the
        # vendored dof_acc spans a 20 ms window. The host simulator updates
        # its own _last_dof_vel every SIM SUBSTEP (5 ms, see
        # genesis_simulator.py:40) -- that buffer also feeds PD velocity
        # damping and must not be touched, so the mixin keeps its own
        # control-rate copy, refreshed in the compute_observations wrap below.
        self._wty_last_dof_vel = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float,
            device=self.device, requires_grad=False)
        # --- vendored command-resampling state (_init_buffers) ---
        self.commands_resampling_step = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_xy_accumulation = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.max_move_distance = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.stop_heading = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_is_limit_vel = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.zero_command_proba = 0.0
        self.max_lin_vel = max(
            abs(self.command_ranges["lin_vel_x"][0]), abs(self.command_ranges["lin_vel_x"][1]),
            abs(self.command_ranges["lin_vel_y"][0]), abs(self.command_ranges["lin_vel_y"][1]))
        self.limit_vel_prob = self.cfg.commands.limit_vel_prob
        self.limit_vel_comb = torch.tensor(list(product(
            self.cfg.commands.limit_vel["lin_vel_x"],
            self.cfg.commands.limit_vel["lin_vel_y"],
            self.cfg.commands.limit_vel["ang_vel_yaw"]
        )), device=self.device, requires_grad=False)
        # --- vendored reward-scale curriculum (__init__ / update_reward_curriculum) ---
        self.reward_curriculum_configs = getattr(self.cfg.rewards, "curriculum_rewards", None) or []
        self.reward_curriculum_scales = {
            config["reward_name"]: config["start_value"] for config in self.reward_curriculum_configs}
        # --- vendored dynamic tracking sigma (_create_envs) ---
        self.dynamic_sigma_cfg = getattr(self.cfg.rewards, "dynamic_sigma", None)
        if self.dynamic_sigma_cfg is not None:
            self.terrain_max_sigmas = torch.tensor(
                self.dynamic_sigma_cfg["max_sigma"], device=self.device, requires_grad=False)
        # --- vendored base-height scan mask (_init_buffers) ---
        # The mask selects the 0.4m x 0.3m window of the 17x11 height scan
        # directly under the base; ordering matches the simulator's
        # meshgrid(x, y, indexing='ij') height-point layout.
        if self.cfg.terrain.measure_heights:
            x_points = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device)
            y_points = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device)
            grid_x, grid_y = torch.meshgrid(x_points, y_points, indexing="ij")
            x_mask = (grid_x.flatten() >= -0.2) & (grid_x.flatten() <= 0.2)  # 0.4m length
            y_mask = (grid_y.flatten() >= -0.15) & (grid_y.flatten() <= 0.15)  # 0.3m width
            self.base_height_scan_mask = (x_mask & y_mask).float()
            self.num_base_height_scan_points = self.base_height_scan_mask.sum()
            assert self.num_base_height_scan_points > 0, "No height scan points within the specified area."
        # --- per-env command ranges (vendored env_command_ranges) ---
        self.env_command_ranges = {
            key: torch.tensor(value, device=self.device, dtype=torch.float,
                              requires_grad=False).repeat(self.num_envs, 1)
            for key, value in self.command_ranges.items()
        }
        # --- terrain curriculum assignment (vendored _get_env_origins port) ---
        self.wty_terrain_ids = None
        self._wty_name2cols = {}
        self._wty_curriculum_active = False
        self._wty_setup_terrain_curriculum()

    def _wty_setup_terrain_curriculum(self):
        """Round-robin terrain level/type assignment + semantic terrain ids.

        [vendored _get_env_origins]: the host simulator assigns random initial
        levels, while the vendored repo spreads envs round-robin over the
        initial levels and over all terrain columns, which the game-inspired
        curriculum and the per-terrain-type command limits rely on. Only
        active on a moe_grid terrain; otherwise the mixin stays inactive.
        """
        if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            return
        terrain = getattr(self.simulator, "_terrain", None)
        cols2id = getattr(terrain, "cols2id", None)
        if not cols2id or not hasattr(self.simulator, "_terrain_origins"):
            return  # defensive fallback: no moe_grid bookkeeping -> curriculum off
        # vendored: curriculum=True -> start on levels 0..max_init_terrain_level
        # (the mixin owns the curriculum, so cfg.terrain.curriculum stays False
        # and max_init_terrain_level applies unconditionally here).
        max_init_level = self.cfg.terrain.max_init_terrain_level
        self.simulator._terrain_levels = torch.fmod(
            torch.arange(self.num_envs, device=self.device), max_init_level + 1)
        self.simulator._terrain_types = torch.div(
            torch.arange(self.num_envs, device=self.device),
            (self.num_envs / self.cfg.terrain.num_cols), rounding_mode="floor").to(torch.long)
        # semantic terrain id per env: column index -> vendored terrain id
        terrain_cols2id = torch.tensor(cols2id, device=self.device)
        self.wty_terrain_ids = terrain_cols2id[self.simulator.terrain_types]
        self.simulator._env_origins[:] = self.simulator._terrain_origins[
            self.simulator.terrain_levels, self.simulator.terrain_types]
        self._wty_name2cols = {
            name: torch.tensor(sorted(cols), device=self.device)
            for name, cols in terrain.name2cols.items()
        }
        self._wty_curriculum_active = True
        self._update_env_command_ranges()

    # ------------------------------------------------------------------
    # Resume support (runner save/load env-curriculum protocol)
    # ------------------------------------------------------------------

    WTY_CURRICULUM_STATE_VERSION = 1

    def curriculum_state_dict(self):
        """Env curriculum state the runner persists with the weights.

        Only state that is NOT re-derivable from common_step_counter belongs
        here: the per-env terrain levels, which the promote/demote curriculum
        walks over training. Command ranges, reward ramp scales and the
        zero-command probability are all pure functions of progress and are
        rebuilt by _wty_resync_curriculum instead.

        Returns None when the terrain curriculum is inactive (flat/plane
        terrain, no moe_grid bookkeeping) so nothing is written for runs that
        have no such state.
        """
        if not self._wty_curriculum_active:
            return None
        return {
            "version": self.WTY_CURRICULUM_STATE_VERSION,
            "num_envs": int(self.num_envs),
            "num_cols": int(self.cfg.terrain.num_cols),
            "terrain_levels": self.simulator.terrain_levels.detach().cpu().clone(),
            "terrain_types": self.simulator.terrain_types.detach().cpu().clone(),
        }

    def load_curriculum_state_dict(self, state):
        """Restore the per-env terrain levels saved by curriculum_state_dict.

        Fails closed on a version or geometry mismatch: silently continuing
        with fresh round-robin levels is exactly the regression this protocol
        exists to prevent, and a level vector sized for a different env/terrain
        layout cannot be index-mapped onto this one.
        """
        if not self._wty_curriculum_active:
            print("[wty] terrain curriculum inactive; ignoring the checkpoint's "
                  "env_curriculum_state")
            return
        version = state.get("version")
        if version != self.WTY_CURRICULUM_STATE_VERSION:
            raise ValueError(
                f"env_curriculum_state version {version!r} != expected "
                f"{self.WTY_CURRICULUM_STATE_VERSION}")
        if int(state["num_envs"]) != int(self.num_envs) or \
                int(state["num_cols"]) != int(self.cfg.terrain.num_cols):
            raise ValueError(
                "env_curriculum_state geometry mismatch: checkpoint "
                f"num_envs={state['num_envs']} num_cols={state['num_cols']}, "
                f"env num_envs={self.num_envs} num_cols={self.cfg.terrain.num_cols}")
        # terrain_types is a deterministic function of (num_envs, num_cols), so
        # a mismatch here means the round-robin assignment itself changed and
        # the saved levels no longer describe the same envs.
        saved_types = state["terrain_types"].to(self.simulator.terrain_types.device)
        if not torch.equal(saved_types, self.simulator.terrain_types):
            raise ValueError(
                "env_curriculum_state terrain_types differ from this env's "
                "round-robin assignment; refusing to map saved levels onto it")
        levels = state["terrain_levels"].to(self.simulator._terrain_levels.device)
        self.simulator._terrain_levels[:] = levels.to(self.simulator._terrain_levels.dtype)
        # env origins are a lookup on (level, type); refresh so the next reset
        # spawns on the restored levels rather than the round-robin ones.
        self.simulator._env_origins[:] = self.simulator._terrain_origins[
            self.simulator.terrain_levels, self.simulator.terrain_types]
        print(f"[wty] restored terrain levels (mean "
              f"{self.simulator.terrain_levels.float().mean():.2f})")

    # ------------------------------------------------------------------
    # Per-step bookkeeping (vendored post_physics_step)
    # ------------------------------------------------------------------

    def _post_physics_step_callback(self):
        """[vendored post_physics_step, EARLY part]: per-env command-resample
        countdown + max-move-distance tracking only.

        The vendored resample/push CALLS no longer live here -- they run
        further down the host post_physics_step chain to match the vendored
        order (check_termination -> compute_reward -> RESAMPLE -> reset_idx
        -> PUSH -> compute_observations): see _resample_commands_if_due /
        _push_robots_if_due below. The host's fixed-period resample
        (episode_length % resampling_time) and every-step heading recompute
        are intentionally replaced: the vendored resampler owns per-env
        resampling and heading (stop_heading) logic.
        """
        self.commands_resampling_step -= 1
        # max xy distance from the env origin over the episode (drives the
        # terrain curriculum instead of the instantaneous distance).
        self.max_move_distance = self.max_move_distance.maximum(torch.norm(
            self.simulator.base_pos[:, :2] - self.simulator.env_origins[:, :2], dim=1))
        if self.cfg.domain_rand.push_links and (self.common_step_counter % self.cfg.domain_rand.push_links_interval == 0):
            self.simulator.push_links()
        # --- telemetry accumulation (emitted per episode in reset_idx): raw
        # tracking error against the UNSCALED command active during this step
        # (command resampling runs later, inside compute_reward), and torque
        # saturation |tau/tau_lim| (same buffers the privileged obs uses).
        self._tele_lin_vel_err_sum += torch.norm(
            self.commands[:, :2] - self.simulator.base_lin_vel[:, :2], dim=1)
        self._tele_ang_vel_err_sum += torch.abs(
            self.commands[:, 2] - self.simulator.base_ang_vel[:, 2])
        torque_sat = torch.abs(self.simulator.torques / self.simulator.torque_limits)
        self._tele_torque_sat_sum += torch.mean(torque_sat, dim=1)
        self._tele_torque_sat_max = torch.maximum(
            self._tele_torque_sat_max, torch.max(torque_sat, dim=1).values)

    def _resample_commands_if_due(self):
        """Vendored resample trigger (vendored post_physics_step: called AFTER
        compute_reward, so a resample-step reward is still scored against the
        OLD command); the guard skips the last step of an episode (vendored).
        Invoked at the end of the mixin's compute_reward.
        """
        resampling_env_ids = ((self.commands_resampling_step <= 0.0)
                              * (self.episode_length_buf < self.max_episode_length - 1)
                              ).nonzero(as_tuple=False).flatten()
        self._resample_commands(resampling_env_ids)

    def _push_robots_if_due(self):
        """Vendored per-env push trigger (vendored post_physics_step: called
        AFTER reset_idx and BEFORE compute_observations -- envs reset this
        step have episode_length_buf == 0 and therefore get their spawn
        velocity overwritten, exactly like the vendored incidental push).
        Invoked at the start of compute_observations (mixin wrap on the HIM
        arm, inline on the MoE arm).
        """
        if self.cfg.domain_rand.push_robots:
            # [vendored _push_robots]: per-env trigger on each env's own episode
            # clock (no global lockstep), and OVERWRITE of world-frame xy lin
            # vel U(+-max_push_vel_xy) + all ang vel U(+-max_push_ang_vel) in
            # the simulator, replacing the host's additive xy-only push.
            push_env_ids = (self.episode_length_buf % int(self.cfg.domain_rand.push_interval) == 0
                            ).nonzero(as_tuple=False).flatten()
            self.simulator.push_robots_overwrite(push_env_ids)

    # ------------------------------------------------------------------
    # Terrain curriculum (vendored _update_terrain_curriculum)
    # ------------------------------------------------------------------

    def _update_terrain_curriculum(self, env_ids):
        """[vendored _update_terrain_curriculum]: game-inspired curriculum.

        Promotion uses the episode's max move distance; demotion compares it
        against the accumulated xy command distance
        (move_down_by_accumulated_xy_command). The level update, max-level
        wraparound and env_origins reassignment live in the host simulator's
        update_terrain_curriculum (same semantics as the vendored inline code).
        """
        if not self.init_done or not self._wty_curriculum_active:
            # don't change on initial reset / without a moe_grid terrain
            return
        distance = self.max_move_distance[env_ids]
        # robots that walked far enough progress to harder terrains
        move_up = distance > self.simulator._terrain.env_length / 2
        if getattr(self.cfg.terrain, "move_down_by_accumulated_xy_command", False):
            move_down = (distance < torch.norm(self.commands_xy_accumulation[env_ids], dim=1)
                         * (self.cfg.commands.resampling_time * (1 - self.zero_command_proba)) * 0.5) * ~move_up
        else:
            # robots that walked less than half of their required distance go to simpler terrains
            move_down = (distance < torch.norm(
                self.commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5) * ~move_up
        # telemetry counters (emitted as Episode/terrain_promotions /
        # terrain_demotions by the next reset_idx, then reset)
        self._tele_terrain_promotions += int(move_up.sum())
        self._tele_terrain_demotions += int(move_down.sum())
        self.simulator.update_terrain_curriculum(env_ids, move_up, move_down)
        self.max_move_distance[env_ids] = 0.0

    # ------------------------------------------------------------------
    # Command resampling (vendored _resample_commands / _update_env_command_ranges)
    # ------------------------------------------------------------------

    def _update_env_command_ranges(self):
        """[vendored _update_env_command_ranges]: per-terrain-type command
        limits, clamped to the current global command ranges."""
        if self.wty_terrain_ids is None:
            self.env_command_ranges = {
                key: torch.tensor(value, device=self.device, dtype=torch.float,
                                  requires_grad=False).repeat(self.num_envs, 1)
                for key, value in self.command_ranges.items()
            }
            return
        for terrain_id, terrain_command_ranges in enumerate(self.cfg.commands.terrain_max_command_ranges):
            env_ids = (self.wty_terrain_ids == terrain_id).nonzero(as_tuple=False).flatten()
            if len(env_ids) == 0:
                continue
            self.env_command_ranges["lin_vel_x"][env_ids, 0] = max(
                terrain_command_ranges["lin_vel_x"][0],
                self.command_ranges["lin_vel_x"][0])
            self.env_command_ranges["lin_vel_x"][env_ids, 1] = min(
                terrain_command_ranges["lin_vel_x"][1],
                self.command_ranges["lin_vel_x"][1])
            self.env_command_ranges["lin_vel_y"][env_ids, 0] = max(
                terrain_command_ranges["lin_vel_y"][0],
                self.command_ranges["lin_vel_y"][0])
            self.env_command_ranges["lin_vel_y"][env_ids, 1] = min(
                terrain_command_ranges["lin_vel_y"][1],
                self.command_ranges["lin_vel_y"][1])
            self.env_command_ranges["ang_vel_yaw"][env_ids, 0] = max(
                terrain_command_ranges["ang_vel_yaw"][0],
                self.command_ranges["ang_vel_yaw"][0])
            self.env_command_ranges["ang_vel_yaw"][env_ids, 1] = min(
                terrain_command_ranges["ang_vel_yaw"][1],
                self.command_ranges["ang_vel_yaw"][1])
            if self.cfg.commands.heading_command:
                self.env_command_ranges["heading"][env_ids, 0] = max(
                    terrain_command_ranges["heading"][0],
                    self.command_ranges["heading"][0])
                self.env_command_ranges["heading"][env_ids, 1] = min(
                    terrain_command_ranges["heading"][1],
                    self.command_ranges["heading"][1])

    def _apply_command_range_curriculum(self):
        """[vendored _resample_commands, curriculum head]: promote the global
        command ranges to every stage whose ratio the current progress has
        passed, popping the consumed entries.

        Split out of _resample_commands so a resumed run can land on the
        correct stage immediately (see _wty_resync_curriculum) instead of
        walking the first envs through the narrowest band until they resample.

        Crossed entries are applied in ASCENDING ratio order, unlike the
        vendored backwards pop loop. Live training crosses one threshold at a
        time, where the two orders agree; a resumed run can cross several at
        once, and the vendored order would leave the EARLIEST (narrowest) band
        in place because each later assignment overwrites the previous one.
        """
        curriculum = self.cfg.commands.command_range_curriculum
        if not len(curriculum):
            return
        current_iter = self.common_step_counter // self.num_steps_per_iter
        progress = self._wty_progress()
        crossed = sorted(
            (i for i, cmd_cfg in enumerate(curriculum) if progress >= cmd_cfg["ratio"]),
            key=lambda i: curriculum[i]["ratio"])
        for i in crossed:
            cmd_cfg = curriculum[i]
            self.command_ranges["lin_vel_x"] = cmd_cfg["lin_vel_x"]
            self.command_ranges["lin_vel_y"] = cmd_cfg["lin_vel_y"]
            self.command_ranges["ang_vel_yaw"] = cmd_cfg["ang_vel_yaw"]
            self.command_ranges["heading"] = cmd_cfg["heading"]
            self.max_lin_vel = max(
                abs(self.command_ranges["lin_vel_x"][0]), abs(self.command_ranges["lin_vel_x"][1]),
                abs(self.command_ranges["lin_vel_y"][0]), abs(self.command_ranges["lin_vel_y"][1]))
            self._update_env_command_ranges()
            print(f"Command range updated at iter {current_iter} "
                  f"(progress {progress:.3f}): {self.command_ranges}")
        # pop consumed entries after the pass (descending, so indices hold)
        for i in sorted(crossed, reverse=True):
            curriculum.pop(i)

    def _resample_commands(self, env_ids):
        """[vendored _resample_commands]: dynamic (traverse-guaranteeing)
        command resampling with per-terrain-type limits, limit-velocity draws,
        a zero-command curriculum and an iteration-based command-range
        curriculum. Replaces the host's uniform resampler.

        The vendored turn_over zero-command block is dropped: the host has no
        turn_over machinery and the ported go2 config keeps turn_over=False.
        """
        if len(env_ids) == 0:
            return
        self.stop_heading[env_ids] = False
        self._apply_command_range_curriculum()
        remaining_dist = torch.clip(
            0.625 * self.cfg.terrain.terrain_length
            - torch.norm(self.commands_xy_accumulation[env_ids], dim=1) * self.cfg.commands.resampling_time,
            0.0)
        self.commands_resampling_step[env_ids] = self.cfg.commands.resampling_time / self.dt
        if self.cfg.commands.dynamic_resample_commands:
            # arrive at boundary 0.625 times the width of the remaining distance.
            # The reset path (host reset_idx) calls this BEFORE zeroing
            # episode_length_buf, so the remaining step count can legitimately be
            # <= 0 here: exactly 0 for an env that terminates on contact at step
            # max_episode_length, -1 for a timeout. Both mean "no minimum speed
            # to enforce". The old timeout case (-1) already produced a negative
            # ratio that got clipped to 0, so mapping every non-positive budget
            # to 0 is behaviour-preserving there and removes the division-by-~0
            # (and the crash it used to raise) at exactly 0.
            remaining_steps = self.max_episode_length - self.episode_length_buf[env_ids]
            vel_low_bound = torch.where(
                remaining_steps > 0,
                remaining_dist / (torch.clamp(remaining_steps, min=1.0) * self.dt),
                torch.zeros_like(remaining_dist))
            vel_low_bound = torch.clip(vel_low_bound, 0.0)
            self.commands[env_ids, 0] = _sample_disjoint_intervals(
                env_ids,
                vel_low_bound,
                self.env_command_ranges["lin_vel_x"][env_ids, 0],
                self.env_command_ranges["lin_vel_x"][env_ids, 1],
                self.device)
            self.commands[env_ids, 1] = _sample_disjoint_intervals(
                env_ids,
                vel_low_bound,
                self.env_command_ranges["lin_vel_y"][env_ids, 0],
                self.env_command_ranges["lin_vel_y"][env_ids, 1],
                self.device)
            if self.cfg.commands.heading_command:
                r = torch.rand(len(env_ids), device=self.device)
                lower = self.env_command_ranges["heading"][env_ids, 0]
                upper = self.env_command_ranges["heading"][env_ids, 1]
                self.commands[env_ids, 3] = (upper - lower) * r + lower
            else:
                r = torch.rand(len(env_ids), device=self.device)
                lower = self.env_command_ranges["ang_vel_yaw"][env_ids, 0]
                upper = self.env_command_ranges["ang_vel_yaw"][env_ids, 1]
                self.commands[env_ids, 2] = (upper - lower) * r + lower
        else:
            self.commands[env_ids, 0] = _sample_single_interval(
                env_ids,
                self.env_command_ranges["lin_vel_x"][env_ids, 0],
                self.env_command_ranges["lin_vel_x"][env_ids, 1],
                self.device)
            self.commands[env_ids, 1] = _sample_single_interval(
                env_ids,
                self.env_command_ranges["lin_vel_y"][env_ids, 0],
                self.env_command_ranges["lin_vel_y"][env_ids, 1],
                self.device)
            if self.cfg.commands.heading_command:
                self.commands[env_ids, 3] = _sample_single_interval(
                    env_ids,
                    self.env_command_ranges["heading"][env_ids, 0],
                    self.env_command_ranges["heading"][env_ids, 1],
                    self.device)
            else:
                self.commands[env_ids, 2] = _sample_single_interval(
                    env_ids,
                    self.env_command_ranges["ang_vel_yaw"][env_ids, 0],
                    self.env_command_ranges["ang_vel_yaw"][env_ids, 1],
                    self.device)
            # set small commands to zero
            self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        rand_prob = torch.rand(len(env_ids), device=self.device)
        min_prob, max_prob = 0.0, 0.0
        # set limitation lin vel
        if self.limit_vel_prob > 0.0:
            max_prob += self.limit_vel_prob
            lim_mask = (rand_prob >= min_prob) * (rand_prob < max_prob)
            lim_env_ids = env_ids[lim_mask]
            if len(lim_env_ids) > 0:
                change_lim_env_ids = lim_env_ids
                if self.cfg.commands.limit_vel_invert_when_continuous:
                    was_limited = self.last_is_limit_vel[lim_env_ids]
                    invert_env_ids = lim_env_ids[was_limited]
                    self.commands[invert_env_ids, 0] *= -1.0
                    self.commands[invert_env_ids, 1] *= -1.0
                    self.commands[invert_env_ids, 2] *= -1.0
                    change_lim_env_ids = lim_env_ids[~was_limited]
                vel_idx = torch.randint(0, self.limit_vel_comb.shape[0], (len(change_lim_env_ids),), device=self.device)
                lin_vel_x_lim = torch.where(
                    self.limit_vel_comb[vel_idx, 0] == -1,
                    self.env_command_ranges["lin_vel_x"][change_lim_env_ids, 0],
                    self.env_command_ranges["lin_vel_x"][change_lim_env_ids, 1])
                lin_vel_x_lim[self.limit_vel_comb[vel_idx, 0] == 0] = 0.0
                lin_vel_y_lim = torch.where(
                    self.limit_vel_comb[vel_idx, 1] == -1,
                    self.env_command_ranges["lin_vel_y"][change_lim_env_ids, 0],
                    self.env_command_ranges["lin_vel_y"][change_lim_env_ids, 1])
                lin_vel_y_lim[self.limit_vel_comb[vel_idx, 1] == 0] = 0.0
                ang_vel_z_lim = torch.where(
                    self.limit_vel_comb[vel_idx, 2] == -1,
                    self.env_command_ranges["ang_vel_yaw"][change_lim_env_ids, 0],
                    self.env_command_ranges["ang_vel_yaw"][change_lim_env_ids, 1])
                ang_vel_z_lim[self.limit_vel_comb[vel_idx, 2] == 0] = 0.0
                self.commands[change_lim_env_ids, 0] = lin_vel_x_lim
                self.commands[change_lim_env_ids, 1] = lin_vel_y_lim
                self.commands[change_lim_env_ids, 2] = ang_vel_z_lim
                if self.cfg.commands.heading_command and self.cfg.commands.stop_heading_at_limit:
                    self.stop_heading[lim_env_ids] = True  # stop heading to current heading
                self.last_is_limit_vel[env_ids] = False
                self.last_is_limit_vel[lim_env_ids] = True
            else:
                self.last_is_limit_vel[env_ids] = False
            min_prob += self.limit_vel_prob

        # set all commands to zero with some probability
        if self.cfg.commands.zero_command_curriculum is not None:
            self.zero_command_proba = self._get_current_scale(self.cfg.commands.zero_command_curriculum)
        if self.zero_command_proba > 0.0:
            max_prob += self.zero_command_proba
            next_resampling_step = torch.clip(
                self.max_episode_length - self.episode_length_buf[env_ids]
                - (remaining_dist / (0.8 * self.max_lin_vel * self.dt + 1e-9)),
                min=0.0,
                max=self.cfg.commands.resampling_time / self.dt)
            zero_mask = (rand_prob >= min_prob) * (rand_prob < max_prob) * (next_resampling_step > 0.0)
            zero_env_ids = env_ids[zero_mask]
            if len(zero_env_ids) > 0:
                self.commands[zero_env_ids, :2] = 0.0
                self.commands_resampling_step[zero_env_ids] = next_resampling_step[zero_mask]
                if self.cfg.commands.limit_ang_vel_at_zero_command_prob > 0.0:
                    ang_vel_rand = torch.rand(len(zero_env_ids), device=self.device)  # independent distribution
                    add_ang_mask = ang_vel_rand < self.cfg.commands.limit_ang_vel_at_zero_command_prob
                    add_ang_env_ids = zero_env_ids[add_ang_mask]
                    if len(add_ang_env_ids) > 0:
                        direction_rand = torch.rand(len(add_ang_env_ids), device=self.device)
                        self.commands[add_ang_env_ids, 2] = torch.where(
                            direction_rand < 0.5,
                            self.env_command_ranges["ang_vel_yaw"][add_ang_env_ids, 0],
                            self.env_command_ranges["ang_vel_yaw"][add_ang_env_ids, 1])
                        if self.cfg.commands.heading_command:
                            self.stop_heading[add_ang_env_ids] = True
            min_prob += self.zero_command_proba

        self.commands_xy_accumulation[env_ids] += self.commands[env_ids, :2]

        if self.cfg.commands.heading_command:
            heading_env_ids = env_ids[self.stop_heading[env_ids] == 0.0]
            if len(heading_env_ids) > 0:
                forward = quat_apply(self.simulator.base_quat[heading_env_ids], self.forward_vec[heading_env_ids])
                heading = torch.atan2(forward[:, 1], forward[:, 0])
                self.commands[heading_env_ids, 2] = torch.clip(
                    0.5 * wrap_to_pi(self.commands[heading_env_ids, 3] - heading),
                    self.env_command_ranges["ang_vel_yaw"][heading_env_ids, 0],
                    self.env_command_ranges["ang_vel_yaw"][heading_env_ids, 1])

    # ------------------------------------------------------------------
    # Termination (vendored check_termination; REPLACEMENT override)
    # ------------------------------------------------------------------

    def compute_observations(self):
        """Cooperative wrap (calls super): after the arm's own
        compute_observations has consumed the PREVIOUS control-rate
        last_dof_vel (dof_acc obs feature) and this step's reward is already
        computed, refresh the tracker's copy -- mirroring the vendored
        post_physics_step tail (ref legged_robot.py:143-145:
        compute_observations, then last_dof_vel[:] = dof_vel[:]). The
        observation contract itself is untouched (each arm keeps its own).

        MRO note: this wrap only wins on the HIM arm (Go2MoECTSHIM defines no
        compute_observations of its own). Go2MoECTS.compute_observations
        shadows it, so the MoE arm performs the same refresh inline at the
        end of its own method -- keep the two in sync.
        """
        # vendored post_physics_step order: push AFTER reward + reset_idx,
        # BEFORE observations (ref legged_robot.py:138-143).
        self._push_robots_if_due()
        super().compute_observations()
        self._wty_last_dof_vel[:] = self.simulator.dof_vel[:]

    def _reward_dof_acc(self):
        """[vendored _reward_dof_acc]: control-rate joint acceleration penalty.

        go2_rl_gym/legged_gym/envs/base/legged_robot.py:1257-1259 computes
        sum(((last_dof_vel - dof_vel) / dt)^2) with last_dof_vel spanning one
        CONTROL step (20 ms). The host base class computes the same formula
        but over the simulator's per-substep buffer (5 ms window), making the
        penalty ~16x weaker. REPLACEMENT override (no super()) reading the
        mixin's control-rate tracker instead.
        """
        return torch.sum(torch.square(
            (self._wty_last_dof_vel - self.simulator.dof_vel) / self.dt), dim=1)

    def check_termination(self):
        """[vendored check_termination]: same-step base-contact termination.

        go2_rl_gym/legged_gym/envs/base/legged_robot.py:179 terminates the
        SAME step any termination body (go2: ["base"] only, via
        cfg.asset.terminate_after_contacts_on) exceeds the contact-force
        threshold; the vendored tilt/projected-gravity check is commented out
        and there is no consecutive-failure counter. The host default
        (LeggedRobot.check_termination: 10 N threshold, projected-gravity
        tilt check, fail_to_terminal_time_s consecutive-step counter) is
        intentionally replaced for the moects tasks; the time-out path is
        unchanged. Threshold comes from
        cfg.env.base_contact_terminate_threshold (2.5 N; vendored 1.0).

        REPLACEMENT override (the mixin's only non-cooperative one): no
        super() call -- the host fail-counter / tilt logic would be dead
        weight. The shared force-norm buffers are re-populated exactly like
        the host (the collision reward and the privileged-obs feet forces
        consume them) and fail_buf degrades to a long-typed 0/1 same-step
        flag; reset_buf / time_out_buf keep their usual semantics for
        _reward_termination and the eval harness (reset_buf & ~time_out_buf).
        """
        # force-norm buffer population, identical to the host's (the 4D branch
        # is the IsaacLab contact-sensor history layout)
        if len(self.simulator.link_contact_forces.shape) == 4:
            self.terminated_bodies_force_norm = torch.max(torch.norm(self.simulator.link_contact_forces[:, :, self.simulator.termination_contact_indices, :], dim=-1), dim=1)[0]
            self.penalized_bodies_force_norm = torch.max(torch.norm(self.simulator.link_contact_forces[:, :, self.simulator.penalized_contact_indices, :], dim=-1), dim=1)[0]
            self.feet_force_norm = torch.max(torch.norm(self.simulator.link_contact_forces[:, :, self.simulator.feet_contact_indices, :], dim=-1), dim=1)[0]
            self.feet_max_force_z = torch.max(self.simulator.link_contact_forces[:, :, self.simulator.feet_contact_indices, 2], dim=1)[0]
        else:
            self.terminated_bodies_force_norm = torch.norm(self.simulator.link_contact_forces[:, self.simulator.termination_contact_indices, :], dim=-1)
            self.penalized_bodies_force_norm = torch.norm(self.simulator.link_contact_forces[:, self.simulator.penalized_contact_indices, :], dim=-1)
            self.feet_force_norm = torch.norm(self.simulator.link_contact_forces[:, self.simulator.feet_contact_indices, :], dim=-1)
            self.feet_max_force_z = self.simulator.link_contact_forces[:, self.simulator.feet_contact_indices, 2]
        # vendored: torch.any(torch.norm(contact_forces[termination_bodies]) > threshold, dim=1)
        self.terminated_by_base_contact = torch.any(
            self.terminated_bodies_force_norm > self.base_contact_terminate_threshold, dim=1)
        self.fail_buf = self.terminated_by_base_contact.to(torch.long)  # 0/1, no consecutive counter
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf = self.terminated_by_base_contact | self.time_out_buf

    # ------------------------------------------------------------------
    # Reset lifecycle (vendored _reset_dofs / reset_idx)
    # ------------------------------------------------------------------

    def _reset_dofs(self, env_ids):
        """[vendored _reset_dofs]: multiplicative reset joint positions.

        go2_rl_gym/legged_gym/envs/base/legged_robot.py:629 samples
        dof_pos = default_dof_pos * U(0.5, 1.5) and zeroes dof_vel; the host
        base class (LeggedRobot._reset_dofs) uses the additive
        default + U(-0.2, 0.2) instead. REPLACEMENT override scoped to the
        moects tasks via this mixin (no super() call; dof_vel is zeroed in
        both, so it is unchanged).
        """
        dof_pos = self.simulator.default_dof_pos * torch_rand_float(
            0.5, 1.5, (len(env_ids), self.num_actions), self.device)
        dof_vel = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float,
                              device=self.device, requires_grad=False)
        self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)

    def reset_idx(self, env_ids):
        """[vendored reset_idx]: run the terrain curriculum and clear the
        command accumulators around the host reset, then log the
        per-terrain-type curriculum metrics."""
        if len(env_ids) > 0:
            # True episode lengths of the finishing episodes, captured BEFORE
            # the zeroing below (and the host's inside super().reset_idx);
            # _write_episode_telemetry divides per-episode means by them.
            ep_lengths = self.episode_length_buf[env_ids].float().clone()
            # The host only calls _update_terrain_curriculum when
            # cfg.terrain.curriculum is set; moe_grid keeps that flag off (the
            # builder dispatch), so the mixin drives the vendored curriculum
            # explicitly. Early-out guards (init_done / active) live inside
            # _update_terrain_curriculum.
            if not self.cfg.terrain.curriculum:
                self._update_terrain_curriculum(env_ids)
            # Cleared BEFORE super().reset_idx() re-samples commands: the
            # vendored accumulation starts from the freshly sampled command.
            self.commands_xy_accumulation[env_ids] = 0.0
            self.commands_resampling_step[env_ids] = self.cfg.commands.resampling_time / self.dt
            # Vendored reset_idx zeroes episode_length_buf and only THEN calls
            # _resample_commands, so its dynamic velocity lower bound is computed
            # against a full fresh episode. The host reverses that order
            # (legged_robot.reset_idx: _resample_commands, ... , buf = 0), which
            # would make _resample_commands size the bound against the *old*
            # episode's leftover steps -- and hit the degenerate zero-budget case
            # for an env terminating exactly at max_episode_length. Zero it here
            # to restore the vendored ordering; the host's own zeroing below is
            # then a no-op, and nothing between here and there reads the buffer.
            self.episode_length_buf[env_ids] = 0
            # vendored reset_idx (ref legged_robot.py:224): zero the control-rate
            # last_dof_vel for the fresh episode, same as the vendored
            # last_dof_vel[env_ids] = 0 (and the simulator's own buffer, which
            # genesis_simulator already zeroes on reset).
            self._wty_last_dof_vel[env_ids] = 0.0
        super().reset_idx(env_ids)
        # termination-reason telemetry: fraction of this reset batch that ended
        # by base contact / by time-out (watch metric for the 2.5 N threshold).
        # Both flags still hold the termination-step values here -- the host
        # reset_idx does not clear them. The runner averages the per-batch
        # fractions into Episode/* over the logging window, same as the rew_*
        # episode sums.
        if len(env_ids) > 0:
            self.extras["episode"]["termination_base_contact"] = torch.mean(
                self.terminated_by_base_contact[env_ids].float())
            self.extras["episode"]["termination_timeout"] = torch.mean(
                self.time_out_buf[env_ids].float())
        # per-terrain-type curriculum metrics (vendored reset_idx extras)
        if len(env_ids) > 0 and self._wty_curriculum_active:
            self.extras["episode"]["terrain_level_all"] = torch.mean(
                self.simulator.terrain_levels.float())
            for name, cols in self._wty_name2cols.items():
                self.extras["episode"]["terrain_level_" + name] = torch.mean(
                    self.simulator.terrain_levels[torch.isin(self.simulator.terrain_types, cols)].float())
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # training telemetry (Group 2): curriculum state, physical tracking
        # error, torque saturation, terrain promote/demote counters
        if len(env_ids) > 0:
            self._write_episode_telemetry(env_ids, ep_lengths)

    def _write_episode_telemetry(self, env_ids, ep_lengths):
        """Emit the mixin's Episode/* training telemetry into extras["episode"].

        Called from reset_idx AFTER super().reset_idx rebuilt extras["episode"];
        the runner averages the per-reset-batch values over the logging window,
        exactly like the rew_* episode sums. ep_lengths must be captured by the
        caller BEFORE this method runs: the mixin zeroes episode_length_buf
        ahead of super().reset_idx (vendored ordering), so the per-episode
        means below divide by the true episode length, not the nominal max.
        """
        ep = self.extras["episode"]
        ep_len = torch.clamp(ep_lengths, min=1.0)
        # Instantaneous curriculum state: makes the command-range pops
        # (command_range_curriculum), the reward-scale ramps
        # (curriculum_rewards) and the zero-command ramp visible in TB.
        ep["zero_command_proba"] = float(self.zero_command_proba)
        ep["cmd_range_lin_vel_x_max"] = float(self.command_ranges["lin_vel_x"][1])
        ep["cmd_range_ang_vel_yaw_max"] = float(self.command_ranges["ang_vel_yaw"][1])
        for name, scale in self.reward_curriculum_scales.items():
            ep["rewscale_" + name] = float(scale)
        # Physical tracking error, raw units (no reward sigma): per-episode
        # mean per-step error, then the mean over this reset batch.
        ep["tracking_lin_vel_err"] = torch.mean(self._tele_lin_vel_err_sum[env_ids] / ep_len)
        ep["tracking_ang_vel_err"] = torch.mean(self._tele_ang_vel_err_sum[env_ids] / ep_len)
        # Torque saturation |tau/tau_lim|: per-episode mean of the per-step
        # joint mean, and the episode's worst single joint (running max).
        ep["torque_sat_mean"] = torch.mean(self._tele_torque_sat_sum[env_ids] / ep_len)
        ep["torque_sat_max"] = torch.mean(self._tele_torque_sat_max[env_ids])
        self._tele_lin_vel_err_sum[env_ids] = 0.0
        self._tele_ang_vel_err_sum[env_ids] = 0.0
        self._tele_torque_sat_sum[env_ids] = 0.0
        self._tele_torque_sat_max[env_ids] = 0.0
        # Terrain curriculum: level max (the per-type means live in reset_idx)
        # and promote/demote counts since the previous reset batch -- per-batch
        # counts averaged over the window, the episode_sums emission pattern.
        if self._wty_curriculum_active:
            ep["terrain_level_max"] = torch.max(self.simulator.terrain_levels.float())
        ep["terrain_promotions"] = float(self._tele_terrain_promotions)
        ep["terrain_demotions"] = float(self._tele_terrain_demotions)
        self._tele_terrain_promotions = 0
        self._tele_terrain_demotions = 0

    # ------------------------------------------------------------------
    # Reward computation (vendored compute_reward + update_reward_curriculum)
    # ------------------------------------------------------------------

    def compute_reward(self):
        """[vendored compute_reward + update_reward_curriculum]: host reward
        loop plus per-term iteration-based scale ramping
        (cfg.rewards.curriculum_rewards)."""
        self._update_reward_curriculum()
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            if name in self.reward_curriculum_scales:
                rew = rew * self.reward_curriculum_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
        # host UED accounting tail, kept for contract parity (never active here)
        if self.ued_adapter is not None and getattr(self.cfg.env, "ued_enabled", False):
            self.ued_adapter.record_step(self.rew_buf)
        # vendored post_physics_step order: resample commands AFTER the
        # reward is computed (ref legged_robot.py:133-134), so the reward on
        # a resample step is still scored against the old command.
        self._resample_commands_if_due()

    def _update_reward_curriculum(self, force_update: bool = False):
        """[vendored update_reward_curriculum]: refresh ramped reward scales
        once per PPO iteration."""
        if self.reward_curriculum_configs:
            if self.common_step_counter % self.num_steps_per_iter == 0 or force_update:
                for config in self.reward_curriculum_configs:
                    self.reward_curriculum_scales[config["reward_name"]] = self._get_current_scale(config)

    def _get_current_scale(self, config):
        """[vendored get_current_scale, ratio-based]: linear interpolation
        between start_value and end_value over the budget fraction
        [start_ratio, end_ratio] (the vendored absolute thresholds
        1500/5000/20000/50000 are stored as their fractions of the vendored
        150k budget, so the curricula rescale with the planned total).

        config: {'start_ratio': 0.0, 'end_ratio': 0.01, 'start_value': 1.0, 'end_value': 0.0}
        """
        percentage = ((self._wty_progress() - config["start_ratio"])
                      / (config["end_ratio"] - config["start_ratio"]))
        percentage = max(min(percentage, 1.0), 0.0)
        return (1.0 - percentage) * config["start_value"] + percentage * config["end_value"]

    # ------------------------------------------------------------------
    # Dynamic tracking sigma (vendored _get_dynamic_sigma + tracking rewards)
    # ------------------------------------------------------------------

    def _get_dynamic_sigma(self, target_vel_abs, v_min, v_max):
        """[vendored _get_dynamic_sigma]: per-terrain, per-level tracking-sigma
        scaling. The vendored gate on cfg.terrain.curriculum becomes the
        mixin's curriculum-active flag (moe_grid keeps the host flag off)."""
        default_sigma = self.cfg.rewards.tracking_sigma
        if not self._wty_curriculum_active or self.dynamic_sigma_cfg is None:
            return torch.full_like(target_vel_abs, default_sigma)
        target_sigmas = self.terrain_max_sigmas[self.wty_terrain_ids]
        sigma = torch.full_like(target_vel_abs, default_sigma)
        # based on velocity ranges, compute sigma
        # v_min <= v < v_max (linear interpolation)
        mask = (target_vel_abs >= v_min) & (target_vel_abs < v_max)
        if mask.any():
            ratio = (target_vel_abs[mask] - v_min) / (v_max - v_min)
            sigma[mask] = default_sigma + ratio * (target_sigmas[mask] - default_sigma)
        # v >= v_max
        mask = target_vel_abs >= v_max
        if mask.any():
            sigma[mask] = target_sigmas[mask]
        # based on terrain level, compute sigma
        level_scale = torch.clamp(torch.exp((self.simulator.terrain_levels.float() + 1.0) / 10.0) - 1.0, max=1.0)
        sigma = default_sigma + level_scale * (sigma - default_sigma)
        return sigma

    def _reward_tracking_lin_vel(self):
        """[vendored _reward_tracking_lin_vel]: per-axis dynamic sigma."""
        if self.dynamic_sigma_cfg is None:
            sigma_x = sigma_y = self.cfg.rewards.tracking_sigma
        else:
            vmin = self.dynamic_sigma_cfg["min_lin_vel"]
            vmax = self.dynamic_sigma_cfg["max_lin_vel"]
            sigma_x = self._get_dynamic_sigma(torch.abs(self.commands[:, 0]), vmin, vmax)
            sigma_y = self._get_dynamic_sigma(torch.abs(self.commands[:, 1]), vmin, vmax)
        lin_vel_error_sq = torch.square(self.commands[:, :2] - self.simulator.base_lin_vel[:, :2])
        scaled_error = lin_vel_error_sq[:, 0] / sigma_x + lin_vel_error_sq[:, 1] / sigma_y
        return torch.exp(-scaled_error)

    def _reward_tracking_ang_vel(self):
        """[vendored _reward_tracking_ang_vel]: dynamic sigma on yaw rate."""
        if self.dynamic_sigma_cfg is None:
            sigma = self.cfg.rewards.tracking_sigma
        else:
            vmin = self.dynamic_sigma_cfg["min_ang_vel"]
            vmax = self.dynamic_sigma_cfg["max_ang_vel"]
            sigma = self._get_dynamic_sigma(torch.abs(self.commands[:, 2]), vmin, vmax)
        ang_vel_error_sq = torch.square(self.commands[:, 2] - self.simulator.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error_sq / sigma)

    # ------------------------------------------------------------------
    # Base-height and foot-lift rewards (vendored _get_base_height,
    # _reward_correct_base_height, _reward_feet_regulation)
    # ------------------------------------------------------------------

    def _get_base_height(self):
        """[vendored _get_base_height]: base height above the mean ground
        estimated from the masked height scan."""
        if not self.cfg.terrain.measure_heights:
            return self.simulator.base_pos[:, 2]
        masked_heights = self.simulator.measured_heights * self.base_height_scan_mask.unsqueeze(0)
        estimated_ground_z = masked_heights.sum(dim=1) / self.num_base_height_scan_points
        return self.simulator.base_pos[:, 2] - estimated_ground_z

    def _reward_correct_base_height(self):
        """[vendored _reward_correct_base_height]"""
        base_height = self._get_base_height()
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_feet_regulation(self):
        """[vendored _reward_feet_regulation]: CTS foot-lift regularizer --
        with growing foot xy velocity, demand proportionally higher lift."""
        base_height = self._get_base_height()
        feet_pos = self.simulator.feet_pos                    # (N, 4, 3), world frame
        feet_xy_vel = self.simulator.feet_vel[:, :, :2]       # (N, 4, 2), world frame
        delta_feet = feet_pos - self.simulator.base_pos.unsqueeze(1)
        # foot height relative to the base, then relative to the ground (N, 4)
        feet2base_height = (delta_feet * self.simulator.projected_gravity.unsqueeze(1)).sum(-1)
        feet_height = torch.clamp(base_height.unsqueeze(1) - feet2base_height, min=0.0)
        return (feet_xy_vel.pow(2).sum(-1)
                * torch.exp(-feet_height / (0.025 * self.cfg.rewards.base_height_target))).sum(-1)

    def _reward_hip_to_default(self):
        """[vendored go2_env.py _reward_hip_to_default]: keep hip joints close
        to the default pose (prevents the overly narrow CTS gait)."""
        hip_dof_indices = [0, 3, 6, 9]
        hip_pos = self.simulator.dof_pos[:, hip_dof_indices]
        default_hip_pos = self.simulator.default_dof_pos[:, hip_dof_indices]
        return torch.sum(torch.abs(hip_pos - default_hip_pos), dim=1)

    # ------------------------------------------------------------------
    # Collision reward (vendored formula, Genesis-calibrated threshold)
    # ------------------------------------------------------------------

    def _reward_collision(self):
        """[vendored _reward_collision]: count penalized-body (thigh/calf)
        links whose contact-force norm exceeds the threshold.

        Same formula as the reference (go2_rl_gym legged_robot.py) and the
        host LeggedRobot, but the threshold comes from
        cfg.rewards.collision_force_threshold (0.1 N, reference parity)
        instead of the host's hardcoded 10.0 N. Justified by Genesis
        measurement: non-contacting penalized links read exactly 0.0 N (no
        solver noise floor; every observed nonzero reading was a genuine
        ground contact) -- see tmp/collision_force_stats.json and
        tmp/rear_calf_contact_probe.json.
        """
        return torch.sum(
            1. * (self.penalized_bodies_force_norm
                  > self.cfg.rewards.collision_force_threshold), dim=1)
