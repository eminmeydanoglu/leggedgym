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

from __future__ import annotations

import time
import os
import math
from collections import deque
import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import wandb
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv


# Type aliases for configuration dictionaries
RunnerConfig = Dict[str, Any]
AlgorithmConfig = Dict[str, Any]
PolicyConfig = Dict[str, Any]
TrainConfig = Dict[str, Any]


class OnPolicyRunner:
    """On-policy RL training runner for PPO-style algorithms."""

    def __init__(
        self,
        env: VecEnv,
        train_cfg: TrainConfig,
        log_dir: Optional[str] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """Initialize the on-policy runner.

        Args:
            env: Vectorized environment for training.
            train_cfg: Training configuration containing runner, algorithm, and policy configs.
            log_dir: Directory for logging and saving models.
            device: Device to run training on (e.g., 'cpu', 'cuda').
        """
        self.cfg: RunnerConfig = train_cfg["runner"]
        self.alg_cfg: AlgorithmConfig = train_cfg["algorithm"]
        self.policy_cfg: PolicyConfig = train_cfg["policy"]
        self.all_cfg: TrainConfig = train_cfg
        self.wandb_run_name: str = (
            self.cfg["experiment_name"]
            + "_"
            + datetime.now().strftime("%b%d_%H-%M-%S")
            + "_"
            + self.cfg["run_name"]
        )
        self.device: torch.device = torch.device(device)
        self.env: VecEnv = env
        self._init_agent_and_algo()
        self.num_steps_per_env: int = self.cfg["num_steps_per_env"]
        self.save_interval: int = self.cfg["save_interval"]

        # In-distribution Eval V2 / best_tracking.pt selection (opt-in via cfg).
        # eval_interval == 0 disables it entirely (default), preserving legacy
        # behavior for tasks that don't configure eval.
        self.eval_interval: int = self.cfg.get("eval_interval", 0)
        self.eval_steps: int = self.cfg.get("eval_steps", 1100)
        self.eval_warmup: int = self.cfg.get("eval_warmup", 50)
        self.eval_seed: int = self.cfg.get("eval_seed", 12345)
        self.eval_fall_guard: float = self.cfg.get("eval_fall_guard", 0.25)
        self.best_eval_score: float = float("inf")
        self.best_tracking_key: Optional[Tuple[float, ...]] = None

        # Iteration-based command schedule (opt-in via runner cfg). A list of
        # {"start_iteration": int, "lin_vel_x": [lo, hi]} stages: at each boundary
        # the whole population's lin_vel_x range switches, decoupling the command
        # distribution from policy performance so every method sees the same
        # commands at the same training stage (see codex_plan.md sec. 2). None
        # (default) preserves legacy performance-based curriculum behaviour.
        self.command_schedule: Optional[List[Dict[str, Any]]] = self.cfg.get("command_schedule", None)
        self._active_schedule_start: Optional[int] = None
        self._active_schedule_range: Optional[List[float]] = None

        # Training seed, carried into checkpoints so the seed travels with the
        # weights (statistical unit of replication). Sourced from train_cfg.seed.
        seed = self.all_cfg.get("seed")
        self.training_seed: Optional[int] = int(seed) if seed is not None else None

        self._init_storage()

        # Log
        self.log_dir: Optional[str] = log_dir
        self.sync_wandb: bool = self.cfg.get("sync_wandb", False)
        self.writer: Optional[SummaryWriter] = None
        self.tot_timesteps: int = 0
        self.tot_time: float = 0.0
        self.current_learning_iteration: int = 0

        self.env.reset()
    
    def _init_agent_and_algo(self) -> None:
        """Initialize the actor-critic network and PPO algorithm."""
        if self.env.num_privileged_obs is not None:
            num_critic_obs: int = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"])
        actor_critic: ActorCritic = actor_critic_class(
            self.env.num_obs,
            num_critic_obs,
            self.env.num_actions,
            **self.policy_cfg
        ).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
    
    def _init_storage(self) -> None:
        """Initialize the rollout storage for the algorithm."""
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env, 
            (self.env.num_obs,),
            (self.env.num_privileged_obs,), 
            (self.env.num_actions,),
        )
    
    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
    ) -> None:
        """Run the training loop for a specified number of iterations.

        Args:
            num_learning_iterations: Number of learning iterations to run.
            init_at_random_ep_len: Whether to initialize episode lengths randomly.
        """
        self._pre_learn(init_at_random_ep_len)
        if self.training_seed is not None:
            print(f"[train] seed={self.training_seed} "
                  f"start_iter={self.current_learning_iteration}")
        # land on the correct schedule stage for the (possibly resumed) start iter
        self._apply_command_schedule(self.current_learning_iteration)
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()

        ep_infos: List[Dict[str, Any]] = []
        rewbuffer: deque = deque(maxlen=100)
        lenbuffer: deque = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # advance the command schedule if this iteration crosses a stage boundary
            self._apply_command_schedule(it)
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                assert self.log_dir is not None
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)), iteration=it)

            # Periodic in-distribution eval -> Eval/* + best_tracking.pt selection.
            # The eval rollout resets the env, so we refresh obs/critic_obs and the
            # training-only reward/length bookkeeping before resuming rollout.
            if self.eval_interval > 0 and it % self.eval_interval == 0:
                # The training rollout above ran under torch.inference_mode(), so
                # the env's step/reset buffers are inference tensors. The eval
                # rollout reset()s and steps the same env, which are in-place
                # updates -> must also run inside inference_mode or PyTorch raises
                # "Inplace update to inference tensor outside InferenceMode".
                with torch.inference_mode():
                    self._run_eval(it)
                obs = self.env.get_observations().to(self.device)
                privileged_obs = self.env.get_privileged_observations()
                critic_obs = (privileged_obs if privileged_obs is not None else obs).to(self.device)
                cur_reward_sum.zero_()
                cur_episode_length.zero_()
                self.alg.actor_critic.train()

            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        assert self.log_dir is not None
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    @staticmethod
    def command_stage_for_iter(
        schedule: Optional[List[Dict[str, Any]]],
        it: int,
    ) -> Optional[Dict[str, Any]]:
        """Return the active command-schedule stage for iteration ``it``.

        The active stage is the one with the largest ``start_iteration`` that is
        still ``<= it``. Returns ``None`` for an empty schedule or an iteration
        before the first stage. Pure function -> unit-testable without an env.
        """
        if not schedule:
            return None
        active: Optional[Dict[str, Any]] = None
        for stage in sorted(schedule, key=lambda s: s["start_iteration"]):
            if it >= stage["start_iteration"]:
                active = stage
            else:
                break
        return active

    def _apply_command_schedule(self, it: int) -> None:
        """Switch the env's ``lin_vel_x`` command range to stage active at ``it``.

        No-op unless a ``command_schedule`` is configured. Only re-applies when the
        active stage actually changes (so a resume lands on the correct stage and
        mid-training boundaries fire exactly once). Disables the performance-based
        command curriculum (the schedule is the single source of truth) and
        re-samples every env's command so the new range takes effect immediately.
        """
        stage = self.command_stage_for_iter(self.command_schedule, it)
        if stage is None:
            return
        if stage["start_iteration"] == self._active_schedule_start:
            return
        self._active_schedule_start = stage["start_iteration"]
        rng = list(stage["lin_vel_x"])
        self._active_schedule_range = rng
        # the schedule owns the command distribution -> turn off perf curriculum
        self.env.cfg.commands.curriculum = False
        if hasattr(self.env, "command_ranges") and "lin_vel_x" in self.env.command_ranges:
            self.env.command_ranges["lin_vel_x"] = list(rng)
        # re-sample all envs under the new range so the switch is immediate
        if hasattr(self.env, "_resample_commands"):
            all_ids = torch.arange(self.env.num_envs, device=self.env.device)
            self.env._resample_commands(all_ids)
        print(f"[schedule] it={it} stage_start={stage['start_iteration']} lin_vel_x={rng}")

    def _pre_learn(self, init_at_random_ep_len: bool) -> None:
        """Prepare for training by initializing logging and episode buffers.

        Args:
            init_at_random_ep_len: Whether to randomize initial episode lengths.
        """
        if self.log_dir is not None and self.writer is None:
            if self.sync_wandb:
                wandb.init(
                    project="LeggedGym-Ex",
                    name=self.wandb_run_name,
                    sync_tensorboard=True,
                    config=self.all_cfg,
                )
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
    
    def log(
        self,
        locs: Dict[str, Any],
        width: int = 80,
        pad: int = 35,
    ) -> None:
        """Log training metrics to tensorboard and console.

        Args:
            locs: Dictionary containing iteration metrics and buffers.
            width: Width of the log output.
            pad: Padding for log formatting.
        """
        assert self.writer is not None
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str_iter = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str_iter.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str_iter.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def _run_eval(self, it: int) -> Dict[str, float]:
        """Run one in-distribution eval and update ``best_tracking.pt``.

        The exact V2 lexicographic rule is safety first (fixed-window fall rate),
        then normalized tracking, then earlier iteration. It shares the rollout
        with the standalone in-distribution evaluator and leaves the env freshly
        reset; the caller refreshes observation handles.
        """
        from legged_gym.scripts.eval.indist import (
            run_indist_eval, tracking_score, tracking_selection_key,
        )

        self.alg.actor_critic.eval()
        metrics = run_indist_eval(
            self.env,
            self.alg.actor_critic.act_inference,
            steps=self.eval_steps,
            warmup=self.eval_warmup,
            seed=self.eval_seed,
        )
        self.alg.actor_critic.train()

        if self.writer is not None:
            for key, val in metrics.items():
                self.writer.add_scalar('Eval/' + key, val, it)

        score = tracking_score(metrics)
        key = tracking_selection_key(metrics, it, self.eval_fall_guard)
        print(f"[eval] it={it} return={metrics['mean_return']:.2f} "
              f"fall_rate={metrics['fall_rate']:.3f} ep_len={metrics['mean_ep_len']:.1f} "
              f"lin_err={metrics['tracking_lin_err']:.3f} tracking_score={score:.4f} "
              f"(best_key={self.best_tracking_key})")

        if (self.best_tracking_key is None or key < self.best_tracking_key) and self.log_dir is not None:
            self.best_tracking_key = key
            self.best_eval_score = score
            if self.writer is not None:
                self.writer.add_scalar('Eval/best_tracking_score', score, it)
            self.save(os.path.join(self.log_dir, 'best_tracking.pt'), iteration=it,
                      infos={
                          'eval_metrics': metrics,
                          'selection_metric': 'v2_tracking_lexicographic',
                          'tracking_score': score,
                          'validation_seed': self.eval_seed,
                          'fall_threshold': self.eval_fall_guard,
                          'selected_iteration': it,
                          'selection_key': list(key),
                      })
            print(f"[eval] new best_tracking.pt @ it={it} (key={key})")
        return metrics

    def save(
        self,
        path: str,
        infos: Optional[Dict[str, Any]] = None,
        iteration: Optional[int] = None,
    ) -> None:
        """Save the model checkpoint to disk.

        Args:
            path: File path to save the checkpoint.
            infos: Optional additional information to save with the checkpoint.
            iteration: Iteration count to store in the checkpoint. Defaults to
                self.current_learning_iteration.
        """
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': iteration if iteration is not None else self.current_learning_iteration,
            # Keep eval selection state with the training checkpoint so resume
            # cannot treat an already-evaluated run as having no best model.
            'best_eval_score': self.best_eval_score,
            'best_tracking_key': self.best_tracking_key,
            # Provenance travelling with the weights (see codex_plan.md sec. 2):
            # the training seed and the active command-schedule stage.
            'training_seed': self.training_seed,
            'schedule_stage_start': self._active_schedule_start,
            'schedule_lin_vel_x': self._active_schedule_range,
            'infos': infos,
        }, path)

    @staticmethod
    def _checkpoint_eval_score(checkpoint: Dict[str, Any]) -> Optional[float]:
        """Return a persisted scalar diagnostic from old or new checkpoints."""
        score = checkpoint.get('best_eval_score')
        if score is None:
            infos = checkpoint.get('infos') or {}
            score = infos.get('tracking_score', infos.get('eval_score'))
        if isinstance(score, torch.Tensor):
            score = score.item()
        try:
            score = float(score)
        except (TypeError, ValueError):
            return None
        return None if math.isnan(score) else score

    def load(
        self,
        path: str,
        load_optimizer: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Load a model checkpoint from disk.

        Args:
            path: File path to load the checkpoint from.
            load_optimizer: Whether to load the optimizer state.

        Returns:
            Optional infos dict stored in the checkpoint.
        """
        loaded_dict = torch.load(path, weights_only=False)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']

        # Restore the full lexicographic selection state from the resumed model
        # or the sibling best_tracking checkpoint. Legacy best.pt has no V2 key;
        # it remains loadable, but the next V2 evaluation establishes a new key.
        choices = [loaded_dict]
        tracking_path = os.path.join(os.path.dirname(path), 'best_tracking.pt')
        if os.path.abspath(tracking_path) != os.path.abspath(path) and os.path.isfile(tracking_path):
            choices.append(torch.load(tracking_path, map_location='cpu', weights_only=False))
        keys = []
        for item in choices:
            key = item.get('best_tracking_key') or (item.get('infos') or {}).get('selection_key')
            if isinstance(key, (list, tuple)):
                try:
                    keys.append(tuple(float(x) for x in key))
                except (TypeError, ValueError):
                    pass
        self.best_tracking_key = min(keys) if keys else None
        scores = [self._checkpoint_eval_score(item) for item in choices]
        persisted_scores = [score for score in scores if score is not None]
        self.best_eval_score = min(persisted_scores, default=float('inf'))
        return loaded_dict['infos']

    def get_inference_policy(
        self,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Get the inference policy function.

        Args:
            device: Device to run inference on. If None, uses current device.

        Returns:
            Callable that takes observations and returns actions.
        """
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
