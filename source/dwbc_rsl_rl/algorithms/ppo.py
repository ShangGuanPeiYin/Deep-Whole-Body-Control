"""Dual-objective PPO used by Deep Whole-Body Control."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from dwbc_rsl_rl.storage import RolloutStorage, Transition


def stack_dual_rewards(leg_reward: torch.Tensor, arm_reward: torch.Tensor) -> torch.Tensor:
    if leg_reward.shape != arm_reward.shape:
        raise ValueError(f"leg and arm reward shapes differ: {leg_reward.shape} != {arm_reward.shape}")
    return torch.stack((leg_reward, arm_reward), dim=-1)


def bootstrap_timeouts(rewards: torch.Tensor, values: torch.Tensor, timeouts: torch.Tensor, gamma: float) -> torch.Tensor:
    if rewards.shape != values.shape or rewards.shape[-1] != 2:
        raise ValueError("rewards and values must share shape (N,2)")
    return rewards + gamma * values * timeouts.to(values.dtype).unsqueeze(-1)


def schedule_value(counter: int, schedule: Sequence[float]) -> float:
    """Legacy ramp: (target, start, duration), protected against zero duration."""
    target, start, duration = schedule
    if duration <= 0:
        return float(target if counter >= start else 0.0)
    stage = min(max((counter - start) / duration, 0.0), 1.0)
    return float(stage * target)


def decay_schedule_value(counter: int, schedule: Sequence[float]) -> float:
    target, start, duration = schedule
    if duration <= 0:
        return float(target if counter < start else 0.0)
    stage = min(max((counter - start) / duration, 0.0), 1.0)
    return float((1.0 - stage) * target)


def mix_advantages(advantages: torch.Tensor, ratio: float) -> torch.Tensor:
    if advantages.shape[-1] != 2:
        raise ValueError("dual advantages must have trailing width 2")
    mixed = torch.empty_like(advantages)
    mixed[..., 0] = advantages[..., 0] + ratio * advantages[..., 1]
    mixed[..., 1] = advantages[..., 1] + ratio * advantages[..., 0]
    return mixed


def compute_ppo_losses(new_log_prob, old_log_prob, advantages, values, old_values, returns, entropy, *,
                       clip_param, value_loss_coef, entropy_coef, mixing_ratio, use_clipped_value_loss=True):
    mixed = mix_advantages(advantages, mixing_ratio)
    probability_ratio = torch.exp(new_log_prob - old_log_prob)
    surrogate = -mixed * probability_ratio
    clipped_surrogate = -mixed * probability_ratio.clamp(1.0 - clip_param, 1.0 + clip_param)
    surrogate_loss = torch.maximum(surrogate, clipped_surrogate).mean()
    if use_clipped_value_loss:
        clipped_values = old_values + (values - old_values).clamp(-clip_param, clip_param)
        value_loss = torch.maximum((values - returns).square(), (clipped_values - returns).square()).mean()
    else:
        value_loss = (values - returns).square().mean()
    entropy_mean = entropy.mean()
    loss = surrogate_loss + value_loss_coef * value_loss - entropy_coef * entropy_mean
    return {"loss": loss, "surrogate_loss": surrogate_loss, "value_loss": value_loss, "entropy": entropy_mean}


class PPO:
    """A narrow, dependency-free port of the project's custom PPO update."""

    def __init__(self, actor_critic, *, num_learning_epochs=1, num_mini_batches=1, clip_param=0.2,
                 gamma=0.998, lam=0.95, value_loss_coef=1.0, entropy_coef=0.0,
                 learning_rate=1e-3, max_grad_norm=1.0, use_clipped_value_loss=True,
                 device="cpu", mixing_schedule=(0.5, 2000, 4000), torque_supervision=True,
                 torque_supervision_schedule=(0.1, 1000, 1000), min_policy_std=None,
                 dagger_update_freq=20, priv_reg_schedule=(0.0, 0.0, 0, 1)):
        self.device = torch.device(device)
        self.actor_critic = actor_critic.to(self.device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.hist_encoder_optimizer = torch.optim.Adam(self.actor_critic.actor.history_encoder.parameters(), lr=learning_rate)
        self.transition = Transition()
        self.storage: RolloutStorage | None = None
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.mixing_schedule = tuple(mixing_schedule)
        self.torque_supervision = torque_supervision
        self.torque_supervision_schedule = tuple(torque_supervision_schedule)
        self.min_policy_std = None if min_policy_std is None else torch.as_tensor(min_policy_std, device=self.device)
        self.dagger_update_freq = dagger_update_freq
        self.priv_reg_schedule = tuple(priv_reg_schedule)
        self.counter = 0
        self.default_arm_p_gains: torch.Tensor | None = None
        self.default_arm_d_gains: torch.Tensor | None = None
        self.default_arm_dof_pos: torch.Tensor | None = None

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape=(860,), critic_obs_shape=(860,), action_shape=None):
        if action_shape is None:
            action_shape = (self.actor_critic.num_actions,)
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def act(self, obs, critic_obs, hist_encoding=False):
        self.transition.actions = self.actor_critic.act(obs, hist_encoding).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, leg_rewards, arm_rewards, dones, infos):
        if self.storage is None:
            raise RuntimeError("init_storage() must be called before collecting transitions")
        if self.torque_supervision:
            for key in ('target_arm_torques', 'current_arm_dof_pos', 'current_arm_dof_vel'):
                if key not in infos:
                    raise ValueError(f'torque supervision requires {key}')
        rewards = stack_dual_rewards(leg_rewards, arm_rewards)
        if "time_outs" in infos:
            rewards = bootstrap_timeouts(rewards, self.transition.values, infos["time_outs"], self.gamma)
        self.transition.rewards = rewards
        self.transition.dones = dones
        for key in ("target_arm_torques", "current_arm_dof_pos", "current_arm_dof_vel"):
            if key in infos:
                setattr(self.transition, key, infos[key].detach())
        self.storage.add_transitions(self.transition, self.torque_supervision and "target_arm_torques" in infos)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        if self.storage is None:
            raise RuntimeError("storage is not initialized")
        self.storage.compute_returns(self.actor_critic.evaluate(last_critic_obs).detach(), self.gamma, self.lam)

    def _priv_regularizer(self, obs):
        privileged = self.actor_critic.infer_priv_latent(obs)
        with torch.no_grad():
            historical = self.actor_critic.infer_history_latent(obs)
        return torch.linalg.vector_norm(privileged - historical, dim=1).mean()

    def _priv_coefficient(self):
        start_coef, end_coef, start, duration = self.priv_reg_schedule
        if duration <= 0:
            stage = float(self.counter >= start)
        else:
            stage = min(max((self.counter - start) / duration, 0.0), 1.0)
        return start_coef + stage * (end_coef - start_coef)

    def set_arm_default_coeffs(self, p_gains, d_gains, default_pos):
        self.default_arm_p_gains = torch.as_tensor(p_gains, device=self.device)
        self.default_arm_d_gains = torch.as_tensor(d_gains, device=self.device)
        self.default_arm_dof_pos = torch.as_tensor(default_pos, device=self.device)

    def _predicted_arm_torques(self, obs, current_pos, current_vel):
        target = self.actor_critic.act_inference(obs)[:, 12:18]
        gains = self.actor_critic.arm_gain_delta
        if self.default_arm_p_gains is None or self.default_arm_dof_pos is None:
            raise RuntimeError("set_arm_default_coeffs() is required for torque supervision")
        p_gains = self.default_arm_p_gains + gains
        d_gains = 2.0 * torch.sqrt(p_gains.clamp_min(1e-6)) if self.actor_critic.actor.adaptive_arm_gains else self.default_arm_d_gains
        return p_gains * (target + self.default_arm_dof_pos - current_pos) - d_gains * current_vel

    def update(self):
        if self.storage is None:
            raise RuntimeError("storage is not initialized")
        totals = {"value_loss": 0.0, "surrogate_loss": 0.0, "torque_loss": 0.0, "priv_reg_loss": 0.0}
        updates = 0
        mixing = schedule_value(self.counter, self.mixing_schedule)
        torque_weight = decay_schedule_value(self.counter, self.torque_supervision_schedule)
        priv_coefficient = self._priv_coefficient()
        for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            (obs, critic_obs, actions, old_values, advantages, returns, old_log_prob, _old_mu, _old_sigma,
             target_torques, current_pos, current_vel) = batch
            self.actor_critic.update_distribution(obs)
            losses = compute_ppo_losses(
                self.actor_critic.get_actions_log_prob(actions), old_log_prob, advantages,
                self.actor_critic.evaluate(critic_obs), old_values, returns, self.actor_critic.entropy,
                clip_param=self.clip_param, value_loss_coef=self.value_loss_coef,
                entropy_coef=self.entropy_coef, mixing_ratio=mixing,
                use_clipped_value_loss=self.use_clipped_value_loss,
            )
            priv_loss = self._priv_regularizer(obs)
            loss = losses["loss"] + priv_coefficient * priv_loss
            torque_loss = torch.zeros((), device=self.device)
            if self.torque_supervision:
                torque_loss = (self._predicted_arm_torques(obs, current_pos, current_vel) - target_torques).square().mean()
                loss = loss + torque_weight * torque_loss
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            totals["value_loss"] += losses["value_loss"].item()
            totals["surrogate_loss"] += losses["surrogate_loss"].item()
            totals["torque_loss"] += torque_loss.item()
            totals["priv_reg_loss"] += priv_loss.item()
            updates += 1
        for key in totals:
            totals[key] /= max(updates, 1)
        totals.update({"value_mixing_ratio": mixing, "torque_weight": torque_weight, "priv_reg_coef": priv_coefficient})
        self.storage.clear()
        self.counter += 1
        self.enforce_min_std()
        return totals

    def update_dagger(self):
        if self.storage is None:
            raise RuntimeError("storage is not initialized")
        total = 0.0
        updates = 0
        for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            obs = batch[0]
            with torch.no_grad():
                privileged = self.actor_critic.infer_priv_latent(obs)
            historical = self.actor_critic.infer_history_latent(obs)
            loss = torch.linalg.vector_norm(privileged - historical, dim=1).mean()
            self.hist_encoder_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.actor.history_encoder.parameters(), self.max_grad_norm)
            self.hist_encoder_optimizer.step()
            total += loss.item()
            updates += 1
        self.storage.clear()
        self.counter += 1
        return total / max(updates, 1)

    def enforce_min_std(self):
        if self.min_policy_std is not None:
            with torch.no_grad():
                self.actor_critic.std.copy_(torch.maximum(self.actor_critic.std, self.min_policy_std))
