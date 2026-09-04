"""Rollout storage with the original DWBC two-objective tensor contract."""

from dataclasses import dataclass
import torch


@dataclass
class Transition:
    observations: torch.Tensor | None = None
    critic_observations: torch.Tensor | None = None
    actions: torch.Tensor | None = None
    rewards: torch.Tensor | None = None
    dones: torch.Tensor | None = None
    values: torch.Tensor | None = None
    actions_log_prob: torch.Tensor | None = None
    action_mean: torch.Tensor | None = None
    action_sigma: torch.Tensor | None = None
    target_arm_torques: torch.Tensor | None = None
    current_arm_dof_pos: torch.Tensor | None = None
    current_arm_dof_vel: torch.Tensor | None = None

    def clear(self):
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name, None)


class RolloutStorage:
    Transition = Transition

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, critic_obs_shape, action_shape, device="cpu"):
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.critic_observations = torch.zeros(num_transitions_per_env, num_envs, *critic_obs_shape, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *action_shape, device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 2, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, dtype=torch.bool, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 2, device=self.device)
        self.returns = torch.zeros_like(self.values)
        self.advantages = torch.zeros_like(self.values)
        self.actions_log_prob = torch.zeros_like(self.values)
        self.mu = torch.zeros_like(self.actions)
        self.sigma = torch.zeros_like(self.actions)
        torque_shape = (num_transitions_per_env, num_envs, 6)
        self.target_arm_torques = torch.zeros(torque_shape, device=self.device)
        self.current_arm_dof_pos = torch.zeros(torque_shape, device=self.device)
        self.current_arm_dof_vel = torch.zeros(torque_shape, device=self.device)
        self.step = 0

    @staticmethod
    def _required(transition, name):
        value = getattr(transition, name)
        if value is None:
            raise ValueError(f"transition field {name!r} is required")
        return value

    def add_transitions(self, transition, torque_supervision=None):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("rollout buffer overflow")
        index = self.step
        mappings = {"observations": self.observations, "critic_observations": self.critic_observations,
                    "actions": self.actions, "rewards": self.rewards, "values": self.values,
                    "actions_log_prob": self.actions_log_prob, "action_mean": self.mu, "action_sigma": self.sigma}
        for source_name, destination in mappings.items():
            destination[index].copy_(self._required(transition, source_name))
        self.dones[index].copy_(self._required(transition, "dones").view(-1, 1))
        has_torque = transition.target_arm_torques is not None
        if torque_supervision is True and not has_torque:
            raise ValueError("torque supervision requested without torque targets")
        if has_torque:
            self.target_arm_torques[index].copy_(transition.target_arm_torques)
            self.current_arm_dof_pos[index].copy_(self._required(transition, "current_arm_dof_pos"))
            self.current_arm_dof_vel[index].copy_(self._required(transition, "current_arm_dof_vel"))
        self.step += 1

    def compute_returns(self, last_values, gamma, lam):
        advantage = torch.zeros_like(last_values)
        for step in reversed(range(self.num_transitions_per_env)):
            next_values = last_values if step == self.num_transitions_per_env - 1 else self.values[step + 1]
            not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + not_terminal * gamma * next_values - self.values[step]
            advantage = delta + not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]
        raw = self.returns - self.values
        self.advantages.copy_((raw - raw.mean()) / (raw.std() + 1e-8))

    def mini_batch_generator(self, num_mini_batches, num_epochs=1):
        batch_size = self.num_envs * self.num_transitions_per_env
        if batch_size % num_mini_batches:
            raise ValueError("rollout batch size must be divisible by num_mini_batches")
        names = ("observations", "critic_observations", "actions", "values", "advantages", "returns",
                 "actions_log_prob", "mu", "sigma", "target_arm_torques", "current_arm_dof_pos", "current_arm_dof_vel")
        flat = {name: getattr(self, name).flatten(0, 1) for name in names}
        size = batch_size // num_mini_batches
        # Legacy DWBC samples one permutation and reuses its mini-batch partition
        # over all learning epochs.
        indices = torch.randperm(batch_size, device=self.device)
        for _ in range(num_epochs):
            for start in range(0, batch_size, size):
                selected = indices[start:start + size]
                yield tuple(flat[name][selected] for name in names)

    def clear(self):
        self.step = 0
