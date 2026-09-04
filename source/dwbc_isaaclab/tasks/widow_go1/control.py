"""Legacy-compatible delayed action and explicit PD helpers."""

from __future__ import annotations

import torch


class ActionDelayBuffer:
    def __init__(self, num_envs: int, action_dim: int, delay: int, device: str | torch.device):
        if delay < 0:
            raise ValueError("delay must be non-negative")
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.delay = delay
        self.history = torch.zeros(num_envs, delay + 1, action_dim, device=device)

    def push(self, actions: torch.Tensor) -> torch.Tensor:
        expected = (self.num_envs, self.action_dim)
        if tuple(actions.shape) != expected:
            raise ValueError(f"expected actions shape {expected}, got {tuple(actions.shape)}")
        if self.delay:
            self.history[:, :-1] = self.history[:, 1:].clone()
        self.history[:, -1] = actions
        return self.history[:, 0].clone()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.history.zero_()
        else:
            self.history[env_ids] = 0.0


def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angles + torch.pi, 2.0 * torch.pi) - torch.pi


def compute_pd_torques(
    actions: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    *,
    default_joint_pos: torch.Tensor,
    action_scale: torch.Tensor,
    motor_strength: torch.Tensor,
    p_gains: torch.Tensor,
    d_gains: torch.Tensor,
    effort_limits: torch.Tensor,
) -> torch.Tensor:
    """Compute 18 policy torques in canonical order and append two zero gripper torques."""
    num_envs = actions.shape[0]
    expected = (num_envs, 18)
    if tuple(actions.shape) != expected:
        raise ValueError(f"actions must have shape {expected}, got {tuple(actions.shape)}")
    if tuple(joint_pos.shape) != (num_envs, 20) or tuple(joint_vel.shape) != (num_envs, 20):
        raise ValueError("joint position and velocity must have shape (num_envs, 20)")
    current = joint_pos[:, :18].clone()
    current[:, 12] = wrap_to_pi(current[:, 12])
    target = default_joint_pos[:18] + actions * motor_strength * action_scale
    policy_torque = p_gains * (target - current) - d_gains * joint_vel[:, :18]
    torques = torch.cat((policy_torque, torch.zeros(num_envs, 2, device=actions.device)), dim=-1)
    return torch.clamp(torques, min=-effort_limits, max=effort_limits)
