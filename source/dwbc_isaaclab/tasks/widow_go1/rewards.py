"""Named legacy reward terms with separate leg and arm channels."""

from __future__ import annotations

import torch

from .goals import cart_to_sphere


def leg_reward(
    *,
    actions: torch.Tensor,
    torques: torch.Tensor,
    joint_vel: torch.Tensor,
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    commands: torch.Tensor,
    foot_force_z: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    terms = {
        "survive": torch.ones_like(commands[:, 0]) * 0.2,
        "tracking_lin_vel_x_l1": (
            -torch.abs(commands[:, 0] - base_lin_vel[:, 0]) + torch.abs(commands[:, 0])
        ) * 0.5,
        "tracking_ang_vel_yaw_exp": torch.exp(
            -torch.abs(commands[:, 2] - base_ang_vel[:, 2])
        ) * 0.15,
        "hip_action_l2": torch.sum(actions[:, (0, 3, 6, 9)] ** 2, dim=-1) * -0.01,
        "foot_contacts_z": torch.sum(foot_force_z**2, dim=-1) * -1.0e-4,
        "energy_square": torch.sum((torques[:, :12] * joint_vel[:, :12]) ** 2, dim=-1) * -6.0e-5,
    }
    return torch.stack(tuple(terms.values())).sum(dim=0) / 100.0, terms


def arm_reward(
    *,
    ee_position_local: torch.Tensor,
    ee_goal_sphere: torch.Tensor,
    torques: torch.Tensor,
    joint_vel: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    sphere_scale = torch.tensor((2.0, 5.0 / (3.0 * torch.pi), 5.0 / (3.0 * torch.pi)), device=torques.device)
    tracking_error = torch.sum(torch.abs(cart_to_sphere(ee_position_local) - ee_goal_sphere) * sphere_scale, dim=-1)
    terms = {
        "tracking_ee_sphere": torch.exp(-tracking_error) * 0.55,
        "arm_energy_abs_sum": torch.sum(torch.abs(torques[:, 12:18] * joint_vel[:, 12:18]), dim=-1) * -0.004,
    }
    return torch.stack(tuple(terms.values())).sum(dim=0) / 100.0, terms


def combine_rewards(
    leg: torch.Tensor, arm: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return leg, {"leg_reward": leg, "arm_reward": arm}


def termination_flags(
    roll: torch.Tensor,
    pitch: torch.Tensor,
    height: torch.Tensor,
    ee_goal_sphere: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return legacy task failures and stable per-environment reason codes."""
    roll_failure = ((roll > 0.2) & (ee_goal_sphere[:, 2] >= 0)) | (
        (roll < -0.2) & (ee_goal_sphere[:, 2] <= 0)
    )
    pitch_failure = ((pitch > 0.2) & (ee_goal_sphere[:, 1] >= 0)) | (
        (pitch < -0.2) & (ee_goal_sphere[:, 1] <= 0)
    )
    height_failure = height < 0.325
    reason = torch.zeros_like(height, dtype=torch.int8)
    reason[roll_failure] = 1
    reason[pitch_failure] = 2
    reason[height_failure] = 3
    return roll_failure | pitch_failure | height_failure, reason
