"""Deterministic reset sampling for the WidowGo1 task."""

from __future__ import annotations

from dataclasses import dataclass

import torch


DEFAULT_JOINT_POS = torch.tensor(
    [
        -0.1, 0.8, -1.5,
        0.1, 0.8, -1.5,
        -0.1, 0.8, -1.5,
        0.1, 0.8, -1.5,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.015, -0.015,
    ],
    dtype=torch.float32,
)


@dataclass(frozen=True)
class ResetState:
    root_pose: torch.Tensor
    root_velocity: torch.Tensor
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor


def sample_reset_state(
    num_envs: int,
    seed: int,
    device: str | torch.device,
    *,
    origins: torch.Tensor | None = None,
    default_joint_pos: torch.Tensor | None = None,
    origin_perturb: float = 0.5,
    velocity_perturb: float = 0.1,
) -> ResetState:
    """Sample the legacy reset distribution without touching global RNG state."""
    device = torch.device(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    if origins is None:
        origins = torch.zeros(num_envs, 3, device=device)
    else:
        origins = origins.to(device=device, dtype=torch.float32)
    if origins.shape != (num_envs, 3):
        raise ValueError(f"origins must have shape {(num_envs, 3)}, got {tuple(origins.shape)}")
    defaults = DEFAULT_JOINT_POS if default_joint_pos is None else default_joint_pos
    defaults = defaults.to(device=device, dtype=torch.float32)
    if defaults.shape != (20,):
        raise ValueError(f"default_joint_pos must have shape (20,), got {tuple(defaults.shape)}")

    root_pose = torch.zeros(num_envs, 7, device=device)
    root_pose[:, :3] = origins
    root_pose[:, :2] += (2.0 * torch.rand(num_envs, 2, generator=generator, device=device) - 1.0) * origin_perturb
    root_pose[:, 2] += 0.42
    root_pose[:, 3] = 1.0  # Isaac Lab boundary convention: wxyz.
    root_velocity = (
        2.0 * torch.rand(num_envs, 6, generator=generator, device=device) - 1.0
    ) * velocity_perturb
    joint_scale = 0.8 + 0.4 * torch.rand(num_envs, 20, generator=generator, device=device)
    joint_scale[:, 18:] = 1.0  # legacy zeros are illegal in Lab; keep fingers at the nearest limits.
    joint_position = defaults.unsqueeze(0) * joint_scale
    joint_velocity = torch.zeros(num_envs, 20, device=device)
    return ResetState(root_pose, root_velocity, joint_position, joint_velocity)
