"""End-effector goal and quaternion boundary helpers."""

from __future__ import annotations

import torch


def xyzw_to_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    return quaternion[..., (3, 0, 1, 2)]


def wxyz_to_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    return quaternion[..., (1, 2, 3, 0)]


def _quat_conjugate_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    result = quaternion.clone()
    result[..., :3] *= -1.0
    return result


def _quat_mul_xyzw(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lx, ly, lz, lw = left.unbind(dim=-1)
    rx, ry, rz, rw = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dim=-1,
    )


def orientation_error_xyzw(desired: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    relative = _quat_mul_xyzw(desired, _quat_conjugate_xyzw(current))
    return relative[..., :3] * torch.sign(relative[..., 3:4])


def cart_to_sphere(cartesian: torch.Tensor) -> torch.Tensor:
    radius = torch.linalg.vector_norm(cartesian, dim=-1)
    eps = torch.finfo(cartesian.dtype).eps
    elevation = torch.asin(torch.clamp(cartesian[..., 2] / radius.clamp_min(eps), -1.0, 1.0))
    yaw = torch.atan2(cartesian[..., 1], cartesian[..., 0])
    return torch.stack((radius, elevation, yaw), dim=-1)


def sphere_to_cart(spherical: torch.Tensor) -> torch.Tensor:
    radius, elevation, yaw = spherical.unbind(dim=-1)
    projection = radius * torch.cos(elevation)
    return torch.stack(
        (projection * torch.cos(yaw), projection * torch.sin(yaw), radius * torch.sin(elevation)), dim=-1
    )


def goal_collision_mask(start, goal):
    """Original ten-sample spherical path rejection against body box/ground."""
    t = torch.linspace(0, 1, 10, device=start.device)[None, :, None]
    cart = sphere_to_cart(torch.lerp(start[:, None], goal[:, None], t))
    lower = cart.new_tensor((-0.2, -0.15, -0.515))
    upper = cart.new_tensor((0.3, 0.15, -0.115))
    inside = ((cart > lower) & (cart < upper)).all(-1).any(-1)
    return inside | (cart[..., 2] < -0.57).any(-1)
