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


def compute_osc_torques(
    arm_mass_matrix: torch.Tensor,
    ee_jacobian: torch.Tensor,
    pose_error: torch.Tensor,
    ee_velocity: torch.Tensor,
    kp: torch.Tensor,
    kd: torch.Tensor,
    body_jacobians: torch.Tensor,
    body_masses: torch.Tensor,
) -> torch.Tensor:
    """Legacy six-axis operational-space feedback plus gravity compensation.

    Jacobians and pose/velocity inputs are world-frame, in linear/angular order.
    Body Jacobians contain only the six actuated arm columns. Preserve pinverse
    (including singular configurations); do not add damping or clip targets.
    """
    mass_inverse = torch.pinverse(arm_mass_matrix)
    operational_mass = torch.pinverse(ee_jacobian @ mass_inverse @ ee_jacobian.transpose(1, 2))
    wrench = (kp * pose_error - kd * ee_velocity).unsqueeze(-1)
    feedback = (ee_jacobian.transpose(1, 2) @ operational_mass @ wrench).squeeze(-1)
    gravity_wrench = body_jacobians.new_zeros(*body_masses.shape, 6, 1)
    gravity_wrench[:, :, 2, 0] = body_masses * 9.81
    gravity = (body_jacobians.transpose(2, 3) @ gravity_wrench).squeeze(-1).sum(dim=1)
    return feedback + gravity


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
    """Compute canonical PD torques, with an optional six-axis arm-gain branch.

    The stable task contract is 18 position actions.  The repaired research
    branch has 24 entries: those 18 positions followed by six direct arm-PD
    stiffness deltas.  The gain actions intentionally bypass motor-strength
    and action-scale multiplication, matching the original controller's
    intended adaptive-gain formula.
    """
    num_envs = actions.shape[0]
    expected_prefix = (num_envs,)
    if tuple(actions.shape[:1]) != expected_prefix or actions.shape[1] not in (18, 24):
        raise ValueError(f"actions must have shape ({num_envs},18) or ({num_envs},24), got {tuple(actions.shape)}")
    position_actions = actions[:, :18]
    if tuple(joint_pos.shape) != (num_envs, 20) or tuple(joint_vel.shape) != (num_envs, 20):
        raise ValueError("joint position and velocity must have shape (num_envs, 20)")
    current = joint_pos[:, :18].clone()
    # Legacy control wraps index -8 of the *18-wide* native vector: native
    # RR_thigh_joint (canonical index 7), not the waist. Observation wrapping
    # uses the 20-wide vector and really does wrap the waist. Preserve this
    # distinction; correcting the legacy controller belongs in a research fork.
    current[:, 7] = wrap_to_pi(current[:, 7])
    target = default_joint_pos[:18] + position_actions * motor_strength * action_scale
    policy_torque = p_gains * (target - current) - d_gains * joint_vel[:, :18]
    if actions.shape[1] == 24:
        # The legacy formula is defined only for positive stiffness.  Preserve
        # it on that domain and protect the repaired optional branch from NaNs.
        arm_p_gains = (p_gains[12:] + actions[:, 18:]).clamp_min(1.0e-6)
        arm_d_gains = 2.0 * torch.sqrt(arm_p_gains)
        arm_torque = arm_p_gains * (target[:, 12:] - current[:, 12:]) - arm_d_gains * joint_vel[:, 12:18]
        policy_torque = torch.cat((policy_torque[:, :12], arm_torque), dim=-1)
    torques = torch.cat((policy_torque, torch.zeros(num_envs, 2, device=actions.device)), dim=-1)
    return torch.clamp(torques, min=-effort_limits, max=effort_limits)
