"""Pure builders for the frozen WidowGo1 observation layout."""

from __future__ import annotations

import torch

from .contracts import ObservationLayout, POLICY_ACTION_NAMES


def build_privileged_observation(mass_params, friction, canonical_motor_strength):
    # Proprioception is reordered by the old task; privileged motor strengths
    # are deliberately NOT reordered there (FL, FR, RL, RR, arm).
    legacy_names = (POLICY_ACTION_NAMES[3:6] + POLICY_ACTION_NAMES[:3]
                    + POLICY_ACTION_NAMES[9:12] + POLICY_ACTION_NAMES[6:9]
                    + POLICY_ACTION_NAMES[12:])
    native_order = [POLICY_ACTION_NAMES.index(name) for name in legacy_names]
    return torch.cat((mass_params, friction, canonical_motor_strength[:, native_order] - 1), -1)


def relative_joint_positions(joint_positions, default_positions):
    positions = joint_positions.clone()
    positions[:, 12] = torch.remainder(positions[:, 12] + torch.pi, 2 * torch.pi) - torch.pi
    reference = default_positions.clone()
    # Legal reset limits do not redefine the legacy observation coordinate zero.
    reference[18:] = 0
    return positions - reference


PROPRIO_WIDTHS = {
    "orientation": 2,
    "angular_velocity": 3,
    "dof_pos": 20,
    "dof_vel": 20,
    "previous_action": 18,
    "feet_contacts": 4,
    "command": 3,
    "ee_goal": 3,
    "ee_orientation_error": 3,
}


def compose_proprioception(**fields: torch.Tensor) -> torch.Tensor:
    missing = [name for name in PROPRIO_WIDTHS if name not in fields]
    if missing:
        raise ValueError(f"missing proprioception fields: {missing}")
    batch_size = next(iter(fields.values())).shape[0]
    ordered = []
    for name, width in PROPRIO_WIDTHS.items():
        tensor = fields[name]
        expected = (batch_size, width)
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}")
        ordered.append(tensor)
    result = torch.cat(ordered, dim=-1)
    if result.shape[-1] != ObservationLayout.proprio:
        raise AssertionError(f"internal proprioception width error: {result.shape[-1]}")
    return result


def build_legacy_observation(
    proprioception: torch.Tensor, privileged: torch.Tensor, history: torch.Tensor
) -> torch.Tensor:
    batch_size = proprioception.shape[0]
    expected = {
        "proprioception": (batch_size, ObservationLayout.proprio),
        "privileged": (batch_size, ObservationLayout.privileged),
        "history": (batch_size, ObservationLayout.history, ObservationLayout.proprio),
    }
    actual = {
        "proprioception": tuple(proprioception.shape),
        "privileged": tuple(privileged.shape),
        "history": tuple(history.shape),
    }
    for name in expected:
        if actual[name] != expected[name]:
            raise ValueError(f"{name} must have shape {expected[name]}, got {actual[name]}")
    result = torch.cat((proprioception, privileged, history.reshape(batch_size, -1)), dim=-1)
    if result.shape[-1] != ObservationLayout.flat:
        raise AssertionError(f"internal observation width error: {result.shape[-1]}")
    return result
