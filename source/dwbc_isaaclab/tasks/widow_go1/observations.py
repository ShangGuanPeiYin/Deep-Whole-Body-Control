"""Pure builders for the frozen WidowGo1 observation layout."""

from __future__ import annotations

import torch

from .contracts import ObservationLayout


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
