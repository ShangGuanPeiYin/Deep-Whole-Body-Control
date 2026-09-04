import pytest
import torch

from dwbc_isaaclab.tasks.widow_go1.observations import build_legacy_observation, compose_proprioception


def test_full_observation_width_and_frozen_block_order():
    proprio = torch.full((2, 76), 1.0)
    privileged = torch.full((2, 24), 2.0)
    history = torch.full((2, 10, 76), 3.0)
    observation = build_legacy_observation(proprio, privileged, history)
    assert observation.shape == (2, 860)
    assert torch.equal(observation[:, :76], proprio)
    assert torch.equal(observation[:, 76:100], privileged)
    assert torch.equal(observation[:, 100:], history.reshape(2, -1))


def test_proprioception_rejects_wrong_component_width():
    fields = {
        "orientation": torch.zeros(1, 2),
        "angular_velocity": torch.zeros(1, 3),
        "dof_pos": torch.zeros(1, 20),
        "dof_vel": torch.zeros(1, 20),
        "previous_action": torch.zeros(1, 17),
        "feet_contacts": torch.zeros(1, 4),
        "command": torch.zeros(1, 3),
        "ee_goal": torch.zeros(1, 3),
        "ee_orientation_error": torch.zeros(1, 3),
    }
    with pytest.raises(ValueError, match="previous_action"):
        compose_proprioception(**fields)
