import pytest
import torch

from dwbc_isaaclab.tasks.widow_go1.resets import DEFAULT_JOINT_POS, sample_reset_state


def test_fixed_seed_reset_is_reproducible_and_has_legacy_shapes():
    first = sample_reset_state(3, 7, "cpu")
    second = sample_reset_state(3, 7, "cpu")
    assert torch.equal(first.root_pose, second.root_pose)
    assert torch.equal(first.root_velocity, second.root_velocity)
    assert torch.equal(first.joint_position, second.joint_position)
    assert first.root_pose.shape == (3, 7)
    assert first.joint_position.shape == (3, 20)


def test_reset_keeps_zero_default_joints_zero_and_perturbs_nonzero_pose():
    state = sample_reset_state(8, 3, "cpu")
    zero_defaults = DEFAULT_JOINT_POS == 0
    assert torch.count_nonzero(state.joint_position[:, zero_defaults]) == 0
    expected = DEFAULT_JOINT_POS[~zero_defaults]
    ratios = state.joint_position[:, ~zero_defaults] / expected
    assert torch.all((ratios >= 0.8) & (ratios <= 1.2))
    assert torch.all(state.root_pose[:, 2] == 0.42)
    assert torch.all(torch.abs(state.root_velocity) <= 0.1)
    assert DEFAULT_JOINT_POS[-2:].tolist() == pytest.approx([0.015, -0.015])
