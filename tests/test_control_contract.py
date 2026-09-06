import pytest
import torch

from dwbc_isaaclab.tasks.widow_go1.control import ActionDelayBuffer, compute_pd_torques


def test_two_step_action_delay():
    delay = ActionDelayBuffer(1, 18, 2, "cpu")
    assert torch.equal(delay.push(torch.ones(1, 18)), torch.zeros(1, 18))
    assert torch.equal(delay.push(torch.full((1, 18), 2.0)), torch.zeros(1, 18))
    assert torch.equal(delay.push(torch.full((1, 18), 3.0)), torch.ones(1, 18))


def test_delay_rejects_wrong_action_width_and_can_reset_subset():
    delay = ActionDelayBuffer(2, 18, 2, "cpu")
    with pytest.raises(ValueError, match="expected actions shape"):
        delay.push(torch.zeros(2, 17))
    delay.push(torch.ones(2, 18))
    delay.reset(torch.tensor([1]))
    assert torch.count_nonzero(delay.history[1]) == 0


def test_pd_preserves_legacy_unwrapped_waist_and_never_actuates_grippers():
    actions = torch.zeros(1, 18)
    joint_pos = torch.zeros(1, 20)
    joint_pos[0, 12] = 2 * torch.pi - 0.1
    joint_pos[0, 7] = 2 * torch.pi - 0.2
    torques = compute_pd_torques(
        actions,
        joint_pos,
        torch.zeros_like(joint_pos),
        default_joint_pos=torch.zeros(20),
        action_scale=torch.ones(18),
        motor_strength=torch.ones(1, 18),
        p_gains=torch.ones(18),
        d_gains=torch.zeros(18),
        effort_limits=torch.full((20,), 100.0),
    )
    assert torch.allclose(torques[0, 12], torch.tensor(-2 * torch.pi + 0.1), atol=1e-6)
    assert torch.allclose(torques[0, 7], torch.tensor(0.2), atol=1e-6)
    assert torch.equal(torques[:, 18:], torch.zeros(1, 2))


def test_pd_applies_the_six_delayed_adaptive_arm_gain_actions():
    actions = torch.zeros(1, 24)
    actions[:, 12:18] = 1.0
    actions[:, 18:] = 4.0
    torques = compute_pd_torques(
        actions,
        torch.zeros(1, 20),
        torch.zeros(1, 20),
        default_joint_pos=torch.zeros(20),
        action_scale=torch.ones(18),
        motor_strength=torch.ones(1, 18),
        p_gains=torch.full((18,), 5.0),
        d_gains=torch.zeros(18),
        effort_limits=torch.full((20,), 100.0),
    )
    # Adaptive stiffness is 5 + 4, while the position branch stays the first
    # 18 action values.  The remaining two gripper torques remain state-only.
    assert torch.allclose(torques[:, 12:18], torch.full((1, 6), 9.0))
    assert torch.equal(torques[:, 18:], torch.zeros(1, 2))
