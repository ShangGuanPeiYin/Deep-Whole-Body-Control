import torch

from dwbc_isaaclab.tasks.widow_go1.rewards import arm_reward, combine_rewards, leg_reward, termination_flags
from dwbc_isaaclab.tasks.widow_go1.goals import sphere_to_cart


def test_leg_scalar_keeps_arm_channel():
    total, extras = combine_rewards(torch.tensor([2.0]), torch.tensor([3.0]))
    assert total.tolist() == [2.0]
    assert extras["leg_reward"].tolist() == [2.0]
    assert extras["arm_reward"].tolist() == [3.0]


def test_zero_state_reward_matches_legacy_active_scales():
    value, terms = leg_reward(
        actions=torch.zeros(1, 18),
        torques=torch.zeros(1, 20),
        joint_vel=torch.zeros(1, 20),
        base_lin_vel=torch.zeros(1, 3),
        base_ang_vel=torch.zeros(1, 3),
        commands=torch.zeros(1, 3),
        foot_force_z=torch.zeros(1, 4),
    )
    assert torch.allclose(value, torch.tensor([(0.2 + 0.15) / 100]))
    assert set(terms) == {
        "survive", "tracking_lin_vel_x_l1", "tracking_ang_vel_yaw_exp",
        "hip_action_l2", "foot_contacts_z", "energy_square",
    }


def test_arm_reward_preserves_tracking_and_energy_terms():
    value, terms = arm_reward(
        ee_position_local=torch.tensor([[0.2, 0.0, 0.0]]),
        ee_goal_sphere=torch.tensor([[0.2, 0.0, 0.0]]),
        torques=torch.zeros(1, 20),
        joint_vel=torch.zeros(1, 20),
    )
    assert torch.allclose(value, torch.tensor([0.55 / 100]))
    assert set(terms) == {"tracking_ee_sphere", "arm_energy_abs_sum"}


def test_arm_yaw_error_uses_full_legacy_yaw_range():
    goal = torch.tensor([[0.4, 0.0, 0.0]])
    actual = sphere_to_cart(torch.tensor([[0.4, 0.0, 0.6 * torch.pi]]))
    value, _ = arm_reward(ee_position_local=actual, ee_goal_sphere=goal,
                          torques=torch.zeros(1,20), joint_vel=torch.zeros(1,20))
    # yaw range is [-3*pi/5, 3*pi/5]; half the full range gives error 0.5.
    torch.testing.assert_close(value, torch.tensor([0.0055]) * torch.exp(torch.tensor(-0.5)))


def test_termination_reason_prioritizes_height_after_tilt_checks():
    terminated, reason = termination_flags(
        roll=torch.tensor([0.3, 0.0]),
        pitch=torch.tensor([0.0, 0.0]),
        height=torch.tensor([0.4, 0.2]),
        ee_goal_sphere=torch.tensor([[0.6, 0.7, 0.1], [0.6, 0.7, 0.1]]),
    )
    assert terminated.tolist() == [True, True]
    assert reason.tolist() == [1, 3]
