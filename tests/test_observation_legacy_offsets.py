import torch


def test_privileged_motor_strength_keeps_legacy_native_order():
    from dwbc_isaaclab.tasks.widow_go1.observations import build_privileged_observation
    actual = build_privileged_observation(torch.zeros(1, 5), torch.ones(1, 1),
                                          torch.arange(18, dtype=torch.float32)[None] + 1)
    torch.testing.assert_close(actual[:, 6:], torch.tensor([
        [3., 4., 5., 0., 1., 2., 9., 10., 11., 6., 7., 8., 12., 13., 14., 15., 16., 17.]
    ]))


def test_legal_gripper_reset_does_not_change_legacy_observation_zero():
    from dwbc_isaaclab.tasks.widow_go1.observations import relative_joint_positions
    from dwbc_isaaclab.tasks.widow_go1.resets import DEFAULT_JOINT_POS
    actual = relative_joint_positions(DEFAULT_JOINT_POS[None], DEFAULT_JOINT_POS)
    torch.testing.assert_close(actual[:, :18], torch.zeros(1, 18))
    torch.testing.assert_close(actual[:, 18:], torch.tensor([[.015, -.015]]))
