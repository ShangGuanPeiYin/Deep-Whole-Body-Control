import torch


def test_net_sensor_at_rest_reports_zero_not_support_force():
    from dwbc_isaaclab.tasks.widow_go1.sensors import net_wrench_local
    zero = torch.zeros(1, 4, 3)
    actual = net_wrench_local(torch.ones(1, 4), torch.eye(3).expand(1, 4, 3, 3),
                              zero, zero, zero, zero)
    torch.testing.assert_close(actual, torch.zeros(1, 4, 6))


def test_sensor_includes_gyroscopic_and_reference_point_moments():
    from dwbc_isaaclab.tasks.widow_go1.sensors import net_wrench_local
    vector = torch.tensor([[[1., 2., 3.]]])
    actual = net_wrench_local(
        torch.tensor([[2.]]), torch.diag(torch.tensor([2., 3., 4.]))[None, None],
        vector, vector, vector, torch.tensor([[[.1, 0., 0.]]]),
    )
    torch.testing.assert_close(actual, torch.tensor([[[2., 4., 6., 8., -.6, 14.4]]]))
