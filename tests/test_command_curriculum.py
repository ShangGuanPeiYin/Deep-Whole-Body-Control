import torch

from dwbc_isaaclab.tasks.widow_go1.commands import curriculum_value, sample_commands, sample_box_offsets
from dwbc_isaaclab.tasks.widow_go1.goals import goal_collision_mask, cart_to_sphere


def test_curriculum_and_deadband():
    assert curriculum_value(0, 0, 0.9) == 0
    assert curriculum_value(1, 0, 0.9) == 0.9
    generator = torch.Generator().manual_seed(42)
    commands = sample_commands(100, generator, 'cpu', (0, 0.2), (-0.5, 0.5))
    assert torch.count_nonzero(commands) == 0
    offsets = sample_box_offsets(100, generator, 'cpu')
    assert (offsets.abs() >= 0.1).all() and (offsets.abs() <= 0.3).all()
    assert (offsets < 0).any() and (offsets > 0).any()


def test_goal_collision_rejects_body_and_underground():
    points = torch.tensor([[0.1, 0, -0.2], [0.5, 0, 0.2], [0.5, 0, -0.6]])
    spheres = cart_to_sphere(points)
    assert goal_collision_mask(spheres, spheres).tolist() == [True, False, True]
