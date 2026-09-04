import torch

from dwbc_isaaclab.tasks.widow_go1.goals import (
    cart_to_sphere,
    orientation_error_xyzw,
    sphere_to_cart,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


def test_quaternion_boundary_round_trip_and_identity_error():
    xyzw = torch.tensor([[0.1, 0.2, 0.3, 0.9]])
    assert torch.equal(wxyz_to_xyzw(xyzw_to_wxyz(xyzw)), xyzw)
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    assert torch.equal(orientation_error_xyzw(identity, identity), torch.zeros(1, 3))


def test_spherical_goal_round_trip():
    cart = torch.tensor([[0.2, -0.1, 0.3], [0.4, 0.2, -0.1]])
    assert torch.allclose(sphere_to_cart(cart_to_sphere(cart)), cart, atol=1e-6)
