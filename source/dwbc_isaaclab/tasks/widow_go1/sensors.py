"""Net body wrench reconstruction for legacy default six-axis force sensors."""

import torch


def net_wrench_local(masses, inertias, linear_acceleration, angular_acceleration,
                     angular_velocity, center_of_mass):
    """Newton–Euler net wrench at the link origin, all vectors in link axes.

    Unlike contact forces, the default Gym sensor includes gravity and internal
    constraint forces. A static supported link therefore has zero net wrench.
    Inertia is about the COM; the moment is translated to the sensor at origin.
    """
    force = masses[..., None] * linear_acceleration
    inertia_alpha = (inertias @ angular_acceleration[..., None]).squeeze(-1)
    angular_momentum = (inertias @ angular_velocity[..., None]).squeeze(-1)
    moment = inertia_alpha + torch.cross(angular_velocity, angular_momentum, dim=-1)
    moment += torch.cross(center_of_mass, force, dim=-1)
    return torch.cat((force, moment), dim=-1)
