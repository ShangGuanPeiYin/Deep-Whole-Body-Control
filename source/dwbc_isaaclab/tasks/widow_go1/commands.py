"""Legacy command sampling and iteration curriculum, independent of simulation."""

import torch


def curriculum_value(counter, initial, final, schedule=(0, 1)):
    start, end = schedule
    fraction = min(max((counter - start) / (end - start), 0.0), 1.0)
    return initial + fraction * (final - initial)


def sample_commands(count, generator, device, lin_range, yaw_range):
    commands = torch.zeros(count, 3, device=device)
    commands[:, 0] = lin_range[0] + (lin_range[1] - lin_range[0]) * torch.rand(count, generator=generator, device=device)
    commands[:, 2] = yaw_range[0] + (yaw_range[1] - yaw_range[0]) * torch.rand(count, generator=generator, device=device)
    active = (commands[:, 0] > 0.3) | (commands[:, 2].abs() > 0.6)
    return commands * active[:, None]


def sample_box_offsets(count, generator, device):
    signs = 2 * torch.randint(0, 2, (count,), generator=generator, device=device) - 1
    return signs * (0.1 + 0.2 * torch.rand(count, generator=generator, device=device))
