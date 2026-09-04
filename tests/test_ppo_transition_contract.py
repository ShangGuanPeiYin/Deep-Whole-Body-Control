import torch

from dwbc_rsl_rl.algorithms.ppo import (
    bootstrap_timeouts,
    decay_schedule_value,
    mix_advantages,
    schedule_value,
    stack_dual_rewards,
)


def test_dual_rewards_and_timeout_bootstrap_are_exact():
    rewards = stack_dual_rewards(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
    values = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    timeouts = torch.tensor([True, False])

    actual = bootstrap_timeouts(rewards, values, timeouts, gamma=0.5)

    torch.testing.assert_close(actual, torch.tensor([[6.0, 13.0], [2.0, 4.0]]))


def test_old_value_mixing_schedule_and_formula_are_preserved():
    advantages = torch.tensor([[1.0, 10.0]])
    assert schedule_value(0, (0.5, 2, 4)) == 0.0
    assert schedule_value(4, (0.5, 2, 4)) == 0.25
    assert decay_schedule_value(4, (0.5, 2, 4)) == 0.25
    torch.testing.assert_close(mix_advantages(advantages, 0.25), torch.tensor([[3.5, 10.25]]))
