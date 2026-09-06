import torch
import pytest

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


def test_enabled_supervision_rejects_missing_targets_instead_of_training_on_zeros():
    from dwbc_rsl_rl.algorithms.ppo import PPO
    from dwbc_rsl_rl.modules import ActorCritic
    algorithm = PPO(ActorCritic(), torque_supervision=True)
    algorithm.init_storage(2, 1)
    obs = torch.zeros(2, 860)
    algorithm.act(obs, obs)
    with pytest.raises(ValueError, match='target_arm_torques'):
        algorithm.process_env_step(torch.zeros(2), torch.zeros(2),
                                   torch.zeros(2, dtype=torch.bool), {})
    assert algorithm.storage.step == 0


def test_adaptive_gain_rollout_storage_uses_the_24_dimensional_policy_sample():
    from dwbc_rsl_rl.algorithms.ppo import PPO
    from dwbc_rsl_rl.modules import ActorCritic

    algorithm = PPO(ActorCritic(num_actions=24, adaptive_arm_gains=True), torque_supervision=False)
    algorithm.init_storage(2, 1)
    obs = torch.zeros(2, 860)
    actions = algorithm.act(obs, obs)

    assert actions.shape == (2, 24)
    assert algorithm.storage.actions.shape == (1, 2, 24)
