import torch

from dwbc_rsl_rl.storage.rollout_storage import RolloutStorage, Transition


def _transition(reward, done):
    return Transition(
        observations=torch.zeros(1, 860),
        critic_observations=torch.zeros(1, 860),
        actions=torch.zeros(1, 18),
        rewards=torch.tensor([reward], dtype=torch.float32),
        dones=torch.tensor([done]),
        values=torch.zeros(1, 2),
        actions_log_prob=torch.zeros(1, 2),
        action_mean=torch.zeros(1, 18),
        action_sigma=torch.ones(1, 18),
        target_arm_torques=torch.arange(6, dtype=torch.float32).view(1, 6),
        current_arm_dof_pos=torch.zeros(1, 6),
        current_arm_dof_vel=torch.zeros(1, 6),
    )


def test_dual_gae_numeric_fixture_and_torque_width():
    storage = RolloutStorage(1, 2, (860,), (860,), (18,))
    storage.add_transitions(_transition([1.0, 10.0], False))
    storage.add_transitions(_transition([2.0, 20.0], True))

    storage.compute_returns(torch.zeros(1, 2), gamma=1.0, lam=1.0)

    torch.testing.assert_close(
        storage.returns[:, 0], torch.tensor([[3.0, 30.0], [2.0, 20.0]])
    )
    assert storage.advantages.shape == (2, 1, 2)
    torch.testing.assert_close(storage.advantages.mean(), torch.tensor(0.0), atol=1e-7, rtol=0)
    torch.testing.assert_close(storage.advantages.std(), torch.tensor(1.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(storage.target_arm_torques[0, 0], torch.arange(6, dtype=torch.float32))


def test_rollout_overflow_fails_closed():
    storage = RolloutStorage(1, 1, (860,), (860,), (18,))
    storage.add_transitions(_transition([0.0, 0.0], False))
    try:
        storage.add_transitions(_transition([0.0, 0.0], False))
    except AssertionError as exc:
        assert "overflow" in str(exc).lower()
    else:
        raise AssertionError("rollout overflow was not rejected")

