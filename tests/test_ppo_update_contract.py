import torch

from dwbc_rsl_rl.algorithms.ppo import PPO, compute_ppo_losses
from dwbc_rsl_rl.modules.actor_critic import ActorCritic


def test_deterministic_dual_ppo_loss_fixture_is_differentiable():
    new_log_prob = torch.tensor([[0.1, -0.2], [0.0, 0.3]], requires_grad=True)
    old_log_prob = torch.zeros(2, 2)
    advantages = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    values = torch.tensor([[1.2, 1.8], [0.3, -0.1]], requires_grad=True)
    old_values = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
    returns = torch.tensor([[1.5, 1.0], [0.25, 0.5]])
    entropy = torch.tensor([[2.0, 1.0], [2.0, 1.0]])

    losses = compute_ppo_losses(
        new_log_prob,
        old_log_prob,
        advantages,
        values,
        old_values,
        returns,
        entropy,
        clip_param=0.2,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        mixing_ratio=0.25,
    )

    assert set(losses) == {"loss", "surrogate_loss", "value_loss", "entropy"}
    torch.testing.assert_close(losses["surrogate_loss"], torch.tensor(-1.0299517), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(losses["value_loss"], torch.tensor(0.2731250), atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(losses["entropy"], torch.tensor(1.5), atol=0, rtol=0)
    losses["loss"].backward()
    assert torch.isfinite(new_log_prob.grad).all()
    assert torch.isfinite(values.grad).all()


def test_one_real_ppo_update_preserves_all_custom_metrics():
    torch.manual_seed(11)
    model = ActorCritic()
    algorithm = PPO(model, torque_supervision=False, num_learning_epochs=1, num_mini_batches=1)
    algorithm.init_storage(num_envs=2, num_transitions_per_env=2)
    obs = torch.zeros(2, 860)
    for step in range(2):
        algorithm.act(obs, obs)
        algorithm.process_env_step(
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0, 4.0]),
            torch.tensor([False, step == 1]),
            {"time_outs": torch.tensor([False, step == 1])},
        )
    algorithm.compute_returns(obs)

    metrics = algorithm.update()

    assert set(metrics) == {
        "value_loss", "surrogate_loss", "torque_loss", "priv_reg_loss",
        "value_mixing_ratio", "torque_weight", "priv_reg_coef",
    }
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert algorithm.storage.step == 0
