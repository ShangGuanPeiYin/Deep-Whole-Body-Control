import torch

from dwbc_rsl_rl.modules.actor_critic import ActorCritic


def test_dual_heads_and_latents_preserve_frozen_widths():
    torch.manual_seed(7)
    model = ActorCritic(860, 18, 76, 24, 10)
    obs = torch.zeros(4, 860)

    actions, values = model(obs, use_history=False)

    assert actions.shape == (4, 18)
    assert values.shape == (4, 2)
    assert model.infer_priv_latent(obs).shape == (4, 20)
    assert model.infer_history_latent(obs).shape == (4, 20)


def test_distribution_splits_leg_and_arm_statistics():
    model = ActorCritic(860, 18, 76, 24, 10, adaptive_arm_gains=True)
    obs = torch.zeros(3, 860)

    actions = model.act(obs)
    log_prob = model.get_actions_log_prob(actions)

    assert actions.shape == (3, 18)
    assert model.arm_gain_delta.shape == (3, 6)
    assert log_prob.shape == (3, 2)
    assert model.entropy.shape == (3, 2)

