from dwbc_isaaclab.tasks.widow_go1.agents import dwbc_ppo_config
from dwbc_rsl_rl.runners.on_policy_runner import contract_for_policy


def test_adaptive_gain_training_config_and_checkpoint_contract_are_explicit():
    config = dwbc_ppo_config(adaptive_arm_gains=True)

    assert config["policy"]["adaptive_arm_gains"] is True
    assert config["policy"]["num_actions"] == 24
    assert len(config["policy"]["init_std"]) == 24
    assert contract_for_policy(config["policy"])["action_dim"] == 24


def test_playback_infers_adaptive_contract_and_rejects_inconsistent_metadata(tmp_path):
    import torch
    import pytest
    from dwbc_rsl_rl.runners.on_policy_runner import make_checkpoint, load_checkpoint

    config = dwbc_ppo_config(adaptive_arm_gains=True)
    checkpoint = make_checkpoint(2, {}, {}, config, contract=contract_for_policy(config['policy']))
    path = tmp_path / 'adaptive.pt'
    torch.save(checkpoint, path)
    assert load_checkpoint(path, infer_contract=True)['contract']['action_dim'] == 24
    checkpoint['config']['policy']['adaptive_arm_gains'] = False
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match='contract'):
        load_checkpoint(path, infer_contract=True)


def test_adaptive_torque_supervision_clamps_negative_stiffness_like_controller():
    import torch
    from dwbc_rsl_rl.modules import ActorCritic
    from dwbc_rsl_rl.algorithms import PPO
    model = ActorCritic(**dwbc_ppo_config(adaptive_arm_gains=True)['policy'])
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.actor.arm_head[-2].bias[6:] = torch.atanh(torch.tensor(-0.9))
    algorithm = PPO(model)
    algorithm.set_arm_default_coeffs(torch.full((6,), 5.), torch.ones(6), torch.ones(6))
    prediction = algorithm._predicted_arm_torques(torch.zeros(2, 860), torch.zeros(2, 6), torch.zeros(2, 6))
    torch.testing.assert_close(prediction, torch.full((2, 6), 1e-6), atol=1e-9, rtol=0)
