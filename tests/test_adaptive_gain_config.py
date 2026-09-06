from dwbc_isaaclab.tasks.widow_go1.agents import dwbc_ppo_config
from dwbc_rsl_rl.runners.on_policy_runner import contract_for_policy


def test_adaptive_gain_training_config_and_checkpoint_contract_are_explicit():
    config = dwbc_ppo_config(adaptive_arm_gains=True)

    assert config["policy"]["adaptive_arm_gains"] is True
    assert config["policy"]["num_actions"] == 24
    assert len(config["policy"]["init_std"]) == 24
    assert contract_for_policy(config["policy"])["action_dim"] == 24
