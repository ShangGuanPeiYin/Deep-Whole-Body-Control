from pathlib import Path

import pytest
import torch

from dwbc_rsl_rl.runners.on_policy_runner import (
    CHECKPOINT_SCHEMA_VERSION,
    OnPolicyRunner,
    load_checkpoint,
    make_checkpoint,
)


def test_checkpoint_records_complete_frozen_contract(tmp_path: Path):
    checkpoint = make_checkpoint(7, {"weight": torch.tensor([1.0])}, {"state": {}}, {"seed": 1})

    assert checkpoint["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert checkpoint["iteration"] == 7
    assert checkpoint["contract"] == {
        "version": "dwbc-widow-go1-v1",
        "observation_dim": 860,
        "action_dim": 18,
        "value_channels": ["leg", "arm"],
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    assert load_checkpoint(path)["iteration"] == 7


def test_checkpoint_contract_mismatch_fails_closed(tmp_path: Path):
    checkpoint = make_checkpoint(1, {}, {}, {})
    checkpoint["contract"]["action_dim"] = 24
    path = tmp_path / "bad.pt"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="contract"):
        load_checkpoint(path)


class _FakeAdapter:
    num_envs = 2
    device = "cpu"

    def __init__(self):
        self.obs = torch.zeros(2, 860)

    def reset(self):
        return self.obs.clone(), self.obs.clone(), {}

    def step(self, actions):
        assert actions.shape == (2, 18)
        leg = torch.full((2,), 0.003)
        arm = torch.full((2,), 0.002)
        dones = torch.zeros(2, dtype=torch.bool)
        return self.obs.clone(), self.obs.clone(), leg, arm, dones, {"time_outs": dones}


def test_runner_executes_dagger_then_ppo_and_saves_loadable_checkpoint(tmp_path: Path):
    config = {
        "policy": {},
        "algorithm": {
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "torque_supervision": False,
            "dagger_update_freq": 2,
        },
        "runner": {"num_steps_per_env": 2, "save_interval": 10},
    }
    runner = OnPolicyRunner(_FakeAdapter(), config, tmp_path, device="cpu")

    metrics, path = runner.learn(2)

    assert metrics["iteration"] == 1
    assert metrics["surrogate_loss"] != 0.0
    assert path.is_file()
    assert load_checkpoint(path)["iteration"] == 2
    resumed = OnPolicyRunner(_FakeAdapter(),config,tmp_path / 'resumed',device='cpu')
    resumed.load(path)
    assert resumed.algorithm.counter == runner.algorithm.counter == 2
    assert resumed.algorithm.hist_encoder_optimizer.state_dict()['state']


def test_runner_trains_torque_supervision_with_environment_coefficients(tmp_path):
    class SupervisedAdapter(_FakeAdapter):
        def arm_default_coefficients(self):
            return torch.full((6,), 5.), torch.full((6,), .5), torch.zeros(6)

        def step(self, actions):
            obs, critic, leg, arm, done, infos = super().step(actions)
            infos.update(target_arm_torques=torch.full((2, 6), 2.),
                         current_arm_dof_pos=torch.zeros(2, 6),
                         current_arm_dof_vel=torch.zeros(2, 6))
            return obs, critic, leg, arm, done, infos

    config = {
        'policy': {},
        'algorithm': {'num_learning_epochs': 1, 'num_mini_batches': 1,
                      'torque_supervision': True, 'torque_supervision_schedule': (.1, 0, 100),
                      'dagger_update_freq': 2},
        'runner': {'num_steps_per_env': 2, 'save_interval': 10},
    }
    runner = OnPolicyRunner(SupervisedAdapter(), config, tmp_path, device='cpu')
    metrics, _ = runner.learn(2)
    assert metrics['torque_loss'] > 0
    assert metrics['torque_weight'] > 0
