import torch

from tools.run_legacy_training import IterationMetrics


def test_legacy_metric_recorder_writes_per_step_dual_reward_means():
    recorder = IterationMetrics(num_envs=2, steps_per_iteration=2)
    recorder.record(torch.tensor([1.0, 3.0]), torch.tensor([10.0, 30.0]))
    recorder.record(torch.tensor([5.0, 7.0]), torch.tensor([50.0, 70.0]))

    record = recorder.finish(iteration=4, fps=123.0)

    assert record == {
        "iteration": 4,
        "leg_reward": 4.0,
        "arm_reward": 40.0,
        "sample_count": 4,
        "fps": 123.0,
    }
