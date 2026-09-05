import json
from pathlib import Path

import numpy as np
import pytest

from tools.compare_training import compare_training_runs, median_within_interval


def test_interval_is_inclusive():
    assert median_within_interval(np.array([9.0, 10.0, 11.0]), 9.5, 10.5)
    assert not median_within_interval(np.array([1.0, 2.0, 3.0]), 2.1, 4.0)


def test_three_seed_gate_reports_statistics_and_threshold(tmp_path: Path):
    runs = []
    for seed, values in ((1, [1.0, 3.0]), (2, [2.0, 4.0]), (3, [3.0, 5.0])):
        path = tmp_path / f"seed-{seed}.jsonl"
        path.write_text("\n".join(json.dumps({"iteration": i, "leg_reward": v, "arm_reward": v / 2})
                                   for i, v in enumerate(values)), encoding="utf-8")
        runs.append({"seed": seed, "metrics": str(path)})
    config = {
        "required_seeds": [1, 2, 3],
        "runs": runs,
        "gates": {
            "leg_reward": {"final_median": [3.5, 4.5], "threshold": 3.0},
            "arm_reward": {"final_median": [1.5, 2.5], "threshold": 1.5},
        },
    }

    report = compare_training_runs(config)

    assert report["gate_f_passed"]
    assert report["metrics"]["leg_reward"]["mean"] == 4.0
    assert report["metrics"]["leg_reward"]["first_threshold_iteration"] == 1


def test_missing_seed_fails_closed():
    config = {"required_seeds": [1, 2, 3], "runs": [{"seed": 1, "metrics": "unused"}], "gates": {}}
    with pytest.raises(ValueError, match="missing required seeds"):
        compare_training_runs(config)

