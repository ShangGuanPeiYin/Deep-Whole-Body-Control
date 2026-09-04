import numpy as np
import pytest

from dwbc_isaaclab.tasks.widow_go1.contracts import validate_trace_arrays
from tools.export_legacy_trace import build_metadata


def _valid_trace():
    return {
        "obs": np.zeros((4, 2, 860), dtype=np.float32),
        "actions": np.zeros((4, 2, 18), dtype=np.float32),
        "leg_reward": np.zeros((4, 2), dtype=np.float32),
        "arm_reward": np.zeros((4, 2), dtype=np.float32),
        "dones": np.zeros((4, 2), dtype=np.bool_),
        "root_state": np.zeros((4, 2, 13), dtype=np.float32),
        "dof_pos": np.zeros((4, 2, 20), dtype=np.float32),
        "dof_vel": np.zeros((4, 2, 20), dtype=np.float32),
        "ee_state": np.zeros((4, 2, 13), dtype=np.float32),
    }


def test_trace_schema_accepts_complete_consistent_arrays():
    validate_trace_arrays(_valid_trace())


def test_trace_schema_rejects_wrong_action_width():
    trace = _valid_trace()
    trace["actions"] = np.zeros((4, 2, 17), dtype=np.float32)
    with pytest.raises(ValueError, match="actions.*18"):
        validate_trace_arrays(trace)


def test_legacy_trace_metadata_freezes_order_and_shapes():
    metadata = build_metadata(seed=7, steps=4, num_envs=2, git_sha="abc123", config_sha256="f" * 64)
    assert metadata["seed"] == 7
    assert metadata["tensor_shapes"]["obs"] == [4, 2, 860]
    assert metadata["action_order"][0] == "FR_hip_joint"
    assert metadata["action_order"][-1] == "widow_wrist_rotate"
