import pytest

from dwbc_isaaclab.tasks.widow_go1.contracts import (
    ObservationLayout,
    POLICY_ACTION_NAMES,
    ROBOT_JOINT_NAMES,
    build_name_index,
    validate_joint_names,
)


def test_policy_contract_matches_legacy_external_order():
    assert POLICY_ACTION_NAMES[:6] == (
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    )
    assert POLICY_ACTION_NAMES[-6:] == (
        "widow_waist", "widow_shoulder", "widow_elbow",
        "widow_forearm_roll", "widow_wrist_angle", "widow_wrist_rotate",
    )
    assert len(POLICY_ACTION_NAMES) == 18
    assert len(ROBOT_JOINT_NAMES) == 20


def test_observation_layout_covers_each_column_once():
    assert ObservationLayout.proprio == 76
    assert ObservationLayout.privileged == 24
    assert ObservationLayout.history == 10
    assert ObservationLayout.flat == 860
    slices = ObservationLayout.proprio_slices()
    assert [(value.start, value.stop) for value in slices.values()] == [
        (0, 2), (2, 5), (5, 25), (25, 45), (45, 63),
        (63, 67), (67, 70), (70, 73), (73, 76),
    ]


def test_name_validation_rejects_duplicates_and_missing_names():
    with pytest.raises(ValueError, match="duplicate"):
        build_name_index(("FR_hip_joint", "FR_hip_joint"))
    with pytest.raises(ValueError, match="missing=.*widow_elbow"):
        validate_joint_names(("FR_hip_joint",), ("FR_hip_joint", "widow_elbow"))
