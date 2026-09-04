"""Stable tensor and naming contracts shared by simulation and learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Mapping, Sequence


LEG_JOINT_NAMES = (
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
)
ARM_JOINT_NAMES = (
    "widow_waist", "widow_shoulder", "widow_elbow",
    "widow_forearm_roll", "widow_wrist_angle", "widow_wrist_rotate",
)
GRIPPER_JOINT_NAMES = ("widow_left_finger", "widow_right_finger")
POLICY_ACTION_NAMES = LEG_JOINT_NAMES + ARM_JOINT_NAMES
ROBOT_JOINT_NAMES = POLICY_ACTION_NAMES + GRIPPER_JOINT_NAMES


@dataclass(frozen=True)
class ObservationLayout:
    proprio: ClassVar[int] = 76
    privileged: ClassVar[int] = 24
    history: ClassVar[int] = 10
    flat: ClassVar[int] = 860

    @classmethod
    def proprio_slices(cls) -> dict[str, slice]:
        return {
            "orientation": slice(0, 2),
            "angular_velocity": slice(2, 5),
            "dof_pos": slice(5, 25),
            "dof_vel": slice(25, 45),
            "previous_action": slice(45, 63),
            "feet_contacts": slice(63, 67),
            "command": slice(67, 70),
            "ee_goal": slice(70, 73),
            "ee_orientation_error": slice(73, 76),
        }


REQUIRED_TRACE_WIDTHS = {
    "obs": ObservationLayout.flat,
    "actions": len(POLICY_ACTION_NAMES),
    "root_state": 13,
    "dof_pos": len(ROBOT_JOINT_NAMES),
    "dof_vel": len(ROBOT_JOINT_NAMES),
    "ee_state": 13,
}
REQUIRED_SCALAR_TRACES = ("leg_reward", "arm_reward", "dones")


def build_name_index(names: Sequence[str]) -> dict[str, int]:
    index: dict[str, int] = {}
    for position, name in enumerate(names):
        if name in index:
            raise ValueError(f"duplicate name: {name}")
        index[name] = position
    return index


def validate_joint_names(actual: Sequence[str], required: Sequence[str]) -> None:
    actual_index = build_name_index(actual)
    required_index = build_name_index(required)
    missing = [name for name in required if name not in actual_index]
    unexpected = [name for name in actual if name not in required_index]
    if missing or unexpected:
        raise ValueError(f"missing={missing}; unexpected={unexpected}")


def validate_trace_arrays(arrays: Mapping[str, object]) -> None:
    missing = [name for name in (*REQUIRED_TRACE_WIDTHS, *REQUIRED_SCALAR_TRACES) if name not in arrays]
    if missing:
        raise ValueError(f"missing trace arrays: {missing}")
    first_shape = getattr(arrays["obs"], "shape", ())
    if len(first_shape) != 3:
        raise ValueError(f"obs must have shape (steps, envs, 860), got {first_shape}")
    prefix = first_shape[:2]
    for name, width in REQUIRED_TRACE_WIDTHS.items():
        shape = getattr(arrays[name], "shape", ())
        if shape != (*prefix, width):
            raise ValueError(f"{name} must have shape {(*prefix, width)}, got {shape}")
    for name in REQUIRED_SCALAR_TRACES:
        shape = getattr(arrays[name], "shape", ())
        if shape != prefix:
            raise ValueError(f"{name} must have shape {prefix}, got {shape}")
