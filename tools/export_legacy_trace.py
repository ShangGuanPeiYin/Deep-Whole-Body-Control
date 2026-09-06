"""Export deterministic traces from the unmodified Isaac Gym WidowGo1 task."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "source"))

from dwbc_isaaclab.tasks.widow_go1.contracts import (
    POLICY_ACTION_NAMES,
    ROBOT_JOINT_NAMES,
    validate_trace_arrays,
)


def build_metadata(seed: int, steps: int, num_envs: int, git_sha: str, config_sha256: str) -> dict:
    return {
        "schema_version": 1,
        "simulator": "Isaac Gym Preview 4",
        "task": "widowGo1",
        "seed": seed,
        "steps": steps,
        "num_envs": num_envs,
        "git_sha": git_sha,
        "config_sha256": config_sha256,
        "action_order": list(POLICY_ACTION_NAMES),
        "joint_order": list(ROBOT_JOINT_NAMES),
        "tensor_shapes": {
            "obs": [steps, num_envs, 860],
            "actions": [steps, num_envs, 18],
            "dof_pos": [steps, num_envs, 20],
            "dof_vel": [steps, num_envs, 20],
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _legacy_args(gymapi, scenario: dict) -> SimpleNamespace:
    device = scenario["sim_device"]
    return SimpleNamespace(
        device=device,
        sim_device=device,
        rl_device=scenario["rl_device"],
        headless=bool(scenario["headless"]),
        physics_engine=gymapi.SIM_PHYSX,
        use_gpu=device.startswith("cuda"),
        use_gpu_pipeline=device.startswith("cuda"),
        subscenes=0,
        num_threads=0,
        num_envs=int(scenario["num_envs"]),
    )


def make_legacy_env(legacy_root: Path, scenario: dict, seed: int, *, disable_randomization: bool = False):
    sys.path.insert(0, str(legacy_root / "legged_gym"))
    sys.path.insert(0, str(legacy_root / "rsl_rl"))
    from isaacgym import gymapi  # imported before torch as required by Isaac Gym
    import torch
    from legged_gym.envs import task_registry

    env_cfg, _ = task_registry.get_cfgs(scenario["task"])
    env_cfg.env.num_envs = int(scenario["num_envs"])
    env_cfg.seed = seed
    if disable_randomization:
        for name in (
            "randomize_friction", "randomize_base_mass", "randomize_base_com",
            "randomize_motor", "randomize_gripper_mass", "push_robots",
        ):
            if hasattr(env_cfg.domain_rand, name):
                setattr(env_cfg.domain_rand, name, False)
    env, _ = task_registry.make_env(
        name=scenario["task"], args=_legacy_args(gymapi, scenario), env_cfg=env_cfg
    )
    return env


def record_trace(legacy_root: Path, scenario: dict, seed: int, action_path: Path) -> dict[str, np.ndarray]:
    env = make_legacy_env(legacy_root, scenario, seed)
    if 'initial_snapshot' in scenario:
        from initial_snapshot import capture_legacy
        capture_legacy(env, PROJECT_ROOT / scenario['initial_snapshot'].format(seed=seed),
                       sanitize_state_only_joint_limits=bool(scenario.get('sanitize_state_only_joint_limits', False)))
    import torch
    from substep_diagnostics import SubstepRecorder
    substeps = SubstepRecorder()
    if scenario.get('substep_diagnostic', False):
        compute_torques = env._compute_torques
        def record_control(actions):
            torque = compute_torques(actions)
            substeps.record(position=env.ig2raisim(env.dof_pos),
                            velocity=env.ig2raisim(env.dof_vel),
                            torque=env.ig2raisim(torque),
                            applied_torque=env.ig2raisim(torque))
            return torque
        env._compute_torques = record_control
    foot_names = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")
    sensor_names = [env.body_names[index] for index in env.feet_indices]
    sensor_order = [sensor_names.index(name) for name in foot_names]
    foot_body_ids = [env.body_names.index(name) for name in foot_names]
    sensor_diagnostic = {}
    if scenario.get('sensor_diagnostic', False):
        original_compute_torques = env._compute_torques
        def diagnostic_compute_torques(actions):
            env.gym.refresh_rigid_body_state_tensor(env.sim)
            sensor_diagnostic['before'] = env.rigid_body_state[:, foot_body_ids].clone()
            return original_compute_torques(actions)
        env._compute_torques = diagnostic_compute_torques
    action_script = np.load(action_path)["actions"]
    if action_script.shape != (int(scenario["steps"]), 18):
        raise ValueError(f"action script must have shape {(scenario['steps'], 18)}, got {action_script.shape}")
    records: dict[str, list[np.ndarray]] = {name: [] for name in (
        "obs", "actions", "leg_reward", "arm_reward", "dones",
        "root_state", "dof_pos", "dof_vel", "ee_state", "foot_wrench", "foot_contact_force",
    )}
    if scenario.get('sensor_diagnostic', False):
        records['foot_state'] = []
        records['foot_state_before_last_substep'] = []
    for action_row in action_script:
        actions = torch.as_tensor(action_row, device=env.device).repeat(env.num_envs, 1)
        obs, _, leg_reward, arm_reward, dones, _ = env.step(actions)
        snapshots = {
            "obs": obs,
            "actions": actions,
            "leg_reward": leg_reward,
            "arm_reward": arm_reward,
            "dones": dones,
            "root_state": env.root_states,
            "dof_pos": env.ig2raisim(env.dof_pos),
            "dof_vel": env.ig2raisim(env.dof_vel),
            "ee_state": env.rigid_body_state[:, env.gripper_idx, :],
            "foot_wrench": env.force_sensor_tensor[:, sensor_order],
            "foot_contact_force": env.contact_forces[:, foot_body_ids],
        }
        if scenario.get('sensor_diagnostic', False):
            snapshots['foot_state'] = env.rigid_body_state[:, foot_body_ids]
            snapshots['foot_state_before_last_substep'] = sensor_diagnostic['before']
        for name, tensor in snapshots.items():
            records[name].append(tensor.detach().cpu().numpy().copy())
    arrays = {name: np.stack(values) for name, values in records.items()}
    arrays.update(substeps.arrays())
    return arrays


def write_trace_artifacts(out: Path, arrays: dict[str, np.ndarray]) -> None:
    """Keep diagnostic measurements separate from the frozen acceptance schema."""
    from dwbc_isaaclab.tasks.widow_go1.contracts import REQUIRED_TRACE_WIDTHS, REQUIRED_SCALAR_TRACES

    validate_trace_arrays(arrays)
    required = set(REQUIRED_TRACE_WIDTHS) | set(REQUIRED_SCALAR_TRACES)
    trace = {name: value for name, value in arrays.items() if name in required}
    diagnostics = {name: value for name, value in arrays.items() if name not in required}
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'trace.npz', **trace)
    if diagnostics:
        np.savez_compressed(out / 'diagnostics.npz', **diagnostics)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--legacy-root", type=Path,
        default=Path(os.environ.get("DWBC_LEGACY_ROOT", "/home/xxs/research/Deep-Whole-Body-Control")),
    )
    args = parser.parse_args()
    scenario_bytes = args.scenario.read_bytes()
    scenario = json.loads(scenario_bytes)
    action_path = Path(scenario["action_file"])
    arrays = record_trace(args.legacy_root, scenario, args.seed, action_path)
    write_trace_artifacts(args.out, arrays)
    git_sha = subprocess.check_output(
        ["git", "-C", str(args.legacy_root), "rev-parse", "HEAD"], text=True
    ).strip()
    metadata = build_metadata(
        args.seed, int(scenario["steps"]), int(scenario["num_envs"]), git_sha,
        hashlib.sha256(scenario_bytes).hexdigest(),
    )
    metadata["action_sha256"] = _sha256(action_path)
    if 'initial_snapshot' in scenario:
        metadata['initial_snapshot_sha256'] = _sha256(PROJECT_ROOT / scenario['initial_snapshot'].format(seed=args.seed))
    metadata["source_sha256"] = {
        str(path.relative_to(args.legacy_root)): _sha256(path)
        for path in (
            args.legacy_root / "legged_gym/legged_gym/envs/widowGo1/widowGo1.py",
            args.legacy_root / "legged_gym/legged_gym/envs/widowGo1/widowGo1_config.py",
            args.legacy_root / "legged_gym/legged_gym/utils/math.py",
        )
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    explicit = subprocess.check_output(["conda", "list", "--explicit", "-n", "dwbc"], text=True)
    (args.out / "conda-dwbc-explicit.txt").write_text(explicit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
