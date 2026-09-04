"""Export the Isaac Lab WidowGo1 trace using the frozen legacy schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scenario", type=Path, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--out", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
scenario_bytes = args.scenario.read_bytes()
scenario = json.loads(scenario_bytes)
args.headless = bool(scenario["headless"])
args.device = scenario["sim_device"]
launcher = AppLauncher(args)
sys.path.insert(0, str(PROJECT_ROOT / "source"))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from dwbc_isaaclab.tasks.widow_go1.contracts import validate_trace_arrays
from dwbc_isaaclab.tasks.widow_go1.widow_go1_env import WidowGo1Env
from dwbc_isaaclab.tasks.widow_go1.widow_go1_env_cfg import WidowGo1EnvCfg
from tools.export_legacy_trace import _sha256, build_metadata


def _state_wxyz_to_legacy_xyzw(state: torch.Tensor) -> torch.Tensor:
    return torch.cat((state[:, :3], state[:, 4:7], state[:, 3:4], state[:, 7:]), dim=-1)


def main() -> int:
    cfg = WidowGo1EnvCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = int(scenario["num_envs"])
    cfg.sim.device = scenario["sim_device"]
    cfg.terrain.terrain_generator.seed = args.seed
    env = None
    try:
        env = WidowGo1Env(cfg)
        env.reset(seed=args.seed)
        action_path = PROJECT_ROOT / scenario["action_file"]
        action_rows = np.load(action_path)["actions"]
        expected = (int(scenario["steps"]), 18)
        if action_rows.shape != expected:
            raise ValueError(f"action script must have shape {expected}, got {action_rows.shape}")
        records: dict[str, list[np.ndarray]] = {
            name: [] for name in (
                "obs", "actions", "leg_reward", "arm_reward", "dones",
                "root_state", "dof_pos", "dof_vel", "ee_state",
            )
        }
        for action_row in action_rows:
            actions = torch.as_tensor(action_row, device=env.device).repeat(env.num_envs, 1)
            obs, _, terminated, truncated, extras = env.step(actions)
            snapshots = {
                "obs": obs["policy"],
                "actions": actions,
                "leg_reward": extras["leg_reward"],
                "arm_reward": extras["arm_reward"],
                "dones": terminated | truncated,
                "root_state": _state_wxyz_to_legacy_xyzw(env._robot.data.root_state_w),
                "dof_pos": env._robot.data.joint_pos[:, env._all_joint_ids],
                "dof_vel": env._robot.data.joint_vel[:, env._all_joint_ids],
                "ee_state": _state_wxyz_to_legacy_xyzw(env._robot.data.body_state_w[:, env._ee_id]),
            }
            for name, tensor in snapshots.items():
                records[name].append(tensor.detach().cpu().numpy().copy())
        arrays = {name: np.stack(rows) for name, rows in records.items()}
        validate_trace_arrays(arrays)
        args.out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out / "trace.npz", **arrays)
        git_sha = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
        metadata = build_metadata(
            args.seed, int(scenario["steps"]), int(scenario["num_envs"]), git_sha,
            hashlib.sha256(scenario_bytes).hexdigest(),
        )
        metadata["simulator"] = "Isaac Lab 2.3.2 / Isaac Sim 5.1.0"
        metadata["action_sha256"] = _sha256(action_path)
        metadata["source_sha256"] = {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path)
            for path in (
                PROJECT_ROOT / "source/dwbc_isaaclab/tasks/widow_go1/widow_go1_env.py",
                PROJECT_ROOT / "source/dwbc_isaaclab/tasks/widow_go1/widow_go1_env_cfg.py",
                PROJECT_ROOT / "source/dwbc_isaaclab/tasks/widow_go1/rewards.py",
            )
        }
        (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return 0
    finally:
        if env is not None:
            env.close()
        launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
