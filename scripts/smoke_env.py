"""Launch a short deterministic WidowGo1 Isaac Lab rollout."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=40)
parser.add_argument("--action-script", type=Path)
parser.add_argument("--print-transition-keys", action="store_true")
parser.add_argument("--torque-supervision", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "source"))

import numpy as np
import torch

from dwbc_isaaclab.tasks.widow_go1.widow_go1_env import WidowGo1Env
from dwbc_isaaclab.tasks.widow_go1.widow_go1_env_cfg import WidowGo1EnvCfg


def main() -> int:
    cfg = WidowGo1EnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.torque_supervision = args.torque_supervision
    env = WidowGo1Env(cfg)
    try:
        obs, _ = env.reset(seed=cfg.seed)
        if args.action_script:
            action_rows = np.load(args.action_script)["actions"]
            if action_rows.shape[0] < args.steps or action_rows.shape[1] != 18:
                raise ValueError(f"invalid action script shape: {action_rows.shape}")
        else:
            action_rows = np.zeros((args.steps, 18), dtype=np.float32)
        transition = None
        for step in range(args.steps):
            actions = torch.as_tensor(action_rows[step], device=env.device).repeat(env.num_envs, 1)
            initial_arm_pos = env._robot.data.joint_pos[:, env._all_joint_ids[12:18]].clone()
            transition = env.step(actions)
            if args.torque_supervision:
                info = transition[-1]
                target = info["target_arm_torques"]
                assert target.shape == (env.num_envs, 6)
                assert torch.isfinite(target).all() and target.abs().max() > 0
                torch.testing.assert_close(info["current_arm_dof_pos"], initial_arm_pos)
        assert transition is not None
        obs, rewards, terminated, truncated, extras = transition
        policy = obs["policy"]
        if policy.shape != (env.num_envs, 860) or not torch.isfinite(policy).all():
            raise RuntimeError(f"invalid policy observation: {policy.shape}")
        if not torch.isfinite(env._robot.data.root_state_w).all():
            raise RuntimeError("non-finite robot root state")
        print(
            f"SMOKE PASS joints={len(env._robot.joint_names)} feet={len(env._feet_ids)} "
            f"obs={tuple(policy.shape)} reward={tuple(rewards.shape)} done={int((terminated | truncated).sum())}"
        )
        if args.print_transition_keys:
            print(f"extras={sorted(extras)}")
        return 0
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        env.close()
        launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
