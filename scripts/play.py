"""Play a schema-validated DWBC checkpoint in Isaac Lab."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=500)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "source"))

import torch

from dwbc_isaaclab.tasks.widow_go1.agents import dwbc_ppo_config
from dwbc_isaaclab.tasks.widow_go1.legacy_adapter import LegacyRunnerAdapter
from dwbc_isaaclab.tasks.widow_go1.widow_go1_env import WidowGo1Env
from dwbc_isaaclab.tasks.widow_go1.widow_go1_env_cfg import WidowGo1EnvCfg
from dwbc_rsl_rl.runners import OnPolicyRunner


def main() -> int:
    cfg = WidowGo1EnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    env = None
    try:
        env = WidowGo1Env(cfg)
        adapter = LegacyRunnerAdapter(env)
        runner = OnPolicyRunner(adapter, dwbc_ppo_config(), args.checkpoint.parent, device=args.device)
        runner.load(args.checkpoint, load_optimizer=False)
        policy = runner.get_inference_policy(use_history=True)
        obs, _, _ = adapter.reset()
        with torch.inference_mode():
            for _ in range(args.steps):
                actions = policy(obs)
                obs, _, _, _, _, _ = adapter.step(actions)
        print(f"PLAY PASS steps={args.steps} envs={args.num_envs}")
        return 0
    finally:
        if env is not None:
            env.close()
        launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())

