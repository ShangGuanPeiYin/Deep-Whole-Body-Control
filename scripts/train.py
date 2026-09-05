"""Train the migrated WidowGo1 task with the project-owned DWBC PPO."""

import argparse
import random
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--max-iterations", type=int, default=1)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--resume", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "source"))

import numpy as np
import torch

from dwbc_isaaclab.tasks.widow_go1.agents import dwbc_ppo_config
from dwbc_isaaclab.tasks.widow_go1.legacy_adapter import LegacyRunnerAdapter
from dwbc_isaaclab.tasks.widow_go1.widow_go1_env import WidowGo1Env
from dwbc_isaaclab.tasks.widow_go1.widow_go1_env_cfg import WidowGo1EnvCfg
from dwbc_rsl_rl.runners import OnPolicyRunner


def main() -> int:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = WidowGo1EnvCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.terrain.terrain_generator.seed = args.seed
    env = None
    try:
        env = WidowGo1Env(cfg)
        adapter = LegacyRunnerAdapter(env)
        runner = OnPolicyRunner(adapter, dwbc_ppo_config(), args.run_dir, device=args.device)
        if args.resume:
            runner.load(args.resume)
        _, checkpoint = runner.learn(args.max_iterations)
        print(f"TRAIN PASS checkpoint={checkpoint}")
        return 0
    finally:
        if env is not None:
            env.close()
        launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())

