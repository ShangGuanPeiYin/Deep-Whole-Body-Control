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
parser.add_argument('--torque-supervision', action='store_true')
parser.add_argument('--adaptive-arm-gains', action='store_true')
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
from dwbc_rsl_rl.runners import OnPolicyRunner, load_checkpoint


def main() -> int:
    training_config = dwbc_ppo_config(adaptive_arm_gains=args.adaptive_arm_gains)
    if args.torque_supervision:
        training_config['algorithm']['torque_supervision'] = True
        training_config['algorithm']['torque_supervision_schedule'] = (.1, 1000, 1000)
    if args.resume:
        resumed_checkpoint = load_checkpoint(args.resume, infer_contract=True)
        training_config = resumed_checkpoint['config']
        if 'environment_state' in resumed_checkpoint:
            args.seed = resumed_checkpoint['environment_state']['seed']
            args.num_envs = resumed_checkpoint['environment_state']['num_envs']
            print('RESUME: restores public simulator and task state; internal PhysX caches are not serialized, so this is not bitwise replay.')
        else:
            print('RESUME: older checkpoint has no environment state; simulation episodes restart.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = WidowGo1EnvCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.terrain.terrain_generator.seed = args.seed
    cfg.torque_supervision = training_config['algorithm'].get('torque_supervision', False)
    cfg.adaptive_arm_gains = training_config['policy'].get('adaptive_arm_gains', False)
    cfg.action_space = training_config['policy'].get('num_actions', 18)
    env = None
    try:
        env = WidowGo1Env(cfg)
        adapter = LegacyRunnerAdapter(env)
        runner = OnPolicyRunner(adapter, training_config, args.run_dir, device=args.device)
        if args.resume:
            runner.load(args.resume)
        _, checkpoint = runner.learn(args.max_iterations)
        print(f"TRAIN PASS checkpoint={checkpoint}")
        return 0
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
