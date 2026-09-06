"""Run the legacy trainer read-only and emit Gate-F-compatible dual reward metrics.

This wrapper never edits the legacy checkout.  Its only intervention is an
in-memory observation of ``env.step`` and ``runner.log`` so the old WandB-only
logger is converted into portable JSONL metrics with the same per-step reward
definition used by the Isaac Lab runner.
"""

from __future__ import print_function

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace


class IterationMetrics(object):
    """Accumulate the two legacy reward channels for exactly one rollout."""

    def __init__(self, num_envs, steps_per_iteration):
        self.expected_count = int(num_envs) * int(steps_per_iteration)
        self.reset()

    def reset(self):
        self.leg_sum = 0.0
        self.arm_sum = 0.0
        self.count = 0

    @staticmethod
    def _sum_and_count(values):
        detached = values.detach()
        return float(detached.sum().item()), int(detached.numel())

    def record(self, leg_rewards, arm_rewards):
        leg_sum, leg_count = self._sum_and_count(leg_rewards)
        arm_sum, arm_count = self._sum_and_count(arm_rewards)
        if leg_count != arm_count:
            raise ValueError("legacy leg and arm reward batch sizes differ")
        self.leg_sum += leg_sum
        self.arm_sum += arm_sum
        self.count += leg_count

    def finish(self, iteration, fps):
        if self.count != self.expected_count:
            raise ValueError("expected {0} rewards, recorded {1}".format(self.expected_count, self.count))
        record = {
            "iteration": int(iteration),
            "leg_reward": self.leg_sum / self.count,
            "arm_reward": self.arm_sum / self.count,
            "sample_count": self.count,
            "fps": float(fps),
        }
        self.reset()
        return record


def _legacy_args(gymapi, args):
    return SimpleNamespace(
        device=args.sim_device,
        sim_device=args.sim_device,
        rl_device=args.rl_device,
        headless=True,
        physics_engine=gymapi.SIM_PHYSX,
        use_gpu=args.sim_device.startswith("cuda"),
        use_gpu_pipeline=args.sim_device.startswith("cuda"),
        subscenes=0,
        num_threads=0,
        num_envs=args.num_envs,
        seed=args.seed,
        max_iterations=args.iterations,
        resume=False,
        experiment_name=None,
        run_name=None,
        load_run="",
        checkpoint=-1,
        test=True,
        exptid=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path,
                        default=Path("/home/xxs/research/Deep-Whole-Body-Control"))
    parser.add_argument("--task", default="widowGo1")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations <= 0 or args.num_envs <= 0:
        parser.error("--iterations and --num-envs must be positive")
    if not args.legacy_root.is_dir():
        parser.error("legacy root does not exist: {0}".format(args.legacy_root))
    args.out.mkdir(parents=True, exist_ok=True)

    # Isaac Gym must load before Torch.  Keep these imports inside main so CPU
    # unit tests of IterationMetrics do not require the legacy environment.
    sys.path.insert(0, str(args.legacy_root / "legged_gym"))
    sys.path.insert(0, str(args.legacy_root / "rsl_rl"))
    from isaacgym import gymapi
    import torch
    from legged_gym.envs import task_registry

    legacy_args = _legacy_args(gymapi, args)
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.env.num_envs = args.num_envs
    env_cfg.seed = args.seed
    train_cfg.seed = args.seed
    train_cfg.runner.max_iterations = args.iterations
    env, _ = task_registry.make_env(args.task, args=legacy_args, env_cfg=env_cfg)
    runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=legacy_args, log_root=str(args.out / "legacy-run")
    )
    recorder = IterationMetrics(env.num_envs, runner.num_steps_per_env)
    original_step = env.step
    metrics_path = args.out / "metrics.jsonl"

    def observed_step(actions):
        output = original_step(actions)
        recorder.record(output[2], output[3])
        return output

    def write_metrics(locs):
        elapsed = float(locs["collection_time"] + locs["learn_time"])
        fps = runner.num_steps_per_env * env.num_envs / max(elapsed, 1.0e-9)
        record = recorder.finish(locs["it"], fps)
        with metrics_path.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)

    env.step = observed_step
    runner.log = write_metrics
    started = time.time()
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
    metadata = {
        "task": args.task,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "iterations": args.iterations,
        "steps_per_iteration": runner.num_steps_per_env,
        "simulator": "Isaac Gym Preview 4",
        "legacy_git_sha": subprocess.check_output(
            ["git", "-C", str(args.legacy_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "elapsed_seconds": time.time() - started,
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
