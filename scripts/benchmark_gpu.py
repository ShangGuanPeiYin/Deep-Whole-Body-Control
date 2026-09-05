"""Benchmark safe Isaac Lab environment counts on the local GPU."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--counts", default="32,64,128,256")
parser.add_argument("--steps", type=int, default=40)
parser.add_argument("--out", type=Path, default=Path("artifacts/benchmark-gpu.json"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
counts = [int(value) for value in args.counts.split(',')]
if len(counts) > 1:
    results = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for count in counts:
        child_out = args.out.with_name(args.out.stem + f'-{count}.json')
        command = [sys.executable, '-u', __file__, '--counts', str(count),
                   '--steps', str(args.steps), '--out', str(child_out), '--device', args.device]
        if args.headless:
            command.append('--headless')
        process = subprocess.run(command, timeout=300)
        if process.returncode != 0 or not child_out.is_file():
            results.append({'num_envs':count, 'passed':False, 'error':'worker failed'})
            break
        record = json.loads(child_out.read_text())['results'][0]
        results.append(record)
        if not record['passed']:
            break
    args.out.write_text(json.dumps({'device':args.device,'steps':args.steps,'results':results},indent=2)+'\n')
    print(json.dumps(results,indent=2),flush=True)
    raise SystemExit(0 if results and all(row['passed'] for row in results) else 1)
launcher = AppLauncher(args)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "source"))

import torch

from dwbc_isaaclab.tasks.widow_go1.widow_go1_env import WidowGo1Env
from dwbc_isaaclab.tasks.widow_go1.widow_go1_env_cfg import WidowGo1EnvCfg


def main() -> int:
    results = []
    for count in [int(value) for value in args.counts.split(",")]:
        env = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            cfg = WidowGo1EnvCfg()
            cfg.scene.num_envs = count
            cfg.sim.device = args.device
            env = WidowGo1Env(cfg)
            env.reset()
            actions = torch.zeros(count, 18, device=env.device)
            min_free_bytes, total_bytes = torch.cuda.mem_get_info()
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(args.steps):
                env.step(actions)
                min_free_bytes = min(min_free_bytes,torch.cuda.mem_get_info()[0])
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            results.append({
                "num_envs": count,
                "passed": True,
                "fps": count * args.steps / elapsed,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "device_total_bytes": total_bytes,
                "minimum_device_free_bytes": min_free_bytes,
                "memory_note": "Device free memory includes other processes and simulation allocations; PyTorch counters do not.",
            })
        except torch.cuda.OutOfMemoryError as exc:
            results.append({"num_envs": count, "passed": False, "error": f"OOM: {exc}"})
            break
        finally:
            if env is not None:
                env.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"device": args.device, "steps": args.steps, "results": results}, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return 0 if results and results[0]["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        launcher.app.close()
