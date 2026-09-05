"""Export the non-randomized Isaac Gym asset properties used by WidowGo1."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "source"))
sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_usd import build_asset_report
from tools.export_legacy_trace import make_legacy_env


def body_property_row(name: str, mass: float, inertia: tuple[float, float, float]) -> dict:
    return {"name": name, "mass": float(mass), "inertia": [float(value) for value in inertia]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--legacy-root", type=Path,
        default=Path(os.environ.get("DWBC_LEGACY_ROOT", "/home/xxs/research/Deep-Whole-Body-Control")),
    )
    args = parser.parse_args()
    scenario = {
        "task": "widowGo1", "num_envs": 1, "steps": 1,
        "sim_device": "cuda:0", "rl_device": "cuda:0", "headless": True,
    }
    env = make_legacy_env(args.legacy_root, scenario, seed=1, disable_randomization=True)
    env_handle = env.envs[0]
    actor_handle = env.actor_handles[0]
    dof_props = env.gym.get_actor_dof_properties(env_handle, actor_handle)
    joints = []
    for index, name in enumerate(env.dof_names):
        joints.append({
            "name": name,
            "limits": [float(dof_props["lower"][index]), float(dof_props["upper"][index])],
            "stiffness": float(dof_props["stiffness"][index]),
            "damping": float(dof_props["damping"][index]),
            "effort_limit": float(dof_props["effort"][index]),
        })
    body_props = env.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
    bodies = [
        body_property_row(name, prop.mass, (prop.inertia.x.x, prop.inertia.y.y, prop.inertia.z.z))
        for name, prop in zip(env.body_names, body_props)
    ]
    for row, prop in zip(bodies, body_props):
        row['inertia_tensor'] = [[getattr(getattr(prop.inertia,a),b) for b in 'xyz'] for a in 'xyz']
        row['center_of_mass'] = [prop.com.x,prop.com.y,prop.com.z]
    collider_count = len(env.gym.get_actor_rigid_shape_properties(env_handle, actor_handle))
    report = build_asset_report(joints, bodies, collider_count)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
