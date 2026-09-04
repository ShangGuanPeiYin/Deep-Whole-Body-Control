"""Generate the deterministic excitation sequence used by both simulators."""

import argparse
from pathlib import Path

import numpy as np


def build_actions(steps: int) -> np.ndarray:
    time = np.arange(steps, dtype=np.float32)[:, None]
    joints = np.arange(18, dtype=np.float32)[None, :]
    actions = 0.15 * np.sin(0.07 * time + 0.31 * joints)
    actions[0] = 0.0
    return actions.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, actions=build_actions(args.steps))


if __name__ == "__main__":
    main()
