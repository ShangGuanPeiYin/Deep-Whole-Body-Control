# Deep Whole-Body Control for Isaac Lab

This repository migrates the WidowGo1 task to Isaac Lab while preserving the project's dual-reward, dual-value PPO implementation. The legacy repository at `../Deep-Whole-Body-Control` is a read-only reference.

## Environments

- `dwbc`: legacy Isaac Gym baseline export only.
- `dwbc-lab`: Isaac Lab 3.0.0 development, tests and training.

Create the project environment with `conda env create -f environment.yml`, activate it with `conda activate dwbc-lab`, install Isaac Lab 3.0.0 into that environment following its source installation script, then run `python -m pip install -e '.[test]'` here. Do not upgrade the existing `/home/xxs/research/IsaacLab` v2.3.2 checkout in place.

## Verification entry points

```bash
pytest -m "not isaaclab" -v
python scripts/smoke_env.py --num-envs 1 --steps 40 --headless
python tools/compare_rollouts.py --legacy artifacts/legacy/smoke/seed-1/trace.npz --candidate artifacts/isaaclab/smoke/seed-1/trace.npz --tolerances configs/alignment/rollout_tolerances.yaml --out artifacts/comparisons/smoke-seed-1.json
python scripts/train.py --num-envs 32 --max-iterations 1 --seed 1 --headless --run-dir artifacts/smoke
```

The simulation, comparison and training commands become available in the tasks that implement them. Gates A–E must pass before three-seed parity training begins.
