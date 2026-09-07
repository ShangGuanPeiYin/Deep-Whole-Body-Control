# Deep Whole-Body Control for Isaac Lab

This repository migrates the WidowGo1 task to Isaac Lab while preserving the project's dual-reward, dual-value PPO implementation. The legacy repository at `../Deep-Whole-Body-Control` is a read-only reference.

## Environments

- `dwbc`: legacy Isaac Gym baseline export only.
- `isaac-lab`: Isaac Lab 2.3.2 development, tests and training.

The validated installation is `/home/xxs/research/IsaacLab` at tag `v2.3.2`. Activate it with `conda activate isaac-lab`, then run `python -m pip install -e '.[test]'` here. `environment.yml` records the Python/test requirements for recreating the named environment; install Isaac Lab from its `v2.3.2` source checkout before running simulator tests.

## Verification entry points

```bash
pytest -m "not isaaclab" -v
python scripts/smoke_env.py --num-envs 1 --steps 40 --headless
python tools/compare_rollouts.py --legacy artifacts/legacy/smoke/seed-1/trace.npz --candidate artifacts/isaaclab/smoke/seed-1/trace.npz --tolerances configs/alignment/rollout_tolerances.yaml --out artifacts/comparisons/smoke-seed-1.json
python scripts/train.py --num-envs 32 --max-iterations 2 --seed 1 --headless --run-dir artifacts/smoke
python scripts/train.py --num-envs 32 --max-iterations 2 --seed 1 --headless --adaptive-arm-gains --run-dir artifacts/adaptive
```

The simulation, comparison and training commands become available in the tasks that implement them. Gates A–E must pass before three-seed parity training begins.  The checked-in
`configs/alignment/training_calibration_32_envs.yaml` instead defines a short,
explicitly non-parity three-seed training regression (20 updates / 25,600
transitions per seed).  It is useful for catching broken training integration;
it does not establish full training convergence or satisfy Gate F.

Acceptance is incomplete; see [the current review](docs/reports/acceptance-review.md).
`--resume` restores the saved training configuration, model, optimizers,
iteration counter, task state and publicly exposed simulator state. PhysX
does not expose its warm-start/contact caches, so a fresh-process resume is
usable for continuing an experiment but is not a bitwise replay guarantee.
Playback infers the 18/24-action branch from the checkpoint automatically.

The optional adaptive-gain experiment has a separate, explicit 24-dimensional
action/checkpoint contract; see [docs/adaptive-arm-gains.md](docs/adaptive-arm-gains.md).
