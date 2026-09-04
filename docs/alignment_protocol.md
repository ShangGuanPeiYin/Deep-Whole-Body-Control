# Quantitative Alignment Protocol

Use the same scenario JSON, fixed action file, seed and tensor schema for the legacy and Isaac Lab exporters. Gate A compares discrete asset names exactly and continuous asset values with `asset_tolerances.yaml`. Gates B–D first reject schema, action-order and termination mismatches, then report absolute, relative and RMS differences for state, observation and reward fields.

Tolerance files are committed before candidate results are generated. A failed result may be explained or fixed, but its limits are not widened in place. Long PhysX trajectories are evaluated by short-horizon state error and long-horizon summary statistics because different PhysX releases are not expected to be step-identical.

Generate the legacy smoke trace from the new repository root:

```bash
conda run --no-capture-output -n dwbc python tools/export_legacy_trace.py --scenario configs/alignment/smoke_32_envs.yaml --seed 1 --out artifacts/legacy/smoke/seed-1
```

Generate and compare the Isaac Lab trace after accepting the Omniverse EULA:

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run --no-capture-output -n isaac-lab python tools/export_isaaclab_trace.py --scenario configs/alignment/smoke_32_envs.yaml --seed 1 --out artifacts/isaaclab/smoke/seed-1
conda run --no-capture-output -n isaac-lab python tools/compare_rollouts.py --legacy artifacts/legacy/smoke/seed-1/trace.npz --candidate artifacts/isaaclab/smoke/seed-1/trace.npz --tolerances configs/alignment/rollout_tolerances.yaml --out artifacts/comparisons/smoke-seed-1.json
```

`rollout_tolerances.yaml` is frozen before producing the first candidate trace. Actions and done flags are exact; continuous thresholds cover only the declared 0.8-second open-loop smoke horizon. Reports include maximum absolute, RMS and maximum relative error plus the first failing step for every field.
