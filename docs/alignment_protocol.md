# Quantitative Alignment Protocol

Use the same scenario JSON, fixed action file, seed and tensor schema for the legacy and Isaac Lab exporters. Gate A compares discrete asset names exactly and continuous asset values with `asset_tolerances.yaml`. Gates B–D first reject schema, action-order and termination mismatches, then report absolute, relative and RMS differences for state, observation and reward fields.

Tolerance files are committed before candidate results are generated. A failed result may be explained or fixed, but its limits are not widened in place. Long PhysX trajectories are evaluated by short-horizon state error and long-horizon summary statistics because different PhysX releases are not expected to be step-identical.

Generate the legacy smoke trace from the new repository root:

```bash
conda run --no-capture-output -n dwbc python tools/export_legacy_trace.py --scenario configs/alignment/smoke_32_envs.yaml --seed 1 --out artifacts/legacy/smoke/seed-1
```
