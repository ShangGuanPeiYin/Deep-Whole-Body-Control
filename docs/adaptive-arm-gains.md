# Adaptive Arm Gains

The default WidowGo1 task remains the frozen 18-dimensional position-action
contract.  The optional adaptive-gain branch is a distinct research contract:

```text
0:18   canonical leg and arm position targets
18:24  six arm proportional-gain deltas, in canonical arm-joint order
```

The six gain actions are delayed with the position actions, do not enter the
frozen 76-dimensional proprioceptive previous-action field, and directly modify
the arm stiffness.  Its damping is `2 * sqrt(clamp(p_default + delta, 1e-6))`.
The clamp preserves the old formula where it is defined and prevents invalid
negative stiffness from producing NaNs during research exploration.

The old implementation declared this feature but could not execute it under
its default configuration: it kept `num_actions=18`, and its enabled DOF
reordering selected only 18 entries before the controller read the extra six.
It therefore cannot serve as a numerical oracle for this branch.  The new
branch is explicitly versioned in checkpoints as
`dwbc-widow-go1-adaptive-gains-v1`; it is a repaired research feature, not a
claim of broken-legacy trajectory equivalence.

Run it with:

```bash
conda activate isaac-lab
python scripts/train.py --num-envs 32 --max-iterations 2 --seed 1 --headless \
  --adaptive-arm-gains --run-dir artifacts/integration-adaptive-gains/seed-1
```

The verified two-iteration run completed DAgger and PPO with this command on
the target RTX 3070 Ti.  Its generated checkpoint records 24 actions while
retaining the 860-dimensional observation contract.
