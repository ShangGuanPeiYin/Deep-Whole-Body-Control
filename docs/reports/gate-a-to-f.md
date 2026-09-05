# Migration verification status — 2026-09-05

Status: INCOMPLETE. Training integration works; physics parity and final
three-seed research acceptance have not passed.

## Verified evidence

- CPU suite: 60 passed in 1.79 s, using `isaac-lab` and
  `pytest -p no:cacheprovider -q`.
- Real GPU 40-step smoke: 20 joints, 4 feet, observation `(1,860)`; both reward
  channels and termination extras present. Full-resolution terrain also ran
  with 32 environments.
- Two-iteration real GPU integration run completed DAgger then PPO and saved
  `artifacts/integration-smoke/seed-1/model_2.pt` and `metrics.jsonl`.
- Independent legacy network oracle: both encoding paths, values, log probability,
  entropy and gradients agree at atol 1e-7 / rtol 1e-6.
- Independent legacy GAE and one complete fixed-gain, non-torque-supervised PPO
  update agree, including compared losses and network parameters.
- Separate-process capacity benchmark passed 32/64/128/256 environments at
  approximately 1172/2290/3978/6770 environment steps per second. The original
  memory counters cover PyTorch only. Whole-device memory sampling was added
  afterward and still requires rerunning the benchmark.

## Quantitative physics results

Original seed-only comparison: failed. Legacy starts after construction while
Lab resets; independent random streams yield different initial conditions.
Keep `artifacts/comparisons/smoke-seed-1.json` as diagnostic evidence.

The shared initial-state comparison is saved as
`artifacts/comparisons/matched-initial-seed-1.json`. Actions match exactly.
Both reward trajectories satisfy existing tolerances (maximum absolute errors
0.00601127 leg and 0.00221275 arm). Observations, joint/base/EE state and done
flags still fail; done mismatch first occurs at step 19. This is not Gate B–D
acceptance. The snapshot still needs complete box physical properties, terrain
verification and post-reset random-stream handling before full parity claims.

Expanded Gate A passed on 2026-09-05 at 18:15 local time. The frozen reference is
`assets/widow_go1/legacy_full_mass_properties.json`; the importer now authors
principal inertias, principal-axis orientation and center of mass from the full
legacy tensors. `artifacts/asset-full-comparison.json` reports `passed: true`
with no failures, using the unchanged 1e-6 thresholds. The audit reconstructs
body-frame tensors rather than comparing only principal values.

The subsequent 40-step/32-environment run is preserved separately in
`artifacts/isaaclab/matched-full-inertia/seed-1`. Its comparison report,
`artifacts/comparisons/matched-full-inertia-seed-1.json`, still fails B–D:
actions agree exactly; arm reward passes (max error 0.00224025), but leg reward
does not (0.06225888). Base/joint velocities differ at step 0 and done flags
first differ at step 20. Do not reuse the earlier passing leg-reward result as
evidence for the current implementation. The CPU suite was rerun: 53 passed.

## Corrections made

- GPU access failure was a sandbox restriction, not an established driver crash.
- Restored mesh stride 1; original-resolution terrain runs on the GPU.
- Corrected box inertia indexing, missing encoder/backbone activations, arm yaw
  reward normalization and separate sensor body-name resolution.
- Added command curriculum/resampling, pushes, box random offsets, goal path
  rejection, and episode reward logging.
- Added history optimizer, algorithm counter and Torch RNG checkpoint state;
  exact simulator-state resume remains pending.
- Removed unsupported Gate F intervals. `training_seeds.yaml` fails closed until
  real baseline training produces predeclared intervals.
- Restored complete OSC targets and first-substep supervision snapshots. The
  original legacy OSC method agrees with the pure candidate calculation at
  atol 1e-7 / rtol 1e-6; analytic mass/gravity/singular-Jacobian fixtures pass.
  `scripts/smoke_env.py --num-envs 32 --steps 40 --headless --torque-supervision`
  passed with finite, nonzero targets and exact first-substep position samples.
- Runtime diagnostics found instance-proxy robot collision offsets of
  0.0004905–0.00187 m, not the legacy configured 0.01 m. Spawn-time overrides
  did not apply to those shapes. PhysX runtime setters now apply 0.01 m and
  a read-back assertion passes. The resulting comparison is preserved as
  `artifacts/comparisons/contact-offset-runtime-seed-1.json`; B–D still fail.
- An independent execution of both terrain generators found all 600x2000
  integer height samples identical for seed 1.
- Preserved native FL/FR/RL/RR ordering specifically for privileged motor
  strengths, while keeping control/proprioception canonical. Legal gripper
  initialization no longer shifts the legacy observation zero reference.

## Newly isolated sensor mismatch (not resolved)

The old task creates its force sensors without overriding defaults. The bundled
Isaac Gym documentation (`docs/_sources/programming/forcesensors.rst.txt`)
defines default readings as net six-axis body wrench in sensor-local axes,
including forward dynamics and solver forces. The task uses its six-dimensional
norm for contact flags and local Z for foot reward. The candidate currently uses
world-frame contact forces, which is not the same observable.

`artifacts/legacy/sensor-diagnostic/seed-1` records both: the first environment
has nonzero sensor wrench while its first-step contact forces are all zero.
The RMS force magnitudes across the trace are 7.8713 and 31.4813 respectively.
Newton–Euler reconstruction has analytic tests but has NOT been connected to the
task: a last-substep finite-difference experiment in
`artifacts/legacy/sensor-acceleration/seed-1` did not reproduce legacy readings
(force RMS residual approximately 9.10). This experiment also adds rigid-body
tensor refreshes and must not be substituted for the frozen baseline trace.
Further backend sensor investigation is required; no sensor parity is claimed.

## Remaining work

1. Expanded asset audit and matched physics rerun are complete; investigate the
   remaining first-step dynamics differences without changing tolerances.
2. Complete snapshot and shared-state task replay, including sensor semantics,
   termination reasons and per-term metrics. Resolve Gates B–D failures.
3. Restore optional adaptive-gain control end-to-end. Real OSC targets are now
   implemented and tested; optional PPO/control-branch parity is still pending.
4. Verify complete resume semantics and DAgger/optional-branch optimizer parity.
5. Freeze a real three-seed legacy training baseline, run candidate trials, and
   evaluate Gate F. No such baseline training has completed yet.
6. Complete review, documentation, commits and authorized repository integration.

## Execution restriction

The earlier usage-quota restriction cleared. Host GPU asset export, import,
audit and matched-state trajectory execution resumed successfully in the
18:07–18:17 local session. It is no longer an active blocker.
