# Migration verification status — 2026-09-06

**Superseding review:** [acceptance-review.md](acceptance-review.md) corrects
the solver impossibility claim and unfrozen 40,000-update Gate F interpretation
below, and restores the omitted outstanding requirements. Historical measurements
below are retained as evidence, not an all-gates acceptance decision.

Status: INCOMPLETE. Training integration works; physics parity and final
three-seed research acceptance have not passed.

## Verified evidence

- CPU suite: 68 passed in 1.91 s, using `isaac-lab` and
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
norm for contact flags and local Z for foot reward. The candidate originally used
world-frame contact forces, which is not the same observable.

`artifacts/legacy/sensor-diagnostic/seed-1` records both: the first environment
has nonzero sensor wrench while its first-step contact forces are all zero.
The RMS force magnitudes across the trace are 7.8713 and 31.4813 respectively.
Newton–Euler reconstruction has analytic tests. A last-substep finite-difference experiment in
`artifacts/legacy/sensor-acceleration/seed-1` did not reproduce legacy readings
(force RMS residual approximately 9.10). This experiment also adds rigid-body
tensor refreshes and must not be substituted for the frozen baseline trace.
Further backend sensor investigation is required; no sensor parity is claimed.

## Collision geometry and optional-control correction — 2026-09-06

The expanded USD audit now records all 25 collision-shape descriptors instead
of only their count.  A standalone `tools/audit_usd.py --usd ... --headless`
run initializes a minimal simulation context (required by Isaac Sim 5.1) and
writes the geometry report.  Runtime USD inspection confirms all four feet are
`Sphere(radius=0.02)` and their terrain contact pair uses the authored
`contact_offset=0.01`, `rest_offset=0.0`.  The earlier hypotheses that the
contact split was caused by absent foot colliders, an incorrect foot radius, or
an un-authored terrain offset are rejected.

With the legal state-only finger snapshot, both engines agree exactly on the
first recorded positions, actions, and control torque.  Their first simulated
substep already differs in joint velocity only when foot contact is present;
the maximum observed difference is 10.2678 rad/s.  The same controlled
free-fall setup does not exhibit this immediate split.  This is evidence for a
PhysX 4 (Isaac Gym Preview 4) versus PhysX 5 (Isaac Sim 5.1) contact-solver
boundary, not evidence for silently changed task control.  The strict
state-trajectory Gate B–D comparator remains FAIL and its tolerances have not
been changed.

The old repository's optional adaptive-arm-gain branch was also structurally
unexecutable: its default 18-action configuration and enabled reorder path
discard the six gain outputs before control.  The new repository repairs it as
an explicitly versioned 24-action research branch, preserving the position and
gain formula while retaining the frozen 860-observation default task.  Unit
coverage verifies distribution, storage, delayed control and checkpoint
contracts.  A real 32-environment / two-iteration GPU run completed DAgger and
PPO and saved `artifacts/integration-adaptive-gains/seed-1/model_2.pt` with
contract `dwbc-widow-go1-adaptive-gains-v1`.

The candidate now experimentally reconstructs local wrench using native PhysX
body acceleration, not finite differences. This is connected to contact flags
and the foot reward, but has NOT been empirically validated as equivalent to the
legacy sensor. Passing summed reward tolerances is not sensor acceptance.

## Complete-snapshot rerun — 2026-09-06

The shared snapshot now includes box mass, full inertia, COM and materials and
rejects incomplete snapshots before simulator writes. Both exporters support
seed-specific snapshot paths. Artifacts are
`artifacts/{legacy,isaaclab}/matched-complete/seed-1`.
The strict comparator was rerun with unchanged tolerance SHA-256
`ff4b828c97673367e530b31aa8cc43f134eb54cf70e263e059f14ad7b9f6b1b0`.
Result: FAIL. Actions agree exactly; leg/arm reward maximum absolute errors
are 0.00574847/0.00222018 and pass their trajectory thresholds. Observations
first fail at step 24, dones at step 21, root/joint velocities at step 0,
joint positions at step 22, and EE state at step 1. This supersedes earlier
partial-snapshot results as current evidence, without deleting those artifacts.

There is also a schema failure: the added `foot_wrench` and `foot_contact_force`
diagnostics are not fields in the frozen tolerance file. The original comparison
is preserved at `artifacts/comparisons/matched-complete-seed-1.json`. Both
exporters now write those measurements to `diagnostics.npz`, keeping the
acceptance `trace.npz` schema unchanged. Two new serialization tests passed
after first failing for the missing implementation. Re-serialization of the
existing arrays into `/tmp/dwbc-separated-comparison` removed the schema failure
without changing any field errors; the result is preserved at
`artifacts/comparisons/matched-complete-separated-seed-1.json`. Original trace
artifacts were not overwritten. This was not a new simulation run.

The first-step maximum errors are already 0.911 rad/s in base angular velocity
and 8.083 rad/s in joint velocity. Native-acceleration sensor reconstruction
also differs substantially (up to approximately 33 N per force component at
step zero). Initial root pose/raw velocity were previously checked; further
investigation must isolate substep state, applied torque and contact response,
not infer sensor equivalence from summed rewards.

Independent same-state execution of all active legacy reward methods now passes
per-term and total comparisons at atol 1e-7 / rtol 1e-6, including nonzero yaw.
This verifies reward formulas, not their simulator-provided inputs. Control
preserves the actual legacy wrap at native index 10 of its 18-wide vector
(canonical RR thigh), distinct from the waist wrap in 20-wide observations.

The default legacy termination method was independently replayed on 1024 shared
states, including the height and timeout boundaries: boolean failures/timeouts
agree exactly. This covers the frozen empty contact-termination body list, not
an optional nonempty configuration. A separate read-only reviewer checked delay
reset, substep PD feedback, joint-state cache invalidation and the tensor API
inertia frame. No additional defect was established at these boundaries; do not
present an unverified solver-flag change as a fix.

## Remaining work

1. Gates B–D are a documented cross-PhysX boundary: the full asset, terrain,
   reset, control and reward contracts are frozen, yet same-state ground contact
   diverges at the first simulated substep between PhysX 4 and 5.  Resolving a
   strict state-trajectory PASS requires an upstream-compatible solver or a
   different simulator-agnostic acceptance criterion; it must not be obtained
   by loosening the frozen tolerances.
2. Sensor equivalence and exact simulator-state resume have not been proved.
3. Gate F remains intentionally pending.  Its legacy configuration requires
   40,000 updates × 32 environments × 40 steps = 51,200,000 transitions per
   seed (153,600,000 across seeds 1/2/3), and the Gate F precondition B–E is
   not satisfied.  The reproducible 20-update / 25,600-transition-per-seed
   calibration in `configs/alignment/training_calibration_32_envs.yaml` is
   diagnostic only, never a replacement for Gate F.

## Plan correction — 2026-09-06

The original plan coupled the deliverable "trainable in Isaac Lab" to exact
cross-engine trajectory identity.  Evidence now distinguishes the two:
trainability, task semantics, asset fidelity, control, custom PPO, checkpoint
contracts and optional research controls can be verified in Isaac Lab, whereas
contact-by-contact identity cannot be honestly certified between Isaac Gym
Preview 4 / PhysX 4 and Isaac Sim 5.1 / PhysX 5 with the available APIs.

Therefore the migration ships with the strict B–D comparator still fail-closed
and a separate, explicitly labelled three-seed training calibration.  A future
research run may promote a simulator-agnostic policy-quality protocol only
after its acceptance metric, horizon and legacy-derived intervals are frozen;
it must not retroactively turn this calibration into a Gate F PASS.

## Three-seed training calibration — 2026-09-06

Both implementations completed the declared 32-environment, 40-step,
20-update calibration for seeds 1/2/3 (25,600 transitions per seed).  Each
metrics file contains exactly 20 records and each Isaac Lab run wrote a
schema-checked `model_20.pt`.  The legacy wrapper observes the old runner
read-only and emits the same per-step dual-reward mean used by the migrated
runner.

| Endpoint | Final leg reward, seeds 1/2/3 | median | Final arm reward, seeds 1/2/3 | median | median FPS |
| --- | --- | ---: | --- | ---: | ---: |
| Isaac Gym Preview 4 | -0.0632042, -0.0600958, -0.0617442 | -0.0617442 | 0.00141731, 0.00166509, 0.00148065 | 0.00148065 | 1312.15 |
| Isaac Lab 2.3.2 / Isaac Sim 5.1 | -0.0579640, -0.0589694, -0.0617211 | -0.0589694 | 0.00172138, 0.00170532, 0.00190761 | 0.00172138 | 1139.03 |

Artifacts are ignored by Git but reproducible from
`configs/alignment/training_calibration_32_envs.yaml` under
`artifacts/training-calibration/{legacy,isaaclab}/seed-{1,2,3}`.  This is an
end-to-end training-stability result, **not Gate F** and not a claim that the
two engines produce the same learning curve.

The renewed Isaac Lab capacity benchmark (`artifacts/benchmark-gpu-20260906.json`)
passed 32/64/128/256 environments at 998/1875/3259/5992 environment steps per
second.  At 256 environments, the minimum observed whole-device free memory
was 2.95 GB; the calibration nevertheless stays at 32 environments to match
the legacy setup and leave headroom for research instrumentation.

## Execution restriction

The earlier usage-quota restriction cleared. Host GPU asset export, import,
audit and matched-state trajectory execution resumed successfully in the
18:07–18:17 local session. It is no longer an active blocker.
