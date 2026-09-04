# DWBC Isaac Lab Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an Isaac Lab-native WidowGo1 task that preserves the custom DWBC PPO and proves quantified alignment with the frozen Isaac Gym baseline.

**Architecture:** Start with one Isaac Lab `DirectRLEnv`. A project-owned adapter converts Lab transitions to the legacy rollout protocol; the project-owned `dwbc_rsl_rl` package retains dual reward/value semantics. The old repository is an executable, read-only oracle and supplies deterministic traces.

**Tech Stack:** Isaac Lab 2.3.2, Isaac Sim 5.1.0.0/PhysX, Python 3.11.15, PyTorch 2.7.0+cu128, USD, PyTest, TensorBoard.

**Spec:** `docs/superpowers/specs/2026-09-04-dwbc-isaaclab-migration-design.md`

## Global Constraints

- Use the installed Conda environment `isaac-lab` for new work; use `dwbc` only to export legacy reference traces.
- Pin Isaac Lab 2.3.2 and record exact Isaac Sim/PyTorch/driver versions in `docs/environment-lock.md`.
- Never modify `../Deep-Whole-Body-Control` during migration.
- Canonical policy action order is FR, FL, RR, RL (hip/thigh/calf each), followed by `widow_waist`, `widow_shoulder`, `widow_elbow`, `widow_forearm_roll`, `widow_wrist_angle`, `widow_wrist_rotate`.
- Policy action width is 18; robot joint width is 20 including state-only grippers.
- Proprioception/privileged/history/flattened widths are 76/24/10/860; quaternions at every module boundary are `xyzw`.
- Resolve all semantic IDs by name. Return scalar leg reward to Isaac Lab and retain `arm_reward`, reward terms, torque targets, `time_outs`, and termination reason in extras.
- Do not begin parity training before Gates A–E pass. Start all GPU runs at 32 environments; benchmark before increasing the count on the 8 GB RTX 3070 Ti.

## Locked File Structure

| Path | Responsibility |
| --- | --- |
| `source/dwbc_isaaclab/tasks/widow_go1/contracts.py` | Names, layouts, validation, extras keys. |
| `source/dwbc_isaaclab/tasks/widow_go1/widow_go1_env*.py` | DirectRLEnv lifecycle and config. |
| `source/dwbc_isaaclab/tasks/widow_go1/{control,goals,observations,rewards,resets,legacy_adapter}.py` | Focused task logic and legacy bridge. |
| `source/dwbc_rsl_rl/{modules,storage,algorithms,runners}` | Custom actor-critic, storage, PPO and runner only. |
| `tools/` | Asset audit, trace export, rollout/training comparison. |
| `configs/alignment/` | Seeds, actions, tolerances and baseline ranges. |
| `tests/` | Contract tests and marker-gated Isaac Lab tests. |

### Task 1: Initialize installable, reproducible project shell

**Files:**
- Create: `pyproject.toml`, `README.md`, `docs/environment-lock.md`, `docs/baseline-manifest.md`
- Create: `source/dwbc_isaaclab/__init__.py`, `source/dwbc_rsl_rl/__init__.py`, `tests/test_project_layout.py`

**Interfaces:** Produces editable packages `dwbc_isaaclab` and `dwbc_rsl_rl`; later tasks use the documented `isaac-lab` commands.

- [ ] **Step 1: Write the failing layout test**

```python
from pathlib import Path
def test_project_declares_two_packages():
    root = Path(__file__).parents[1]
    assert 'include = ["dwbc_isaaclab*", "dwbc_rsl_rl*"]' in (root / "pyproject.toml").read_text()
    assert (root / "docs/environment-lock.md").is_file()
```

- [ ] **Step 2: Run it and confirm failure**

Run: `pytest tests/test_project_layout.py -v`

Expected: FAIL because `pyproject.toml` is absent.

- [ ] **Step 3: Implement metadata and baseline records**

Create setuptools metadata with `package-dir = {"" = "source"}` and `include = ["dwbc_isaaclab*", "dwbc_rsl_rl*"]`. README must specify `conda activate isaac-lab`, `pip install -e .`, unit test, smoke, trace and training commands. Environment lock records exact Lab/Sim/Python/Torch/driver/GPU versions. Baseline manifest records old Git SHA, hashes of dirty legacy files/assets, `dwbc` package list, old WidowGo1 config, seeds 1/2/3, and the existing headless smoke command.

- [ ] **Step 4: Verify**

Run: `pytest tests/test_project_layout.py -v && python -c "import dwbc_isaaclab, dwbc_rsl_rl"`

Expected: PASS and both imports succeed.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml README.md docs source tests/test_project_layout.py
git commit -m "build: initialize reproducible Isaac Lab project"
```

### Task 2: Freeze data and physical contracts; export legacy traces

**Files:**
- Create: `source/dwbc_isaaclab/tasks/widow_go1/{__init__,contracts}.py`
- Create: `docs/action_observation_contract.md`, `configs/alignment/{smoke_32_envs.yaml,asset_tolerances.yaml,fixed_actions_v1.npz}`
- Create: `tools/{export_legacy_trace,audit_usd}.py`, `tests/{test_contracts,test_trace_schema,test_asset_contract}.py`

**Interfaces:** Produces immutable `POLICY_ACTION_NAMES`, `ROBOT_JOINT_NAMES`, `ObservationLayout`, `validate_joint_names()`, `artifacts/legacy/.../trace.npz`, and an asset audit JSON.

- [ ] **Step 1: Write failing contract tests**

```python
import pytest
from dwbc_isaaclab.tasks.widow_go1.contracts import POLICY_ACTION_NAMES, ObservationLayout, validate_joint_names
def test_frozen_dimensions_and_order():
    assert POLICY_ACTION_NAMES[:3] == ("FR_hip_joint", "FR_thigh_joint", "FR_calf_joint")
    assert (len(POLICY_ACTION_NAMES), ObservationLayout.proprio, ObservationLayout.privileged, ObservationLayout.history, ObservationLayout.flat) == (18, 76, 24, 10, 860)
def test_missing_joint_is_rejected():
    with pytest.raises(ValueError, match="widow_elbow"):
        validate_joint_names(("FR_hip_joint",), ("FR_hip_joint", "widow_elbow"))
```

- [ ] **Step 2: Verify failure**

Run: `pytest tests/test_contracts.py -v`

Expected: FAIL because `contracts` does not exist.

- [ ] **Step 3: Implement contract, exporter, and audit**

Define all 20 joint names, 18 action names, and slices: orientation `0:2`, angular velocity `2:5`, position `5:25`, velocity `25:45`, previous action `45:63`, feet `63:67`, command `67:70`, EE goal `70:73`, orientation error `73:76`, privileged `76:100`, history `100:860`. Reject duplicate, missing and unexpected names. Exporter invokes the old repository read-only under `dwbc`, seeds Python/NumPy/Torch, replays `fixed_actions_v1.npz`, and writes `obs`, `actions`, `leg_reward`, `arm_reward`, `dones`, `root_state`, `dof_pos`, `dof_vel`, `ee_state`, reward terms/extras plus metadata hashes. Audit USD to JSON with all body/joint names, limits, masses, inertia, colliders and actuator properties; compare continuous values against the YAML tolerance before Gate A can pass.

- [ ] **Step 4: Verify reference artifact**

Run: `pytest tests/test_contracts.py tests/test_trace_schema.py tests/test_asset_contract.py -v && conda run --no-capture-output -n dwbc python tools/export_legacy_trace.py --scenario configs/alignment/smoke_32_envs.yaml --seed 1 --out artifacts/legacy/smoke/seed-1`

Expected: tests PASS; trace has action width 18, observation width 860, and metadata JSON.

- [ ] **Step 5: Commit**

```bash
git add source/dwbc_isaaclab/tasks/widow_go1 docs configs tools tests .gitignore
git commit -m "test: freeze WidowGo1 contracts and legacy traces"
```

### Task 3: Import and validate WidowGo1 USD asset

**Files:**
- Create: `assets/widow_go1/source_urdf/README.md`, `assets/widow_go1/import_config/widow_go1_urdf_import.py`, `assets/widow_go1/usd/README.md`
- Modify: `tools/audit_usd.py`, `tests/test_asset_contract.py`

**Interfaces:** Consumes the hashed source URDF and produces `assets/widow_go1/usd/widow_go1.usd` and `artifacts/asset-report.json` accepted by Task 4.

- [ ] **Step 1: Write failing importer report test**

```python
from dwbc_isaaclab.tasks.widow_go1.contracts import ROBOT_JOINT_NAMES
from tools.audit_usd import compare_asset_report
def test_identical_asset_report_has_no_failures():
    report = {"joint_names": list(ROBOT_JOINT_NAMES), "joint_limits": {n: [-1., 1.] for n in ROBOT_JOINT_NAMES}}
    assert compare_asset_report(report, report, {"joint_limit_abs": 1e-6}).failures == []
```

- [ ] **Step 2: Verify failure**

Run: `pytest tests/test_asset_contract.py -v`

Expected: FAIL because comparison is absent.

- [ ] **Step 3: Implement explicit import/audit**

Copy the original URDF with SHA-256 provenance; import it with explicit drive mode, stiffness, damping, effort, fixed-joint, collision and mass settings. Audit must fail on discrete name/count differences and report absolute/relative errors for limits, mass, inertia, stiffness, damping and effort. Do not infer any physics setting from USD defaults.

- [ ] **Step 4: Verify Gate A**

Run: `pytest tests/test_asset_contract.py -v && python tools/audit_usd.py --usd assets/widow_go1/usd/widow_go1.usd --out artifacts/asset-report.json`

Expected: PASS; report contains exactly the required 20 joints and all declared physics fields.

- [ ] **Step 5: Commit**

```bash
git add assets tools/audit_usd.py tests/test_asset_contract.py
git commit -m "feat: add audited WidowGo1 USD asset"
```

### Task 4: Implement DirectRLEnv scene, deterministic reset and control

**Files:**
- Create: `source/dwbc_isaaclab/tasks/widow_go1/{widow_go1_env_cfg,widow_go1_env,resets,goals,control}.py`
- Create: `scripts/smoke_env.py`, `tests/{test_reset_contract,test_quaternion_convention,test_control_contract}.py`

**Interfaces:** Produces `WidowGo1EnvCfg`, `WidowGo1Env(DirectRLEnv)`, `sample_reset_state()`, `ActionDelayBuffer`, `_pre_physics_step()` and `_apply_action()`.

- [ ] **Step 1: Write failing reset/control tests**

```python
import torch
from dwbc_isaaclab.tasks.widow_go1.resets import sample_reset_state
from dwbc_isaaclab.tasks.widow_go1.control import ActionDelayBuffer
def test_fixed_seed_reset_is_reproducible():
    assert torch.equal(sample_reset_state(2, 7, "cpu").root_pose, sample_reset_state(2, 7, "cpu").root_pose)
def test_two_step_action_delay():
    delay = ActionDelayBuffer(1, 18, 2, "cpu")
    assert torch.equal(delay.push(torch.ones(1, 18)), torch.zeros(1, 18))
    delay.push(torch.full((1, 18), 2.))
    assert torch.equal(delay.push(torch.full((1, 18), 3.)), torch.ones(1, 18))
```

- [ ] **Step 2: Verify failure**

Run: `pytest tests/test_reset_contract.py tests/test_control_contract.py -v`

Expected: FAIL because helpers do not exist.

- [ ] **Step 3: Implement scene lifecycle**

Use `DirectRLEnv`; config declares 18 actions, dt, decimation, terrain, robot USD, end-effector frame and four foot sensors. `_setup_scene()` uses `find_joints()`/`find_bodies()` then validates names. Reset uses a dedicated Torch generator, legacy root/default 20-joint pose, randomization, spherical goals and zeroes histories. `_pre_physics_step()` validates `(N,18)`, clips and applies legacy two-step action delay. `_apply_action()` maps canonical action names to articulation indices and applies configured PD/effort at every decimation substep; grippers receive no policy action. Store arm torque/position/velocity extras only when supervision is enabled. Implement explicit `xyzw` identity/error utilities.

- [ ] **Step 4: Verify Gate B/control smoke**

Run: `pytest tests/test_reset_contract.py tests/test_quaternion_convention.py tests/test_control_contract.py -v && python scripts/smoke_env.py --num-envs 1 --steps 40 --headless --action-script configs/alignment/fixed_actions_v1.npz`

Expected: PASS; output reports 20 resolved joints, four feet, finite state, and no gripper action.

- [ ] **Step 5: Commit**

```bash
git add source/dwbc_isaaclab/tasks/widow_go1 scripts/smoke_env.py tests
git commit -m "feat: add DirectRLEnv reset and delayed control"
```

### Task 5: Port observations, goals, rewards and terminations

**Files:**
- Create: `source/dwbc_isaaclab/tasks/widow_go1/{observations,rewards}.py`
- Modify: `source/dwbc_isaaclab/tasks/widow_go1/{goals,widow_go1_env}.py`
- Create: `tests/{test_observation_contract,test_reward_contract}.py`

**Interfaces:** Produces `_get_observations() -> dict[str, Tensor]`, `_get_rewards() -> Tensor`, `_get_dones() -> tuple[Tensor, Tensor]` and typed transition extras.

- [ ] **Step 1: Write failing semantic tests**

```python
import torch
from dwbc_isaaclab.tasks.widow_go1.observations import build_legacy_observation
from dwbc_isaaclab.tasks.widow_go1.rewards import combine_rewards
def test_full_observation_width():
    assert build_legacy_observation(torch.zeros(2,76), torch.zeros(2,24), torch.zeros(2,10,76)).shape == (2,860)
def test_leg_scalar_keeps_arm_channel():
    total, extras = combine_rewards(torch.tensor([2.]), torch.tensor([3.]))
    assert total.tolist() == [2.] and extras["arm_reward"].tolist() == [3.]
```

- [ ] **Step 2: Verify failure**

Run: `pytest tests/test_observation_contract.py tests/test_reward_contract.py -v`

Expected: FAIL because builders do not exist.

- [ ] **Step 3: Implement exact old layout and task terms**

Compose the 76 fields in the frozen order; append 24 privileged values (mass, friction, motor-strength-minus-one) and ten history frames to form 860. Port spherical goal interpolation and all nonzero old leg/arm reward functions as named pure functions. Preserve legacy scaling, clipping, `/100`, timeout/fall/task failure conditions and per-term episode sums. Lab receives scalar leg reward while extras expose `leg_reward`, `arm_reward`, term values, `time_outs`, termination reason and torque keys.

- [ ] **Step 4: Verify**

Run: `pytest tests/test_observation_contract.py tests/test_reward_contract.py -v && python scripts/smoke_env.py --num-envs 2 --steps 40 --headless --print-transition-keys`

Expected: PASS; policy tensor `(2,860)`, scalar reward `(2,)`, dual channels/extras present.

- [ ] **Step 5: Commit**

```bash
git add source/dwbc_isaaclab/tasks/widow_go1 tests/test_observation_contract.py tests/test_reward_contract.py
git commit -m "feat: port WidowGo1 observations rewards and terminations"
```

### Task 6: Compare legacy and Isaac Lab rollouts (Gates B–D)

**Files:**
- Create: `tools/{export_isaaclab_trace,compare_rollouts}.py`, `configs/alignment/rollout_tolerances.yaml`, `tests/test_rollout_comparison.py`
- Modify: `docs/alignment_protocol.md`

**Interfaces:** Candidate exporter uses Task 5; comparator produces JSON with field errors, first failing step and `gate_b_to_d_passed`.

- [ ] **Step 1: Write the failing comparator test**

```python
import numpy as np
from tools.compare_rollouts import compare_field
def test_abs_and_relative_tolerance_are_reported():
    result = compare_field("dof_pos", np.array([[1.]]), np.array([[1.001]]), 0.01, 0.01)
    assert result.passed and result.max_abs_error == 0.0009999999999998899
```

- [ ] **Step 2: Verify failure**

Run: `pytest tests/test_rollout_comparison.py -v`

Expected: FAIL because comparator does not exist.

- [ ] **Step 3: Implement equal-schema trace export and fail-closed comparison**

Candidate exporter accepts the same YAML/actions/seed flags and writes the exact Task 2 schema. Comparator first rejects action-order, shape, metadata-version and termination-reason differences; then computes maximum absolute, RMS and maximum relative error (`max(abs(reference), 1e-12)` denominator) per predeclared field. It never changes tolerances after a failure and writes config hashes, first bad step and all metric values to JSON.

- [ ] **Step 4: Verify one seed**

Run: `python tools/export_isaaclab_trace.py --scenario configs/alignment/smoke_32_envs.yaml --seed 1 --out artifacts/isaaclab/smoke/seed-1 && python tools/compare_rollouts.py --legacy artifacts/legacy/smoke/seed-1/trace.npz --candidate artifacts/isaaclab/smoke/seed-1/trace.npz --tolerances configs/alignment/rollout_tolerances.yaml --out artifacts/comparisons/smoke-seed-1.json`

Expected: JSON explicitly passes or identifies the field and first step that fails.

- [ ] **Step 5: Commit**

```bash
git add tools configs/alignment tests/test_rollout_comparison.py docs/alignment_protocol.md
git commit -m "test: add Isaac Lab rollout alignment harness"
```

### Task 7: Migrate custom RSL-RL structures, PPO and adapter (Gate E)

**Files:**
- Create: `source/dwbc_rsl_rl/modules/actor_critic.py`, `source/dwbc_rsl_rl/storage/rollout_storage.py`, `source/dwbc_rsl_rl/algorithms/ppo.py`
- Create: package `__init__.py` files and `source/dwbc_isaaclab/tasks/widow_go1/legacy_adapter.py`
- Create: `tests/{test_actor_critic_contract,test_storage_contract,test_ppo_transition_contract,test_ppo_update_contract}.py`

**Interfaces:** `ActorCritic(num_obs=860, num_actions=18, num_proprio=76, num_priv=24, history_len=10)` returns 18 actions and two values. Adapter returns `(obs, critic_obs, leg_reward, arm_reward, dones, infos)`.

- [ ] **Step 1: Write failing algorithm tests**

```python
import torch
from dwbc_rsl_rl.modules.actor_critic import ActorCritic
from dwbc_rsl_rl.algorithms.ppo import stack_dual_rewards
def test_dual_heads_and_rewards_are_preserved():
    model = ActorCritic(860, 18, 76, 24, 10)
    actions, values = model(torch.zeros(4,860), use_history=False)
    assert actions.shape == (4,18) and values.shape == (4,2)
    assert stack_dual_rewards(torch.tensor([1.]), torch.tensor([3.])).tolist() == [[1.,3.]]
```

- [ ] **Step 2: Verify failure**

Run: `pytest tests/test_actor_critic_contract.py tests/test_storage_contract.py tests/test_ppo_transition_contract.py tests/test_ppo_update_contract.py -v`

Expected: FAIL because package modules are absent.

- [ ] **Step 3: Port semantics with narrow APIs**

Port privileged/history encoders, 12-leg and 6-arm action heads, adaptive arm gains, two critic heads and normal-action distribution. Storage must hold rewards/values/returns/advantages with trailing width two and optional six-axis torque targets. PPO preserves reward stacking `[leg,arm]`, dual timeout bootstrap/GAE, value mixing schedule, clipped losses, entropy, privileged regularizer, DAgger history update, and torque supervision. Adapter rebuilds the 860 layout and maps Lab `terminated | time_out` plus extras without importing upstream RSL-RL.

- [ ] **Step 4: Verify deterministic numeric fixture**

Run: `pytest tests/test_actor_critic_contract.py tests/test_storage_contract.py tests/test_ppo_transition_contract.py tests/test_ppo_update_contract.py -v`

Expected: PASS; a saved fixed CPU rollout asserts returns, advantages and all loss keys at `rtol=1e-6, atol=1e-7`.

- [ ] **Step 5: Commit**

```bash
git add source/dwbc_rsl_rl source/dwbc_isaaclab/tasks/widow_go1/legacy_adapter.py tests
git commit -m "feat: migrate custom DWBC PPO algorithm"
```

### Task 8: Add runner and execute Gates E–F report

**Files:**
- Create: `source/dwbc_rsl_rl/runners/on_policy_runner.py`, `source/dwbc_isaaclab/tasks/widow_go1/agents/dwbc_ppo_cfg.py`
- Create: `scripts/{train,play,benchmark_gpu}.py`, `tools/compare_training.py`, `configs/alignment/training_seeds.yaml`
- Create: `tests/{test_checkpoint_schema,test_training_comparison}.py`, `docs/reports/gate-a-to-f.md`

**Interfaces:** CLI `scripts/train.py --num-envs N --max-iterations I --seed S --headless --run-dir PATH`; checkpoint includes `schema_version`, action/observation contract and model/optimizer state.

- [ ] **Step 1: Write failing runner/report tests**

```python
import numpy as np
from dwbc_rsl_rl.runners.on_policy_runner import make_checkpoint
from tools.compare_training import median_within_interval
def test_checkpoint_and_statistical_gate_schema():
    assert make_checkpoint(1, {}, {}, {})["contract"]["observation_dim"] == 860
    assert median_within_interval(np.array([9.,10.,11.]), 9.5, 10.5)
```

- [ ] **Step 2: Verify failure**

Run: `pytest tests/test_checkpoint_schema.py tests/test_training_comparison.py -v`

Expected: FAIL because runner and comparator are absent.

- [ ] **Step 3: Implement executable loop and fail-closed F comparison**

Port 40-step rollout, curriculum timing, DAgger cadence, separate leg/arm logging and all custom loss logs. Train/play accept safe CLI flags and only load checkpoints with matching contract version. Benchmark environment counts `32,64,128,256` sequentially, records peak allocated/reserved memory and FPS, and stops at first OOM. Training comparison requires seeds 1/2/3 and predeclared intervals; calculate median/mean/std and first threshold iteration, failing if any seed/metric is absent. Report every command, SHA, config hash, tolerance, artifact and known PhysX difference.

- [ ] **Step 4: Verify smoke, benchmark, then three seeds**

Run: `pytest -m "not isaaclab" -v && python scripts/train.py --num-envs 32 --max-iterations 1 --seed 1 --headless --run-dir artifacts/smoke && python scripts/benchmark_gpu.py --counts 32,64,128,256 --headless`

Expected: tests PASS; one update emits both rewards/custom losses and a schema-versioned checkpoint. Then run all three YAML seeds at the benchmark-approved count and execute `python tools/compare_training.py --config configs/alignment/training_seeds.yaml --out docs/reports/gate-a-to-f.md`.

- [ ] **Step 5: Commit report and handoff**

```bash
git add source scripts tools configs tests docs/reports README.md
git commit -m "docs: publish DWBC Isaac Lab alignment report"
```

## Plan Self-review

- Coverage: Tasks 1–2 freeze reproducibility and contracts; Task 3 is Gate A; Tasks 4–6 are Gates B–D; Task 7 is Gate E; Task 8 benchmarks hardware and performs Gate F.
- No placeholders: every task names exact files, an interface, a failing test, an invocation, an implementation target, verification and commit.
- Consistency: every task uses 18 actions, 20 joints, 76/24/10/860 observation layout, two reward/value channels, six torque axes and `xyzw` quaternions.
