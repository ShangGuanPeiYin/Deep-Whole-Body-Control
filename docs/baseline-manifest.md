# Frozen Legacy Baseline Manifest

Captured on 2026-09-04 from `/home/xxs/research/Deep-Whole-Body-Control`.

| Item | Value |
| --- | --- |
| Git HEAD | `8159e4ed8695b2d3f62a40d2ab8d88205ac5021a` |
| Conda environment | `dwbc` |
| Python | 3.8.20 |
| PyTorch | 1.10.0+cu113 |
| Task | `widowGo1` |
| Baseline seeds | 1, 2, 3 |
| Development GPU | RTX 3070 Ti 8 GB |

The complete package lock is generated without changing the legacy environment:

```bash
conda list --explicit -n dwbc > artifacts/legacy/conda-dwbc-explicit.txt
```

The generated lock is stored with the legacy trace artifact because it contains machine-specific package URLs; its SHA-256 is added to trace metadata by the exporter.

The baseline worktree was dirty, so comparisons identify behavior by both HEAD and the content hashes below:

| File | SHA-256 |
| --- | --- |
| `legged_gym/legged_gym/envs/widowGo1/widowGo1.py` | `78c8d72342fce2752b865a4560cba230330b474a496bfffc75909b8951e7b472` |
| `legged_gym/legged_gym/envs/widowGo1/widowGo1_config.py` | `0d736f6ee38c82d4f47d2c484e788b37bad8e4fcd7da8e57d4737a4a6b7f9735` |
| `legged_gym/legged_gym/utils/math.py` | `334f00da93e90568879e46e00c4ee84d44b7ace79c6db04969762fb24a965092` |

Reproduce the validated one-iteration legacy smoke run with:

```bash
env PYTHONUNBUFFERED=1 TORCH_EXTENSIONS_DIR=/home/xxs/research/Deep-Whole-Body-Control/.torch_extensions MPLCONFIGDIR=/home/xxs/research/Deep-Whole-Body-Control/.matplotlib WANDB_MODE=disabled conda run --no-capture-output -n dwbc python legged_gym/legged_gym/scripts/train.py --task widowGo1 --run_name smoke_default --exptid 9 --max_iterations 1 --headless
```

Before Gate A, append SHA-256 values for every URDF/mesh consumed by WidowGo1. The old task and custom RSL-RL sources remain read-only.
