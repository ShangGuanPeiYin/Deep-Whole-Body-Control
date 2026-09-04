# Environment Lock

## Required target

| Component | Required value |
| --- | --- |
| Isaac Lab | 2.3.2 |
| Isaac Sim | 5.1.0.0 |
| Python | 3.11.15 |
| PyTorch | 2.7.0+cu128 |
| Conda environment | `isaac-lab` |
| GPU | NVIDIA GeForce RTX 3070 Ti, 8192 MiB |
| NVIDIA driver | 580.173.02 |
| CUDA compute capability | 8.6 |

These values were measured from the existing stable checkout `/home/xxs/research/IsaacLab` at `v2.3.2` before GPU task work began.

Collect the installed values with:

```bash
conda run -n isaac-lab python -c "import platform, torch, importlib.metadata as m; print(platform.python_version()); print(torch.__version__); print(m.version('isaacsim'))"
git -C /home/xxs/research/IsaacLab describe --tags --always --dirty
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
```
