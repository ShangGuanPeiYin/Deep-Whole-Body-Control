# Environment Lock

## Required target

| Component | Required value |
| --- | --- |
| Isaac Lab | 3.0.0, fresh checkout |
| Python | 3.12.x |
| Conda environment | `dwbc-lab` |
| GPU | NVIDIA GeForce RTX 3070 Ti, 8192 MiB |
| NVIDIA driver | 580.173.02 |
| CUDA compute capability | 8.6 |

The exact Isaac Sim build and PyTorch version are recorded from the completed `dwbc-lab` installation before GPU task work begins. At repository initialization the only existing Lab checkout was `/home/xxs/research/IsaacLab` v2.3.2 with Python 3.11.15; it is intentionally preserved.

Collect the installed values with:

```bash
conda run -n dwbc-lab python -c "import platform, torch; print(platform.python_version()); print(torch.__version__)"
git -C /home/xxs/research/IsaacLab-3.0.0 describe --tags --always --dirty
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
```
