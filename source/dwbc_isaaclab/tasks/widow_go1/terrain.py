"""The legacy continuous Perlin terrain as an Isaac Lab sub-terrain."""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh
from isaaclab.utils import configclass


def _perlin_noise_2d(shape: tuple[int, int], resolution: tuple[int, int], rng: np.random.RandomState) -> np.ndarray:
    def fade(value: np.ndarray) -> np.ndarray:
        return 6 * value**5 - 15 * value**4 + 10 * value**3

    delta = (resolution[0] / shape[0], resolution[1] / shape[1])
    repeats = (shape[0] // resolution[0], shape[1] // resolution[1])
    grid = np.mgrid[0 : resolution[0] : delta[0], 0 : resolution[1] : delta[1]].transpose(1, 2, 0) % 1
    angles = 2 * np.pi * rng.rand(resolution[0] + 1, resolution[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[:-1, :-1].repeat(repeats[0], 0).repeat(repeats[1], 1)
    g10 = gradients[1:, :-1].repeat(repeats[0], 0).repeat(repeats[1], 1)
    g01 = gradients[:-1, 1:].repeat(repeats[0], 0).repeat(repeats[1], 1)
    g11 = gradients[1:, 1:].repeat(repeats[0], 0).repeat(repeats[1], 1)
    n00 = np.sum(grid * g00, axis=2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, axis=2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, axis=2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, axis=2)
    weight = fade(grid)
    n0 = n00 * (1 - weight[:, :, 0]) + weight[:, :, 0] * n10
    n1 = n01 * (1 - weight[:, :, 0]) + weight[:, :, 0] * n11
    return np.sqrt(2) * ((1 - weight[:, :, 1]) * n0 + weight[:, :, 1] * n1) * 0.5 + 0.5


def legacy_perlin_height_field(cfg: "LegacyPerlinTerrainCfg") -> np.ndarray:
    """Generate exactly the two-octave field and cliff used by the legacy task."""
    rng = np.random.RandomState(cfg.seed if cfg.seed is not None else 1)
    x_size, y_size = int(cfg.size[0]), int(cfg.size[1])
    frequency = 10
    x_scale, y_scale = frequency * x_size, frequency * y_size
    amplitude = 1.0
    noise = np.zeros((cfg.samples_x, cfg.samples_y))
    for _ in range(2):
        noise += amplitude * _perlin_noise_2d((cfg.samples_x, cfg.samples_y), (x_scale, y_scale), rng) * cfg.z_scale
        amplitude *= 0.25
        x_scale, y_scale = 2 * x_scale, 2 * y_scale
    noise[cfg.samples_x // 2 - 100 :, :] += 100000.0
    # Preserve the legacy int16 cast (including its saturated cliff sentinel) without noisy NumPy warnings.
    with np.errstate(invalid="ignore", over="ignore"):
        return (noise / cfg.vertical_scale).astype(np.int16)


def legacy_perlin_terrain(_difficulty: float, cfg: "LegacyPerlinTerrainCfg"):
    height_field = legacy_perlin_height_field(cfg)
    if cfg.mesh_stride > 1:
        height_field = height_field[:: cfg.mesh_stride, :: cfg.mesh_stride]
    vertices, triangles = convert_height_field_to_mesh(
        height_field, cfg.horizontal_scale * cfg.mesh_stride, cfg.vertical_scale, cfg.slope_threshold
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, 0.0])
    return [mesh], origin


@configclass
class LegacyPerlinTerrainCfg(SubTerrainBaseCfg):
    function = legacy_perlin_terrain
    samples_x: int = 600
    samples_y: int = 2000
    horizontal_scale: float = 0.025
    vertical_scale: float = 1.0e-5
    slope_threshold: float | None = 1.0e8
    z_scale: float = 0.15
    seed: int | None = 1
    mesh_stride: int = 4
