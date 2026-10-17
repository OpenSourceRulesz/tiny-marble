"""Gaussian Splatting for tiny-marble: Marble's technique, miniaturized.

Converts voxels to 3D Gaussians (position, scale, color, opacity),
then rasterizes by forward splatting in Taichi.

This is what World Labs Marble does (millions of splats on GPU).
We do thousands on CPU — same math, smaller scale.
"""

import numpy as np

from .voxel import G, EMPTY, WALL, ACCENT, WISP, LAVA, WATER

# Gaussian: [x, y, z, scale, r, g, b, opacity]
N_PARAMS = 8


def voxels_to_gaussians(voxels, seed=0, jitter=0.15, scale=0.65):
    """Convert surface voxels to Gaussians.

    Only voxels with at least one empty neighbor become splats
    (interior voxels are invisible anyway).
    """
    rng = np.random.default_rng(seed)
    gaussians = []

    # voxel colors (match taichi_render.py)
    colors = {
        WALL: (0.62, 0.66, 0.72),
        ACCENT: (0.95, 0.55, 0.25),
        WISP: (0.65, 0.95, 1.0),
        LAVA: (1.0, 0.30, 0.08),
        WATER: (0.15, 0.45, 0.85),
    }

    for x in range(G):
        for y in range(G):
            for z in range(G):
                v = voxels[x, y, z]
                if v == EMPTY:
                    continue
                # check if surface (has empty neighbor)
                is_surface = False
                for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0),
                                   (0,-1,0), (0,0,1), (0,0,-1)]:
                    nx, ny, nz = x+dx, y+dy, z+dz
                    if 0 <= nx < G and 0 <= ny < G and 0 <= nz < G:
                        if voxels[nx, ny, nz] == EMPTY:
                            is_surface = True
                            break
                    else:
                        is_surface = True
                        break
                if not is_surface:
                    continue

                # position (voxel center + jitter)
                px = x + 0.5 + rng.uniform(-jitter, jitter)
                py = y + 0.5 + rng.uniform(-jitter, jitter)
                pz = z + 0.5 + rng.uniform(-jitter, jitter)

                # color with variation
                base = colors.get(int(v), (0.5, 0.5, 0.5))
                var = rng.uniform(0.85, 1.15)
                r, g, b = base[0]*var, base[1]*var, base[2]*var

                # scale (slightly overlapping for soft look)
                s = scale * rng.uniform(0.9, 1.1)

                # opacity
                op = 0.95 if v != WISP else 0.7

                gaussians.append([px, py, pz, s, r, g, b, op])

    return np.array(gaussians, dtype=np.float32)


def count_surface_voxels(voxels):
    """How many Gaussians will we get? (for planning)"""
    count = 0
    for x in range(G):
        for y in range(G):
            for z in range(G):
                if voxels[x, y, z] == EMPTY:
                    continue
                for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0),
                                   (0,-1,0), (0,0,1), (0,0,-1)]:
                    nx, ny, nz = x+dx, y+dy, z+dz
                    if not (0 <= nx < G and 0 <= ny < G and 0 <= nz < G):
                        count += 1
                        break
                    if voxels[nx, ny, nz] == EMPTY:
                        count += 1
                        break
    return count
