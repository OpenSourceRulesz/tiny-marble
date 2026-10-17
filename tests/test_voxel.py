"""Tests for tiny-4D (worldgen.voxel + voxel_diffusion)."""

import numpy as np

from tiny_marble.voxel import (
    VoxelWorld, procedural_voxel, raycast, G, RES, EMPTY, WALL)
from tiny_marble.voxel_diffusion import VoxelPrior


def test_procedural_voxel_structure():
    for s in range(3):
        v, spawn = procedural_voxel(s, seed=s)
        assert v.shape == (G, G, G)
        assert (v[:, :, 0] == WALL).all(), "floor must be solid"
        assert (v[0, :, 1:] != EMPTY).all() \
            and (v[:, 0, 1:] != EMPTY).all(), "border must be solid"
        cx, cy = spawn
        # spawn clearing is walkable
        assert (v[cx - 1:cx + 2, cy - 1:cy + 2, 1:4] == EMPTY).all()
        assert set(np.unique(v)) <= {0, 1, 2}


def test_raycast_known_scene():
    v = np.zeros((G, G, G), dtype=np.int64)
    v[:, :, 0] = WALL
    v[12, 8, 1:] = WALL  # lone wall ahead
    img = raycast(v, np.array([8.5, 8.5, 10.0]), yaw=0.0)
    assert img.shape == (RES, RES, 3) and img.dtype == np.float32
    assert (img >= 0).all() and (img <= 1).all()
    center = img[RES // 2, RES // 2]
    # wall gray, not sky blue
    assert center[0] > 0.2 and abs(center[0] - center[2]) < 0.25
    # empty world from above-ish: mostly sky/floor, no crash
    img2 = raycast(np.zeros((G, G, G), dtype=np.int64),
                   np.array([8.5, 8.5, 10.0]), yaw=1.0)
    assert img2.shape == (RES, RES, 3)


def test_world_collision():
    v = np.zeros((G, G, G), dtype=np.int64)
    v[:, :, 0] = WALL
    v[10, 8, 1:] = WALL  # wall directly ahead (+x)
    w = VoxelWorld(v, spawn=(8, 8), seed=0)
    w.pos = np.array([8.5, 8.5, 10.0])
    w.yaw = 0.0
    for _ in range(20):
        w.move(1.0)
    assert w.pos[0] < 10.0, "walked through a wall!"


def test_time_pause_and_reverse():
    v, spawn = procedural_voxel(2, seed=0)
    w = VoxelWorld(v, spawn=spawn, seed=0)
    p0 = w.wisps.copy()
    w.pause()
    for _ in range(10):
        w.step(forward=1.0)
    assert np.allclose(w.wisps, p0), "paused time still flowed"
    w.resume()
    for _ in range(30):
        w.step()
    assert not np.allclose(w.wisps, p0), "wisps never moved"
    w.reverse(30)
    assert np.allclose(w.wisps, p0, atol=1e-6), \
        "time reversal did not restore the river"


def test_voxel_prior_trains_and_samples():
    rng = np.random.default_rng(0)
    maps, styles = [], []
    for i in range(48):
        v, _ = procedural_voxel(i % 3, seed=i)
        maps.append(np.where(v > 0, 1.0, -1.0).reshape(-1)
                    .astype(np.float32))
        styles.append(i % 3)
    vp = VoxelPrior(seed=0).fit(np.stack(maps), np.array(styles),
                                steps=30, batch=16, seed=0, log_every=0)
    s = vp.sample(style=2, n=2, seed=0)
    assert s.shape == (2, G, G, G)
    assert set(np.unique(s)) <= {0, 1}
    assert (s[:, :, :, 0] == 1).all(), "floor guarantee broken"


def test_voxel_prior_save_load(tmp_path):
    vp = VoxelPrior(seed=0)
    p = str(tmp_path / "vox.npz")
    vp.save(p)
    vp2 = VoxelPrior.load(p)
    a = vp.sample(style=0, n=1, seed=3)
    b = vp2.sample(style=0, n=1, seed=3)
    assert (a == b).all()


def test_slice_prior_encode_decode():
    from tiny_marble.voxel_diffusion import SlicePrior, HeightfieldPrior
    v, _ = procedural_voxel(0, seed=0)
    h = HeightfieldPrior.encode(v)
    assert h.shape == (2 * G * G,) and (h >= -1).all() and (h <= 1).all()
    v2 = HeightfieldPrior.decode(h)
    assert v2.shape == (G, G, G)
    # footprint round-trips (heights are lossy, footprint is not)
    fp1 = (v[:, :, 1:].max(axis=2) > 0)
    fp2 = (v2[:, :, 1:].max(axis=2) > 0)
    assert (fp1 == fp2).mean() > 0.95


def test_slice_prior_trains_and_samples():
    from tiny_marble.voxel_diffusion import SlicePrior
    rng = np.random.default_rng(0)
    prevs, slices, styles, heights = [], [], [], []
    for i in range(24):
        v, _ = procedural_voxel(0, seed=i)
        for z in (1, 5, 10):
            prevs.append(np.where(v[:, :, z-1].reshape(-1) > 0, 1.0, -1.0)
                         .astype(np.float32))
            slices.append(np.where(v[:, :, z].reshape(-1) > 0, 1.0, -1.0)
                          .astype(np.float32))
            styles.append(0); heights.append(z)
    sp = SlicePrior(seed=0).fit(np.stack(prevs), np.stack(slices),
                                np.array(styles), np.array(heights),
                                steps=20, batch=16, seed=0, log_every=0)
    v = sp.sample(style=0, seed=0)
    assert v.shape == (G, G, G)
    assert (v[:, :, 0] == WALL).all(), "floor guarantee broken"
