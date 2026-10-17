"""Train attribute-conditioned SlicePrior: text -> attributes -> voxels.

Generates procedural worlds with random attributes, trains the diffusion
to generate slices conditioned on attribute vectors (not just style).
This is Phase 2 of tiny-Marble: prompts that *dream* instead of assemble.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from tiny_marble.voxel import G, procedural_voxel, EMPTY, WALL, LAVA, WATER
from tiny_marble.voxel_diffusion import SlicePrior
from tiny_marble.nn import MLP

# attribute vector: [style_onehot(3), lava, water, pillars] = 6 dims
ATTR_DIM = 6


def attrs_to_vec(attrs):
    """{'style': 'cavern', 'lava': True, ...} -> (6,) float32."""
    style_idx = {"rooms": 0, "cavern": 1, "arena": 2}[attrs["style"]]
    vec = np.zeros(ATTR_DIM, dtype=np.float32)
    vec[style_idx] = 3.0  # one-hot (scaled like training)
    vec[3] = 3.0 if attrs.get("lava") else 0.0
    vec[4] = 3.0 if attrs.get("water") else 0.0
    vec[5] = 3.0 if attrs.get("pillars") else 0.0
    return vec


def random_attrs(rng):
    styles = ["rooms", "cavern", "arena"]
    return {
        "style": styles[int(rng.integers(3))],
        "lava": bool(rng.random() < 0.3),
        "water": bool(rng.random() < 0.3),
        "dark": False,  # lighting, not geometry
        "pillars": bool(rng.random() < 0.3),
    }


def add_features(v, attrs, rng):
    """Add lava/water/pillars to procedural voxels (like prompt.py)."""
    if attrs["lava"]:
        for _ in range(int(rng.integers(2, 4))):
            cx, cy = int(rng.integers(3, G-3)), int(rng.integers(3, G-3))
            r = int(rng.integers(2, 4))
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    if dx*dx + dy*dy <= r*r:
                        x, y = cx+dx, cy+dy
                        if 1 <= x < G-1 and 1 <= y < G-1:
                            if v[x, y, 1] == EMPTY:
                                v[x, y, 1] = LAVA
    if attrs["water"]:
        for _ in range(int(rng.integers(2, 4))):
            cx, cy = int(rng.integers(3, G-3)), int(rng.integers(3, G-3))
            r = int(rng.integers(2, 4))
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    if dx*dx + dy*dy <= r*r:
                        x, y = cx+dx, cy+dy
                        if 1 <= x < G-1 and 1 <= y < G-1:
                            if v[x, y, 1] == EMPTY:
                                v[x, y, 1] = WATER
    if attrs["pillars"]:
        for _ in range(int(rng.integers(8, 16))):
            x, y = int(rng.integers(3, G-3)), int(rng.integers(3, G-3))
            h = int(rng.integers(8, 18))
            if (v[x, y, 1:4] == EMPTY).all():
                v[x, y, 1:1+h] = WALL
    return v


def main():
    import time
    t0 = time.time()

    rng = np.random.default_rng(0)
    n_worlds = 3000

    print(f"Generating {n_worlds} attributed worlds...")
    prevs, slices, attrs_list, heights = [], [], [], []
    for i in range(n_worlds):
        attrs = random_attrs(rng)
        style_idx = {"rooms": 0, "cavern": 1, "arena": 2}[attrs["style"]]
        v, _ = procedural_voxel(style=style_idx, seed=int(rng.integers(1e9)))
        v = add_features(v, attrs, rng)
        avec = attrs_to_vec(attrs)
        # binarize: walls/accents -> 1, lava/water -> 1 (solid for structure)
        # (keep it simple: any non-empty is "solid" for the slice model)
        b = (v != EMPTY).astype(np.float32) * 2 - 1  # [-1, 1]
        for z in range(1, G):
            prevs.append(b[:, :, z-1].reshape(-1))
            slices.append(b[:, :, z].reshape(-1))
            attrs_list.append(avec)
            heights.append(z)
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{n_worlds}")

    prevs = np.stack(prevs).astype(np.float32)
    slices = np.stack(slices).astype(np.float32)
    attrs_arr = np.stack(attrs_list).astype(np.float32)
    heights = np.array(heights, dtype=np.int64)
    print(f"  {len(slices)} slice pairs in {time.time()-t0:.0f}s")

    print("Training attribute-conditioned SlicePrior (256-wide, 10k steps)...")
    model = SlicePrior(attr_dim=ATTR_DIM, seed=0, hidden=256)
    model.fit(prevs, slices, attrs_arr, heights, steps=10000, batch=128,
              seed=1, log_every=1000)

    out = os.path.join(os.path.dirname(__file__), "..", "tiny_marble",
                       "checkpoints", "attr_slice_v2.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    params = model.net.parameters()
    params["attr_dim"] = np.array([ATTR_DIM])
    np.savez(out, **params)
    print(f"saved {out} in {time.time()-t0:.0f}s total")


if __name__ == "__main__":
    main()
