"""Showcase: tiny-Marble at its best. Renders hero shots of prompted worlds."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image

from tiny_jepa.worldgen import dream_from_prompt, LAVA, WATER
from tiny_jepa.worldgen.voxel import G

HERE = os.path.dirname(os.path.abspath(__file__))


def hero_shot(prompt, seed, pitch=-0.3):
    w = dream_from_prompt(prompt, seed=seed)
    print(f"'{prompt}' -> {w.attrs} (taichi: {w._use_taichi})")
    w.pitch = pitch
    # best yaw
    best, bv = -1, 0
    for y in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        w.yaw = float(y)
        f = w.render()
        if f.var() > best:
            best, bv = f.var(), float(y)
    w.yaw = bv
    return w.render()


def main():
    shots = [
        ("a dark cavern with lava", 7, -0.35),
        ("an open arena with water", 3, -0.4),
        ("a dungeon with pillars", 11, -0.2),
    ]
    imgs = []
    for prompt, seed, pitch in shots:
        f = hero_shot(prompt, seed, pitch)
        imgs.append(Image.fromarray(
            (np.clip(f, 0, 1) * 255).astype("uint8")))
        print(f"  rendered {f.shape}")

    # side-by-side triptych
    W, H = imgs[0].size
    tri = Image.new("RGB", (W * 3, H))
    for i, im in enumerate(imgs):
        tri.paste(im, (i * W, 0))
    out = os.path.join(HERE, "showcase.png")
    tri.save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
