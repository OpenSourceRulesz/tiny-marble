"""tiny-Marble demo: prompt-to-world.

Type words -> walk through the 3D world they describe.
"a dark cavern with lava" -> a dark cavern, with lava, that you
can walk through in first person.

Usage: .venv/bin/python examples/prompt_demo.py ["your prompt here"]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image

from tiny_jepa.worldgen import dream_from_prompt
from tiny_jepa.worldgen.voxel import EMPTY, EYE_Z, G, LAVA

HERE = os.path.dirname(os.path.abspath(__file__))


def _count_lava_px(frame):
    """Lava-ish pixels in a rendered frame (float32 0..1 or uint8)."""
    a = np.asarray(frame)
    if a.dtype != np.uint8:
        a = (a * 255).astype("uint8")
    r, g, b = a[..., 0].astype(int), a[..., 1].astype(int), a[..., 2].astype(int)
    return int(((r > 180) & (g > 40) & (g < 130) & (b < 60)).sum())


def _aim_at_lava(world):
    """If the world has lava, start at a camera spot with a dramatic lava
    view so the demo actually shows the lava. Falls back to the variance
    scan otherwise."""
    lava = np.argwhere(world.voxels == LAVA)
    if len(lava) == 0:
        return False
    # densest pool center = the aim target
    d2 = ((lava[:, None, :2] - lava[None, :, :2]) ** 2).sum(-1)
    tgt = lava[int(np.argmax((d2 < 9).sum(1)))][:2].astype(float)
    best = None  # (lava_px, x, y, z, yaw, pitch)
    for z in (6.0, 10.0, 14.0, EYE_Z):
        for r in (4.0, 7.0):
            for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                x = int(np.clip(tgt[0] + r * np.cos(ang), 2, G - 3))
                y = int(np.clip(tgt[1] + r * np.sin(ang), 2, G - 3))
                if world.voxels[x, y, int(z)] != EMPTY:
                    continue  # inside rock; skip
                world.pos[:] = [x + 0.5, y + 0.5, z]
                for pitch in (-0.45, -0.15):
                    world.pitch = pitch
                    for yaw in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                        world.yaw = float(yaw)
                        n = _count_lava_px(world.render())
                        if best is None or n > best[0]:
                            best = (n, x, y, z, float(yaw), pitch)
    if best and best[0] > 200:
        _, x, y, z, yaw, pitch = best
        world.pos[:] = [x + 0.5, y + 0.5, z]
        world.yaw, world.pitch = yaw, pitch
        return True
    return False


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "a dark cavern with lava"
    world = dream_from_prompt(prompt, seed=7)
    print(f"prompt: '{prompt}' -> {world.attrs}")

    # find a good starting view: aim at lava if the prompt dreamed any,
    # otherwise pick the highest-variance view from spawn
    if not _aim_at_lava(world):
        world.pitch = -0.25
        best, bv = -1, 0
        for y in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            world.yaw = float(y)
            f = world.render()
            if f.var() > best:
                best, bv = f.var(), float(y)
        world.yaw = bv

    frames = []
    for t in range(96):
        before = world.pos.copy()
        # explore: weave + turn away from walls
        world.step(forward=1.0, dyaw=0.12 * np.sin(t * 0.3) + 0.05)
        if np.allclose(before[:2], world.pos[:2]):
            world.turn(0.9)
        if t % 2 == 0:
            frames.append(world.render())

    print(f"{len(frames)} frames")
    # taichi renders at 256x256 natively; numpy at 64x64 (upscale)
    imgs = []
    for f in frames:
        im = Image.fromarray((f * 255).astype("uint8"))
        if im.size != (256, 256):
            im = im.resize((256, 256), Image.NEAREST)
        imgs.append(im)
    out = os.path.join(HERE, "prompt_world.gif")
    imgs[0].save(out, save_all=True, append_images=imgs[1:],
                  duration=120, loop=0)
    print("saved", out)


if __name__ == "__main__":
    main()
