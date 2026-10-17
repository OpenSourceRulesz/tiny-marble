"""tiny-4D demo: walk through a dreamed voxel world in first person.

A 16^3 voxel world (procedural here; diffusion-dreamed with
voxel_diffusion.py), rendered by a pure-numpy DDA raycaster at ~100 FPS.
Wisps drift on their own -- the 4th dimension. Midway through, time
REVERSES: the camera holds still while the wisps fly backwards along
their trajectories. ("No man ever steps in the same river twice" --
here you can.)

Usage:  .venv/bin/python examples/walk4d_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image

from tiny_jepa.worldgen.voxel import dream_voxel_world

HERE = os.path.dirname(os.path.abspath(__file__))


def best_yaw(world):
    """Sweep yaw, keep the view with the most depth variance."""
    scores = []
    for y in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        world.yaw = float(y)
        f = world.render()
        scores.append(float(f.var()))
    world.yaw = float(np.linspace(0, 2 * np.pi, 12,
                                  endpoint=False)[int(np.argmax(scores))])


def main():
    # procedural 4D world (the 3D diffusion is an open challenge --
    # see README; procedural worlds are beautiful and walkable today)
    world = dream_voxel_world(prior=None, style=2, seed=4)
    best_yaw(world)
    print("procedural arena world, ready to walk")

    frames = []
    # phase 1: explore -- walk forward, turn away when blocked, wisps drift
    for t in range(64):
        before = world.pos.copy()
        world.step(forward=1.0, dyaw=0.04 * np.sin(t * 0.2))
        if np.allclose(before[:2], world.pos[:2]):
            world.turn(0.5)  # bonked: turn away
        if t % 2 == 0:
            frames.append(world.render())
    # phase 2: TIME REVERSES -- camera still, wisps fly backwards
    for t in range(20):
        world.reverse(2)
        if t % 2 == 0:
            frames.append(world.render())
    # phase 3: time resumes, keep exploring
    for t in range(32):
        before = world.pos.copy()
        world.step(forward=1.0, dyaw=0.05 * np.cos(t * 0.3))
        if np.allclose(before[:2], world.pos[:2]):
            world.turn(-0.5)
        if t % 2 == 0:
            frames.append(world.render())

    print(f"{len(frames)} frames, t={world.t}")
    imgs = [Image.fromarray((f * 255).astype("uint8")).resize((256, 256),
            Image.NEAREST) for f in frames]
    out = os.path.join(HERE, "walk4d.gif")
    imgs[0].save(out, save_all=True, append_images=imgs[1:],
                  duration=120, loop=0)
    print("saved", out)


if __name__ == "__main__":
    main()
