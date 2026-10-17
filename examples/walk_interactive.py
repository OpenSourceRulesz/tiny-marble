"""Interactive walk: WASD + mouse through your prompted world.

Run: .venv/bin/python examples/walk_interactive.py "a dark cavern with lava"

Controls:
  W/A/S/D - move, Mouse - look, Q/E - down/up, ESC - quit
Requires: pip install tiny-marble[fast] (Taichi)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from tiny_marble import dream_from_prompt

try:
    import taichi as ti
    from tiny_marble.taichi_render import _ensure_init, _get_kernel
    from tiny_marble.taichi_render import (_voxels_ti, _glow_ti, _img_ti,
                                           _lava_pos_ti, _n_lava)
    from tiny_marble.voxel import _lava_glow, G, EMPTY, WISP
    TI_OK = True
except ImportError:
    TI_OK = False


def main():
    if not TI_OK:
        print("Need Taichi: pip install tiny-marble[fast]")
        return

    prompt = sys.argv[1] if len(sys.argv) > 1 else "a dark cavern with lava"
    world = dream_from_prompt(prompt, seed=7)
    print(f"World: '{prompt}' -> {world.attrs}")
    print("WASD move, mouse look, Q/E down/up, ESC quit")

    _ensure_init()

    # upload voxels
    vox = world.voxels.copy()
    for w_ in world.wisps:
        xi, yi, zi = int(w_[0]), int(w_[1]), int(w_[2])
        if 0 <= xi < G and 0 <= yi < G and 0 <= zi < G:
            if vox[xi, yi, zi] == EMPTY:
                vox[xi, yi, zi] = WISP
    _voxels_ti.from_numpy(np.ascontiguousarray(vox.astype(np.int32)))
    _glow_ti.from_numpy(_lava_glow(world.voxels))

    # lava positions for shadows
    from tiny_marble.taichi_render import _compute_lava_positions
    lava_pos = _compute_lava_positions(vox)
    n = min(len(lava_pos), 64)
    if n > 0:
        _lava_pos_ti.from_numpy(np.ascontiguousarray(
            np.pad(lava_pos[:n], ((0, 64 - n), (0, 0))).astype(np.float32)))
    _n_lava[None] = n

    kernel = _get_kernel()

    window = ti.ui.Window("tiny-marble", (512, 512))
    canvas = window.get_canvas()

    # camera state
    pos = world.pos.copy()
    yaw = float(world.yaw)
    pitch = -0.15

    # for wisp animation
    t = 0

    while window.running:
        # mouse look
        if window.is_pressed(ti.ui.LMB):
            mx, my = window.get_cursor_pos()
            # (simple: use arrow keys for look if no mouse drag)
            pass

        # keyboard look (arrows)
        if window.is_pressed(ti.ui.LEFT):
            yaw += 0.05
        if window.is_pressed(ti.ui.RIGHT):
            yaw -= 0.05
        if window.is_pressed(ti.ui.UP):
            pitch = min(0.5, pitch + 0.03)
        if window.is_pressed(ti.ui.DOWN):
            pitch = max(-0.8, pitch - 0.03)

        # WASD move (relative to yaw)
        fx, fy = np.cos(yaw), np.sin(yaw)
        speed = 0.5
        dx, dy = 0.0, 0.0
        if window.is_pressed('w'):
            dx += fx * speed; dy += fy * speed
        if window.is_pressed('s'):
            dx -= fx * speed; dy -= fy * speed
        if window.is_pressed('a'):
            dx += fy * speed; dy -= fx * speed
        if window.is_pressed('d'):
            dx -= fy * speed; dy += fx * speed

        # collision (simple: check target voxel)
        nx, ny = pos[0] + dx, pos[1] + dy
        if 1 < nx < G - 1 and 1 < ny < G - 1:
            # check at eye level (z=20) and below
            if vox[int(nx), int(ny), 18] == EMPTY:
                pos[0], pos[1] = nx, ny

        # Q/E for pitch (up/down look already on arrows; use for fly)
        # (keep simple: no vertical move)

        if window.is_pressed(ti.ui.ESCAPE):
            break

        # render
        kernel(_voxels_ti, _glow_ti, _lava_pos_ti, _n_lava, _img_ti,
               float(pos[0]), float(pos[1]), float(pos[2]),
               float(yaw), float(pitch), float(world.ambient), 1)
        img = _img_ti.to_numpy()

        # display (Taichi expects HxWx3)
        canvas.set_image(img)
        window.show()

        t += 1

    print("done")


if __name__ == "__main__":
    main()
