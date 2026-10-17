"""Taichi Gaussian Splat rasterizer: forward splatting.

For each 3D Gaussian: project to screen, compute 2D footprint,
rasterize as soft ellipse, alpha-blend into framebuffer.

This is the core of 3D Gaussian Splatting (SIGGRAPH 2023),
miniaturized for CPU. Marble does this with millions on GPU;
we do thousands on CPU.
"""

import numpy as np

try:
    import taichi as ti
    TI_OK = True
except ImportError:
    TI_OK = False

_W, _H = 256, 256
_kernel = None
_img_ti = None
_gauss_ti = None


def _ensure_init():
    global _kernel, _img_ti, _gauss_ti
    if not TI_OK:
        raise ImportError("Need Taichi: pip install tiny-marble[fast]")
    if _kernel is not None:
        return
    ti.init(arch=ti.cpu, random_seed=0)

    _img_ti = ti.field(dtype=ti.f32, shape=(_H, _W, 3))
    # max 20000 Gaussians: [x, y, z, scale, r, g, b, opacity]
    _gauss_ti = ti.field(dtype=ti.f32, shape=(20000, 8))

    @ti.kernel
    def _splat_kernel(gauss: ti.template(), n_gauss: ti.i32,
                      img: ti.template(),
                      cam_x: ti.f32, cam_y: ti.f32, cam_z: ti.f32,
                      yaw: ti.f32, pitch: ti.f32):
        # clear
        for py, px in ti.ndrange(_H, _W):
            # sky gradient
            t = float(py) / float(_H)
            img[py, px, 0] = 0.05 + 0.10 * t
            img[py, px, 1] = 0.08 + 0.12 * t
            img[py, px, 2] = 0.15 + 0.20 * t

        # forward splat each Gaussian
        for i in range(n_gauss):
            gx = gauss[i, 0]
            gy = gauss[i, 1]
            gz = gauss[i, 2]
            gs = gauss[i, 3]
            gr = gauss[i, 4]
            gg = gauss[i, 5]
            gb = gauss[i, 6]
            go = gauss[i, 7]

            # transform to camera space
            # (translate, then rotate by -yaw, -pitch)
            dx = gx - cam_x
            dy = gy - cam_y
            dz = gz - cam_z

            # yaw rotation (around Z)
            cy, sy = ti.cos(yaw), ti.sin(yaw)
            # camera looks along +X in world? (match voxel renderer)
            # Actually: forward = (cos(yaw), sin(yaw), 0) in world
            # Camera space: x=forward, y=right, z=up
            fx = dx * cy + dy * sy  # forward distance
            ry = -dx * sy + dy * cy  # right
            uz = dz  # up (before pitch)

            # pitch rotation (around Y/right axis)
            cp, sp = ti.cos(pitch), ti.sin(pitch)
            fxp = fx * cp + uz * sp
            uzp = -fx * sp + uz * cp

            if fxp < 0.1:  # behind camera
                continue

            # project to screen
            # fov: 70deg horiz, 55deg vert (match voxel renderer)
            sx = 0.5 + (ry / fxp) / ti.tan(0.6109) * 0.5
            sy_screen = 0.5 - (uzp / fxp) / ti.tan(0.4800) * 0.5

            px_c = int(sx * _W)
            py_c = int(sy_screen * _H)

            # screen-space radius (perspective-scaled)
            # world scale gs at distance fxp -> pixels
            rad = int(gs / fxp * _W * 0.8) + 1
            rad = ti.min(rad, 32)  # clamp for perf

            # rasterize ellipse (soft falloff)
            for oy in range(-rad, rad + 1):
                for ox in range(-rad, rad + 1):
                    px = px_c + ox
                    py = py_c + oy
                    if px < 0 or px >= _W or py < 0 or py >= _H:
                        continue
                    # normalized distance from center
                    d2 = (float(ox) / float(rad)) ** 2 + \
                         (float(oy) / float(rad)) ** 2
                    if d2 > 1.0:
                        continue
                    # Gaussian falloff
                    alpha = go * ti.exp(-3.0 * d2)
                    if alpha < 0.01:
                        continue
                    # alpha blend (front-to-back approx via atomics)
                    # (not perfectly sorted, but looks good)
                    for c in ti.static(range(3)):
                        col = gr if c == 0 else (gg if c == 1 else gb)
                        old = img[py, px, c]
                        # atomic blend: new = old*(1-a) + col*a
                        # (use ti.atomic_add for thread safety)
                        ti.atomic_add(img[py, px, c],
                                      (col - old) * alpha)

    _kernel = _splat_kernel


def render_splats(gaussians, cam_pos, yaw, pitch=0.0):
    """Render Gaussians to 256x256 RGB. Returns numpy array."""
    _ensure_init()
    n = min(len(gaussians), 20000)
    padded = np.zeros((20000, 8), dtype=np.float32)
    padded[:n] = gaussians[:n].astype(np.float32)
    _gauss_ti.from_numpy(np.ascontiguousarray(padded))
    _kernel(_gauss_ti, n, _img_ti,
            float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]),
            float(yaw), float(pitch))
    return _img_ti.to_numpy()
