"""Taichi-accelerated voxel raycaster for tiny-4D.

The numpy raycaster (voxel.py) is the reference implementation and fallback.
This module provides the same API at 30-40x the speed, enabling 256x256
real-time rendering with soft shadows.

Soft shadows: from each surface hit, march a shadow ray toward the nearest
lava pool. If blocked, the point is in shadow. This is what makes lava feel
like a *light source*, not just a bright texture.

Usage:
    from tiny_marble.taichi_render import render
    img = render(voxels, pos, yaw, pitch=0.0, ambient=1.0)  # (H, W, 3)

Requires: pip install taichi (optional; falls back to numpy if missing).
"""

import numpy as np

try:
    import taichi as ti
    _TI_AVAILABLE = True
except ImportError:
    _TI_AVAILABLE = False
    ti = None

# module state (initialized lazily)
_initialized = False
_G = 32
_W = 256
_H = 256

# voxel values (must match voxel.py)
EMPTY, WALL, ACCENT, WISP, LAVA, WATER = 0, 1, 2, 3, 4, 5


def _ensure_init():
    global _initialized, _voxels_ti, _glow_ti, _img_ti, _lava_pos_ti, _n_lava
    if _initialized or not _TI_AVAILABLE:
        return
    ti.init(arch=ti.cpu, log_level=ti.ERROR)
    _voxels_ti = ti.field(ti.i32, shape=(_G, _G, _G))
    _glow_ti = ti.field(ti.f32, shape=(_G, _G, _G))
    _img_ti = ti.field(ti.f32, shape=(_H, _W, 3))
    _lava_pos_ti = ti.field(ti.f32, shape=(64, 3))  # up to 64 lava pools
    _n_lava = ti.field(ti.i32, shape=())
    _initialized = True


def is_available():
    return _TI_AVAILABLE


if _TI_AVAILABLE:
    def _define_kernels():
        @ti.func
        def hash3(x, y, z):
            h = (x * 73856093) ^ (y * 19349663) ^ (z * 83492791)
            return float(h % 100) / 100.0

        @ti.func
        def dda(voxels, ox, oy, oz, dx, dy, dz, max_iter):
            """DDA raymarch. Returns (hit_v, hit_t, nx, ny, nz, hx, hy, hz)."""
            rx, ry, rz = int(ox), int(oy), int(oz)
            sx = 1 if dx > 0 else -1
            sy = 1 if dy > 0 else -1
            sz = 1 if dz > 0 else -1
            tmx = ((float(rx + (1 if dx > 0 else 0)) - ox) / dx) if dx != 0 else 1e9
            tmy = ((float(ry + (1 if dy > 0 else 0)) - oy) / dy) if dy != 0 else 1e9
            tmz = ((float(rz + (1 if dz > 0 else 0)) - oz) / dz) if dz != 0 else 1e9
            tdx = ti.abs(1.0 / dx) if dx != 0 else 1e9
            tdy = ti.abs(1.0 / dy) if dy != 0 else 1e9
            tdz = ti.abs(1.0 / dz) if dz != 0 else 1e9
            hit_v = 0
            hit_t = 0.0
            nx, ny, nz = 0, 0, 0
            hx, hy, hz = 0, 0, 0
            for _ in range(max_iter):
                if tmx < tmy and tmx < tmz:
                    rx += sx; hit_t = tmx; tmx += tdx
                    nx, ny, nz = -sx, 0, 0
                elif tmy < tmz:
                    ry += sy; hit_t = tmy; tmy += tdy
                    nx, ny, nz = 0, -sy, 0
                else:
                    rz += sz; hit_t = tmz; tmz += tdz
                    nx, ny, nz = 0, 0, -sz
                if rx < 0 or rx >= _G or ry < 0 or ry >= _G or rz < 0 or rz >= _G:
                    break
                v = voxels[rx, ry, rz]
                if v != EMPTY:
                    hit_v = v
                    hx, hy, hz = rx, ry, rz
                    break
            return hit_v, hit_t, nx, ny, nz, hx, hy, hz

        @ti.kernel
        def _render_kernel(voxels: ti.template(), glow: ti.template(),
                           lava_pos: ti.template(), n_lava: ti.template(),
                           img: ti.template(),
                           cam_x: ti.f32, cam_y: ti.f32, cam_z: ti.f32,
                           yaw: ti.f32, pitch: ti.f32, ambient: ti.f32,
                           do_shadows: ti.i32):
            for py, px in ti.ndrange(_H, _W):
                ha = yaw + (float(px) / float(_W) - 0.5) * 1.2217
                va = pitch - (float(py) / float(_H) - 0.5) * 0.9599
                dx = ti.cos(ha) * ti.cos(va)
                dy = ti.sin(ha) * ti.cos(va)
                dz = ti.sin(va)

                hit_v, hit_t, nx, ny, nz, hx, hy, hz = dda(
                    voxels, cam_x, cam_y, cam_z, dx, dy, dz, 160)

                # water reflection: bounce the ray and see what's mirrored
                refl_r, refl_g, refl_b = 0.0, 0.0, 0.0
                if hit_v == WATER:
                    # reflect direction across face normal
                    fnx, fny, fnz = float(nx), float(ny), float(nz)
                    dot = dx * fnx + dy * fny + dz * fnz
                    rdx = dx - 2.0 * dot * fnx
                    rdy = dy - 2.0 * dot * fny
                    rdz = dz - 2.0 * dot * fnz
                    # start just off the surface
                    sox = float(hx) + 0.5 + fnx * 0.15
                    soy = float(hy) + 0.5 + fny * 0.15
                    soz = float(hz) + 0.5 + fnz * 0.15
                    rv, rt, _, _, _, _, _, _ = dda(
                        voxels, sox, soy, soz, rdx, rdy, rdz, 80)
                    if rv == 0:
                        # reflected sky
                        refl_r, refl_g, refl_b = 0.10, 0.14, 0.22
                    elif rv == LAVA:
                        refl_r, refl_g, refl_b = 1.0 * 1.7, 0.30 * 1.7, 0.08 * 1.7
                    elif rv == WISP:
                        refl_r, refl_g, refl_b = 0.65 * 1.7, 0.95 * 1.7, 1.0 * 1.7
                    elif rv == ACCENT:
                        refl_r, refl_g, refl_b = 0.95, 0.55, 0.25
                    elif rv == WATER:
                        refl_r, refl_g, refl_b = 0.15, 0.45, 0.85
                    else:
                        # reflected wall (dimmer, fresnel-ish)
                        refl_r, refl_g, refl_b = 0.35, 0.38, 0.45
                    # fresnel: more reflective at grazing angles
                    fres = 0.4 + 0.6 * (1.0 - ti.abs(dot))
                    # blend water base with reflection
                    r = 0.15 * (1 - fres) + refl_r * fres
                    g = 0.45 * (1 - fres) + refl_g * fres
                    b = 0.85 * (1 - fres) + refl_b * fres
                    # skip the normal shading path for water
                    img[py, px, 0] = r / (1.0 + 0.03 * hit_t)
                    img[py, px, 1] = g / (1.0 + 0.03 * hit_t)
                    img[py, px, 2] = b / (1.0 + 0.03 * hit_t)
                elif hit_v == 0:
                    t = float(py) / float(_H)
                    img[py, px, 0] = 0.05 + 0.10 * t
                    img[py, px, 1] = 0.08 + 0.12 * t
                    img[py, px, 2] = 0.15 + 0.20 * t
                else:
                    # base color
                    r, g, b = 0.62, 0.66, 0.72
                    if hit_v == ACCENT:
                        r, g, b = 0.95, 0.55, 0.25
                    elif hit_v == WISP:
                        r, g, b = 0.65, 0.95, 1.0
                    elif hit_v == LAVA:
                        r, g, b = 1.0, 0.30, 0.08
                    elif hit_v == WATER:
                        r, g, b = 0.15, 0.45, 0.85
                    if nz > 0 and hit_v == WALL:
                        r, g, b = 0.30, 0.32, 0.38

                    # texture
                    tex = 0.88 + 0.24 * hash3(hx, hy, hz)
                    r *= tex; g *= tex; b *= tex

                    # face shading
                    shade = 0.85
                    if ny != 0:
                        shade = 0.65
                    elif nz > 0:
                        shade = 0.95
                    elif nz < 0:
                        shade = 0.45

                    # AO
                    occ = 0.0
                    bx = ti.max(0, ti.min(_G - 1, hx + nx))
                    by = ti.max(0, ti.min(_G - 1, hy + ny))
                    bz = ti.max(0, ti.min(_G - 1, hz + nz))
                    if nx != 0:
                        for sgn in ti.static((-1, 1)):
                            nyy = ti.max(0, ti.min(_G - 1, by + sgn))
                            if voxels[bx, nyy, bz] != EMPTY:
                                occ += 1.0
                            nzz = ti.max(0, ti.min(_G - 1, bz + sgn))
                            if voxels[bx, by, nzz] != EMPTY:
                                occ += 1.0
                    elif ny != 0:
                        for sgn in ti.static((-1, 1)):
                            nxx = ti.max(0, ti.min(_G - 1, bx + sgn))
                            if voxels[nxx, by, bz] != EMPTY:
                                occ += 1.0
                            nzz = ti.max(0, ti.min(_G - 1, bz + sgn))
                            if voxels[bx, by, nzz] != EMPTY:
                                occ += 1.0
                    else:
                        for sgn in ti.static((-1, 1)):
                            nxx = ti.max(0, ti.min(_G - 1, bx + sgn))
                            if voxels[nxx, by, bz] != EMPTY:
                                occ += 1.0
                            nyy = ti.max(0, ti.min(_G - 1, by + sgn))
                            if voxels[bx, nyy, bz] != EMPTY:
                                occ += 1.0
                    ao = ti.max(1.0 - 0.5 * (occ / 4.0), 0.3)
                    shade *= ao
                    shade = shade / (1.0 + 0.055 * hit_t)

                    is_glow = (hit_v == WISP) or (hit_v == LAVA)
                    boost = 1.7 if is_glow else 1.0
                    amb = 1.0 if is_glow else ambient

                    # soft shadows: shadow ray toward nearest lava
                    shadow = 1.0
                    if do_shadows == 1 and not is_glow and n_lava[None] > 0:
                        # find nearest lava pool
                        best_d2 = 1e9
                        blx, bly, blz = 0.0, 0.0, 0.0
                        for li in range(n_lava[None]):
                            lx = lava_pos[li, 0]
                            ly = lava_pos[li, 1]
                            lz = lava_pos[li, 2]
                            d2 = (lx - float(hx))**2 + (ly - float(hy))**2 + (lz - float(hz))**2
                            if d2 < best_d2:
                                best_d2 = d2
                                blx, bly, blz = lx, ly, lz
                        # shadow ray from hit point toward lava
                        sx = float(hx) + 0.5 + float(nx) * 0.1
                        sy = float(hy) + 0.5 + float(ny) * 0.1
                        sz = float(hz) + 0.5 + float(nz) * 0.1
                        ldx = blx - sx
                        ldy = bly - sy
                        ldz = blz - sz
                        llen = ti.sqrt(ldx*ldx + ldy*ldy + ldz*ldz) + 1e-6
                        ldx /= llen; ldy /= llen; ldz /= llen
                        sh_v, sh_t, _, _, _, _, _, _ = dda(
                            voxels, sx, sy, sz, ldx, ldy, ldz, 80)
                        # if hit something before reaching lava -> shadow
                        # (lava is at distance llen; allow small epsilon)
                        if sh_v != 0 and sh_v != LAVA and sh_t < llen - 0.5:
                            shadow = 0.35
                        elif sh_v == 0:
                            shadow = 0.6  # no direct line (occluded by distance)

                    rr = r * shade * boost * amb * shadow
                    gg = g * shade * boost * amb * shadow
                    bb = b * shade * boost * amb * shadow

                    # lava glow (ambient light from nearby lava)
                    if not is_glow:
                        gl = glow[hx, hy, hz]
                        rr += 1.0 * gl * 0.6 * shadow
                        gg += 0.35 * gl * 0.6 * shadow
                        bb += 0.10 * gl * 0.6 * shadow

                    img[py, px, 0] = rr
                    img[py, px, 1] = gg
                    img[py, px, 2] = bb

        return _render_kernel

    _render_kernel_fn = None

    def _get_kernel():
        global _render_kernel_fn
        if _render_kernel_fn is None:
            _render_kernel_fn = _define_kernels()
        return _render_kernel_fn


def _compute_lava_positions(voxels):
    """Find lava pool centers (for shadow rays)."""
    from .voxel import LAVA as _LAVA
    pos = np.argwhere(voxels == _LAVA)
    if len(pos) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    # cluster into pools (simple: take every Nth)
    return (pos[::max(1, len(pos)//32)] + 0.5).astype(np.float32)[:64]


def render(voxels, cam_pos, yaw, pitch=0.0, ambient=1.0, shadows=True,
           out_res=256):
    """Render with Taichi. Same API as voxel.raycast, but faster.

    Falls back to numpy if Taichi is not available.
    """
    if not _TI_AVAILABLE:
        from .voxel import raycast as _np_raycast
        return _np_raycast(voxels, cam_pos, yaw, pitch=pitch,
                           W=64, H=64, ambient=ambient)

    _ensure_init()
    global _W, _H
    # (for now, fixed at 256; dynamic res is a TODO)
    assert voxels.shape == (_G, _G, _G), f"expected ({_G},{_G},{_G})"

    from .voxel import _lava_glow
    _voxels_ti.from_numpy(np.ascontiguousarray(voxels.astype(np.int32)))
    _glow_ti.from_numpy(_lava_glow(voxels))

    lava_pos = _compute_lava_positions(voxels)
    n = min(len(lava_pos), 64)
    if n > 0:
        _lava_pos_ti.from_numpy(
            np.ascontiguousarray(
                np.pad(lava_pos[:n],
                       ((0, 64-n), (0, 0))).astype(np.float32)))
    _n_lava[None] = n

    kernel = _get_kernel()
    kernel(_voxels_ti, _glow_ti, _lava_pos_ti, _n_lava, _img_ti,
           float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]),
           float(yaw), float(pitch), float(ambient),
           1 if shadows else 0)
    return np.clip(_img_ti.to_numpy(), 0, 1).astype(np.float32)
