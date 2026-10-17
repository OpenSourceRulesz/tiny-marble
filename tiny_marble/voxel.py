"""tiny-4D: a small 4D world model, InSpatio-style.

A 16x16x16 voxel world (4096 cells -- barely bigger than our 256 tiles)
with real first-person rendering, drifting entities, and time control.
The InSpatio mapping, miniaturized:

- their persistent 4D world state  -> explicit voxel grid + entity list
  (viewpoint-independent Local World State, tiny)
- their 6-DoF free roaming          -> first-person camera (pos + yaw),
  vectorized DDA raycaster, pure numpy, real-time
- their temporal control            -> explicit state makes it trivial:
  pause = don't step dynamics, reverse = restore trajectory buffer.
  ("No man ever steps in the same river twice" -- here you can.)
- their generative world prior     -> voxel diffusion (voxel_diffusion.py;
  procedural worlds work without it)

Why voxels instead of faking 3D with a 2D raycaster: a 16^3 grid is
cheap, the diffusion extends directly, and the world is genuinely
three-dimensional -- walls have tops, pillars have heights, you can
look up and down.
"""

import numpy as np
from collections import deque

from .nn import MLP, Adam  # noqa: F401  (re-exported for voxel_diffusion)

# 3D world styles (rooms, cavern, arena)
STYLES = ("rooms", "cavern", "arena")
N_STYLES = len(STYLES)

G = 32          # grid is G^3 (32^3 = 32768 cells, 8x the detail of 16^3)
RES = 64        # render resolution
FOV_H = np.deg2rad(70)
FOV_V = np.deg2rad(55)
EYE_Z = 20.0

# voxel values
EMPTY, WALL, ACCENT, WISP, LAVA, WATER = 0, 1, 2, 3, 4, 5


# ----------------------------------------------------------------------
# procedural voxel worlds
# ----------------------------------------------------------------------
def _clear_spawn(v, rng):
    cx, cy = int(rng.integers(6, G - 6)), int(rng.integers(6, G - 6))
    v[cx - 2:cx + 3, cy - 2:cy + 3, 1:] = EMPTY
    return cx, cy


def _cellular_cave(G, rng, fill=0.42, iterations=4):
    """Organic cave via cellular automata (6-neighborhood).
    Creates connected tunnels, not floating blobs."""
    v = (rng.random((G, G, G)) < fill).astype(np.int64)
    v[0, :, :] = v[-1, :, :] = 1
    v[:, 0, :] = v[:, -1, :] = 1
    v[:, :, 0] = v[:, :, -1] = 1
    for _ in range(iterations):
        # 6-neighborhood (face-adjacent only)
        padded = np.pad(v, 1)
        neighbors = (padded[2:, 1:-1, 1:-1] + padded[:-2, 1:-1, 1:-1] +
                     padded[1:-1, 2:, 1:-1] + padded[1:-1, :-2, 1:-1] +
                     padded[1:-1, 1:-1, 2:] + padded[1:-1, 1:-1, :-2])
        # wall if 4+ of 6 neighbors are walls (stricter smoothing)
        v = (neighbors >= 4).astype(np.int64)
        v[0, :, :] = v[-1, :, :] = 1
        v[:, 0, :] = v[:, -1, :] = 1
        v[:, :, 0] = v[:, :, -1] = 1
    return v


def _add_stalactites(v, rng, n=40):
    """Stalactites (ceiling) and stalagmites (floor) for caverns."""
    G = v.shape[0]
    for _ in range(n):
        x, y = int(rng.integers(2, G-2)), int(rng.integers(2, G-2))
        # find ceiling (first wall from top)
        for z in range(G-2, 0, -1):
            if v[x, y, z] == 1:
                # stalactite hanging down
                length = int(rng.integers(2, 6))
                for dz in range(1, length+1):
                    if z-dz > 1 and v[x, y, z-dz] == 0:
                        v[x, y, z-dz] = 1
                    else:
                        break
                break
        # stalagmite (from floor up)
        for z in range(1, G-1):
            if v[x, y, z] == 0:
                length = int(rng.integers(2, 5))
                for dz in range(length):
                    if z+dz < G-1 and v[x, y, z+dz] == 0:
                        v[x, y, z+dz] = 1
                    else:
                        break
                break
    return v


def procedural_voxel(style=0, seed=None):
    """32^3 voxel world. 0=empty, 1=wall, 2=accent. Floor z=0 solid,
    border walls full height. Returns (voxels, (spawn_x, spawn_y))."""
    rng = np.random.default_rng(seed)
    v = np.zeros((G, G, G), dtype=np.int64)
    v[:, :, 0] = WALL                       # floor
    v[0, :, 1:] = v[-1, :, 1:] = WALL        # border
    v[:, 0, 1:] = v[:, -1, 1:] = WALL
    style = int(style) % N_STYLES

    if style == 0:  # rooms: wall segments with door gaps + pillars
        for _ in range(int(rng.integers(3, 5))):
            if rng.random() < 0.5:  # vertical wall run
                x = int(rng.integers(3, G - 3))
                y0 = int(rng.integers(2, G - 4))
                y1 = int(rng.integers(y0 + 1, G - 2))
                gap = int(rng.integers(y0, y1))
                v[x, y0:y1, 1:] = WALL
                v[x, gap, 1:] = EMPTY
            else:                   # horizontal wall run
                y = int(rng.integers(3, G - 3))
                x0 = int(rng.integers(2, G - 4))
                x1 = int(rng.integers(x0 + 1, G - 2))
                gap = int(rng.integers(x0, x1))
                v[x0:x1, y, 1:] = WALL
                v[gap, y, 1:] = EMPTY
        n_pil = int(rng.integers(12, 25))
    elif style == 1:  # cavern: cellular automata caves + stalactites
        cave = _cellular_cave(G, rng, fill=0.45, iterations=4)
        # carve out the cave (invert: CA gives walls, we want tunnels)
        # Actually CA already gives us walls; empty the interior
        v[cave == 0] = EMPTY
        v[cave == 1] = WALL
        # ensure floor stays
        v[:, :, 0] = WALL
        v = _add_stalactites(v, rng, n=50)
        n_pil = 0
    else:  # arena: open + pillars
        n_pil = int(rng.integers(24, 44))

    for _ in range(n_pil):
        x, y = int(rng.integers(2, G - 2)), int(rng.integers(2, G - 2))
        h = int(rng.integers(6, 18))
        v[x, y, 1:1 + h] = WALL

    # accents: some wall voxels get a warm tint
    wallz = (v == WALL) & (np.arange(G)[None, None, :] > 0)
    v[rng.random((G, G, G)) < 0.12 * wallz] = ACCENT

    cx, cy = _clear_spawn(v, rng)
    return v, (cx, cy)


# ----------------------------------------------------------------------
# vectorized 3D DDA raycaster
# ----------------------------------------------------------------------
def _ray_dirs(yaw, pitch=0.0, W=RES, H=RES):
    cols = (np.arange(W) / W - 0.5) * FOV_H
    rows = (np.arange(H) / H - 0.5) * FOV_V
    ha = yaw + cols[None, :].repeat(H, axis=0)          # (H, W)
    va = pitch - rows[:, None].repeat(W, axis=1)        # (H, W)
    ch, cv = np.cos(ha), np.cos(va)
    dirs = np.stack([ch * cv, np.sin(ha) * cv, np.sin(va)],
                    axis=-1).reshape(-1, 3)
    n = np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs / np.maximum(n, 1e-9)


def _lava_glow(voxels):
    """Blurred lava mask: how much orange light hits each voxel."""
    glow = (voxels == LAVA).astype(np.float32)
    for _ in range(2):
        padded = np.pad(glow, 1)
        glow = sum(padded[i:i + G, j:j + G, k:k + G]
                   for i in range(3) for j in range(3)
                   for k in range(3)) / 27.0
    return np.clip(glow * 4.0, 0, 1).astype(np.float32)


def raycast(voxels, cam_pos, yaw, pitch=0.0, W=RES, H=RES, ambient=1.0,
            glow_grid=None):
    """First-person RGB render. voxels: (G,G,G) int. cam_pos: (3,) float
    in voxel units. ambient scales overall light (dark worlds). Returns
    (H, W, 3) float32 in [0,1]. Pure numpy."""
    N = H * W
    dirs = _ray_dirs(yaw, pitch, W, H).astype(np.float64)
    pos = np.tile(np.asarray(cam_pos, dtype=np.float64), (N, 1))
    step = np.sign(dirs).astype(np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        tDelta = np.where(dirs != 0, np.abs(1.0 / dirs), np.inf)
    ri = np.floor(pos).astype(np.int64)
    tMax = np.full((N, 3), np.inf)
    nz = step != 0
    # distance to first voxel boundary along each axis
    tMax[nz] = np.where(step[nz] > 0,
                        (ri[nz] + 1 - pos[nz]), (pos[nz] - ri[nz])) \
        * tDelta[nz]

    hit = np.zeros(N, dtype=bool)
    hit_t = np.full(N, np.inf)
    hit_n = np.zeros((N, 3), dtype=np.int64)   # face normal
    hit_v = np.zeros(N, dtype=np.int64)        # voxel value
    active = np.ones(N, dtype=bool)
    # camera inside solid -> nudge out (shouldn't happen with collision)
    active &= (voxels[np.clip(ri[:, 0], 0, G - 1),
                      np.clip(ri[:, 1], 0, G - 1),
                      np.clip(ri[:, 2], 0, G - 1)] == EMPTY)

    ar = np.arange(N)
    for _ in range(160):
        if not active.any():
            break
        tm = np.where(active[:, None], tMax, np.inf)
        ax = np.argmin(tm, axis=1)
        adv = active
        hit_t[adv] = tMax[ar[adv], ax[adv]]
        ri[ar[adv], ax[adv]] += step[ar[adv], ax[adv]]
        hit_n[ar[adv]] = 0
        hit_n[ar[adv], ax[adv]] = -step[ar[adv], ax[adv]]
        tMax[ar[adv], ax[adv]] += tDelta[ar[adv], ax[adv]]
        oob = (ri < 0).any(axis=1) | (ri >= G).any(axis=1)
        valid = adv & ~oob
        vv = voxels[ri[valid, 0], ri[valid, 1], ri[valid, 2]]
        got = np.zeros(N, dtype=bool)
        got[valid] = vv > EMPTY
        hit |= got
        hit_v[got] = voxels[ri[got, 0], ri[got, 1], ri[got, 2]]
        active = ~hit & ~oob & adv

    img = np.zeros((N, 3), dtype=np.float64)
    # sky gradient for rays that escaped
    sky = ~hit
    img[sky, 0] = 0.05
    img[sky, 1] = 0.07
    img[sky, 2] = 0.12 + 0.10 * np.clip(dirs[sky, 2], 0, 1)
    # shade hits: face normal + distance + material
    h = hit
    if h.any():
        M = h.sum()
        shade = np.ones(M)
        nx = hit_n[h]
        rih = ri[h]
        # face shading: x faces brightest, y mid, z top bright / bottom dark
        shade = np.where(nx[:, 0] != 0, 0.85,
                np.where(nx[:, 1] != 0, 0.65,
                np.where(nx[:, 2] > 0, 0.95, 0.45)))
        # ambient occlusion: sample the 4 voxels around the hit face;
        # crevices and corners go dark (this is what makes it look 3D)
        occ = np.zeros(M, dtype=np.float32)
        for ax in range(3):
            m_ax = nx[:, ax] != 0
            if not m_ax.any():
                continue
            bx = np.clip(rih[m_ax] + nx[m_ax], 0, G - 1)
            for t in [t for t in range(3) if t != ax]:
                for sgn in (-1, 1):
                    nb = bx.copy()
                    nb[:, t] = np.clip(nb[:, t] + sgn, 0, G - 1)
                    occ[m_ax] += (voxels[nb[:, 0], nb[:, 1],
                                        nb[:, 2]] != EMPTY)
        ao = np.maximum(1.0 - 0.5 * (occ / 4.0), 0.3)
        shade = shade * ao
        dist = np.maximum(hit_t[h], 1e-3)
        shade = shade / (1.0 + 0.055 * dist)          # distance fog
        base = np.zeros((M, 3))
        wall = hit_v[h] == WALL
        acc = hit_v[h] == ACCENT
        wisp = hit_v[h] == WISP
        lava = hit_v[h] == LAVA
        water = hit_v[h] == WATER
        base[wall] = (0.62, 0.66, 0.72)               # cool stone
        base[acc] = (0.95, 0.55, 0.25)                # warm accent
        base[wisp] = (0.65, 0.95, 1.0)                # bright wisp
        base[lava] = (1.0, 0.30, 0.08)                # molten lava
        base[water] = (0.15, 0.45, 0.85)               # water
        # floor (top face of z=0) darker
        is_floor = (nx[:, 2] > 0) & (hit_v[h] == WALL)
        base[is_floor] = (0.30, 0.32, 0.38)
        # texture: deterministic per-voxel variation breaks flat colors
        hsh = ((rih[:, 0] * 73856093) ^ (rih[:, 1] * 19349663) ^
               (rih[:, 2] * 83492791)) % 100 / 100.0
        base *= (0.88 + 0.24 * hsh)[:, None]
        glow = wisp | lava
        img[h] = base * shade[:, None] * np.where(
            glow[:, None], 1.7, 1.0)
        # lava light: nearby walls catch the orange glow
        if glow_grid is not None:
            gl = glow_grid[rih[:, 0], rih[:, 1], rih[:, 2]]
            img[h] += (np.array([1.0, 0.35, 0.1])[None, :] *
                       (gl * 0.6)[:, None] * (~glow)[:, None])
    # emissive materials ignore ambient darkness
    if h.any():
        img[h] *= np.where(glow[:, None], 1.0, ambient)
    return np.clip(img.reshape(H, W, 3), 0, 1).astype(np.float32)


# ----------------------------------------------------------------------
# the 4D world: voxels + entities + time control
# ----------------------------------------------------------------------
class VoxelWorld:
    """A tiny 4D world: persistent voxel state, first-person camera,
    drifting wisp entities, and explicit time control.

    Time is the 4th dimension made trivial by explicit state:
    pause() freezes dynamics, reverse(n) restores the trajectory --
    the river you CAN step in twice.
    """

    N_WISPS = 4

    def __init__(self, voxels, spawn=None, seed=None, ambient=1.0,
                 use_taichi=True):
        self.voxels = np.asarray(voxels, dtype=np.int64)
        assert self.voxels.shape == (G, G, G)
        self.ambient = float(ambient)
        self.pitch = 0.0
        self.rng = np.random.default_rng(seed)
        # taichi backend (256x256, 30x faster) if available; numpy fallback
        self._use_taichi = bool(use_taichi)
        if self._use_taichi:
            try:
                from .taichi_render import is_available
                self._use_taichi = is_available()
            except ImportError:
                self._use_taichi = False
        if spawn is None:
            spawn = (G // 2, G // 2)
        self.pos = np.array([spawn[0] + 0.5, spawn[1] + 0.5, EYE_Z],
                            dtype=np.float64)
        self.yaw = float(self.rng.uniform(0, 2 * np.pi))
        self.wisps = self.rng.uniform(3, G - 3, size=(self.N_WISPS, 3))
        self.wisps[:, 2] = self.rng.uniform(8, 24, size=self.N_WISPS)
        ang = self.rng.uniform(0, 2 * np.pi, size=self.N_WISPS)
        self.wvel = np.stack([np.cos(ang), np.sin(ang),
                              np.zeros(self.N_WISPS)], axis=1) * 0.25
        self._traj = deque(maxlen=256)  # for time reversal
        self._push_traj()
        self.paused = False
        self.t = 0

    # -- time -----------------------------------------------------------
    def _push_traj(self):
        self._traj.append(self.wisps.copy())

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def reverse(self, n=24):
        """Step time backwards n ticks (wisp dynamics only)."""
        for _ in range(min(n, len(self._traj) - 1)):
            self._traj.pop()
        self.wisps = self._traj[-1].copy()
        self._push_traj()

    # -- movement -------------------------------------------------------
    def _solid(self, x, y, z):
        xi, yi, zi = int(x), int(y), int(z)
        if not (0 <= xi < G and 0 <= yi < G and 0 <= zi < G):
            return True
        return self.voxels[xi, yi, zi] != EMPTY

    def move(self, forward, strafe=0.0):
        """forward/strafe in [-1, 1]; sliding collision."""
        fx, fy = np.cos(self.yaw), np.sin(self.yaw)
        dx = (fx * forward - fy * strafe) * 0.7
        dy = (fy * forward + fx * strafe) * 0.7
        for nx, ny in ((self.pos[0] + dx, self.pos[1] + dy),
                       (self.pos[0] + dx, self.pos[1]),
                       (self.pos[0], self.pos[1] + dy)):
            if not (self._solid(nx, ny, self.pos[2] - 2.5)
                    or self._solid(nx, ny, self.pos[2] + 1.0)):
                self.pos[0], self.pos[1] = nx, ny
                break

    def turn(self, dyaw):
        self.yaw += float(dyaw)

    def step(self, forward=0.0, strafe=0.0, dyaw=0.0):
        """Advance the 4th dimension one tick (unless paused)."""
        self.turn(dyaw)
        self.move(forward, strafe)
        if not self.paused:
            self.t += 1
            for i in range(self.N_WISPS):
                for d in (0, 1, 2):
                    np_ = self.wisps[i].copy()
                    np_[d] += self.wvel[i, d]
                    np_[d] += 0.05 * np.sin(self.t * 0.1 + i) \
                        if d == 2 else 0.0
                    xi, yi, zi = int(np_[0]), int(np_[1]), int(np_[2])
                    if not (1 <= xi < G - 1 and 1 <= yi < G - 1
                            and 1 <= zi < G - 1) or \
                            self.voxels[xi, yi, zi] != EMPTY:
                        self.wvel[i, d] *= -1
                    else:
                        self.wisps[i, d] = np_[d]
            self._push_traj()

    # -- rendering ------------------------------------------------------
    def render(self):
        """First-person RGB frame with wisps composited as bright voxels.
        Uses Taichi (256x256) if available, else numpy (64x64)."""
        vox = self.voxels.copy()
        for w in self.wisps:
            xi, yi, zi = int(w[0]), int(w[1]), int(w[2])
            if 0 <= xi < G and 0 <= yi < G and 0 <= zi < G:
                if vox[xi, yi, zi] == EMPTY:
                    vox[xi, yi, zi] = WISP
        if self._use_taichi:
            from .taichi_render import render as _ti_render
            return _ti_render(vox, self.pos, self.yaw, pitch=self.pitch,
                              ambient=self.ambient)
        if not hasattr(self, "_glow") or self._glow_vox is not self.voxels:
            self._glow = _lava_glow(self.voxels)
            self._glow_vox = self.voxels
        return raycast(vox, self.pos, self.yaw, pitch=self.pitch,
                       ambient=self.ambient, glow_grid=self._glow)


def dream_voxel_world(prior=None, style=None, seed=None):
    """Dream a walkable 16^3 voxel world. `prior` is a SlicePrior (the
    working 3D dreamer); pass None for a procedural world. Returns a
    VoxelWorld with a cleared spawn pocket."""
    rng = np.random.default_rng(seed)
    if style is None:
        style = int(rng.integers(N_STYLES))
    if prior is not None:
        v = prior.sample(style=style, seed=int(rng.integers(1e9)))
    else:
        v, _ = procedural_voxel(style=style, seed=int(rng.integers(1e9)))
    # clear a spawn pocket (the prior doesn't know about spawns)
    cx, cy = int(rng.integers(6, G - 6)), int(rng.integers(6, G - 6))
    v[cx - 4:cx + 5, cy - 4:cy + 5, 1:15] = EMPTY
    return VoxelWorld(v, spawn=(cx, cy), seed=int(rng.integers(1e9)))


__all__ = ["G", "RES", "VoxelWorld", "dream_voxel_world",
           "procedural_voxel", "raycast", "STYLES",
           "EMPTY", "WALL", "ACCENT", "WISP", "LAVA", "WATER"]
