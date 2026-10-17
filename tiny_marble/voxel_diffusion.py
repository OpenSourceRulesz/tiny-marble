"""Voxel diffusion prior: dream 16^3 worlds that never existed.

Same proven recipe as the 2D WorldPrior (x_0 parameterization --
epsilon-prediction fails for tiny MLPs; sinusoidal time embedding),
scaled to 4096 binary voxels. ~2.2M params: still tiny next to
InSpatio's 1.3B, still pure numpy.
"""

import time

import numpy as np

from .nn import MLP, Adam
from .voxel import N_STYLES
from .voxel import G, WALL, procedural_voxel

DIM = G * G * G          # 4096
T = 40


def _time_embed(t, dim=16):
    t = np.asarray(t, dtype=np.float64).reshape(-1, 1)
    freqs = np.exp(np.linspace(0, np.log(100.0), dim // 2))[None, :]
    return np.concatenate([np.sin(t * freqs), np.cos(t * freqs)],
                          axis=1).astype(np.float32)


class VoxelPrior:
    def __init__(self, n_styles=N_STYLES, seed=0):
        self.n_styles = n_styles
        self.net = MLP([DIM + 16 + n_styles, 256, 256, DIM], seed=seed)

    def _condition(self, t_idx, styles):
        B = len(t_idx)
        te = _time_embed(t_idx / T) * 2.0
        onehot = np.zeros((B, self.n_styles), dtype=np.float32)
        onehot[np.arange(B), styles] = 3.0
        return np.concatenate([te, onehot], axis=1)

    def _q_sample(self, x0, t_idx, rng):
        # cosine-ish noise schedule on +/-1 voxels
        alpha = 1.0 - (t_idx / T) * 0.85
        alpha = alpha.reshape(-1, 1)
        noise = rng.standard_normal(x0.shape).astype(np.float32)
        return (alpha * x0 + np.sqrt(1 - alpha ** 2) * noise).astype(
            np.float32)

    def train_step(self, x0, styles, opt, rng):
        B = len(x0)
        t_idx = rng.integers(1, T + 1, size=B)
        xt = self._q_sample(x0, t_idx, rng)
        cond = self._condition(t_idx, styles)
        inp = np.concatenate([xt, cond], axis=1)
        pred = self.net.forward(inp)
        err = pred - x0
        loss = float(np.mean(err ** 2))
        self.net.backward((2.0 * err / err.size).astype(np.float32))
        opt.step()
        return loss

    def fit(self, maps, styles, steps=1200, batch=64, seed=0,
            log_every=200):
        rng = np.random.default_rng(seed)
        opt = Adam(self.net.params_and_grads(), lr=2e-4)
        N = len(maps)
        t0 = time.time()
        for step in range(1, steps + 1):
            idx = rng.integers(N, size=batch)
            loss = self.train_step(maps[idx], styles[idx], opt, rng)
            if log_every and step % log_every == 0:
                print(f"voxel step {step:5d}/{steps} loss={loss:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        return self

    def sample(self, style=0, n=1, seed=None, steps=T):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n, DIM)).astype(np.float32)
        styles = np.full(n, int(style) % self.n_styles)
        for ti in range(steps, 0, -1):
            t_idx = np.full(n, ti)
            cond = self._condition(t_idx, styles)
            x0_hat = self.net.forward(
                np.concatenate([x, cond], axis=1))
            if ti > 1:
                alpha = 1.0 - (ti / T) * 0.85
                x = (alpha * x0_hat + np.sqrt(max(1 - alpha ** 2, 0))
                     * rng.standard_normal((n, DIM)).astype(np.float32)
                     ).astype(np.float32)
            else:
                x = x0_hat
        # floor stays solid, border stays walls (structural guarantee)
        vox = (x.reshape(n, G, G, G) > 0).astype(np.int64)
        vox[:, :, :, 0] = 1
        vox[:, 0, :, 1:] = vox[:, -1, :, 1:] = 1
        vox[:, :, 0, 1:] = vox[:, :, -1, 1:] = 1
        return vox

    def save(self, path):
        arrays = {f"net.{k}": v
                  for k, v in self.net.parameters().items()}
        arrays["n_styles"] = np.array([self.n_styles])
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path, seed=0):
        d = np.load(path)
        self = cls(n_styles=int(d["n_styles"][0]), seed=seed)
        self.net.load_parameters(
            {k.split(".", 1)[1]: d[k] for k in d.files
             if k.startswith("net.")})
        return self


__all__ = ["VoxelPrior", "HeightfieldPrior", "SlicePrior",
           "DIM", "T"]


class SlicePrior:
    """Autoregressive 3D dreamer: generate the world one horizontal slice
    at a time, conditioned on the slice below.

    Each slice is GxG dims. The model
    learns P(slice_z | slice_{z-1}, height z, style): walls continue up,
    pillars stop, rooms stay open. Like a 3D printer with a diffusion
    print head. This is the working 3D dreamer.

    If attr_dim is set, conditions on attribute vectors instead of style
    indices (for prompt-to-world: text -> attributes -> voxels).
    """

    SDIM = G * G

    def __init__(self, n_styles=N_STYLES, seed=0, attr_dim=0):
        self.n_styles = n_styles
        self.attr_dim = attr_dim
        cond_dim = attr_dim if attr_dim > 0 else n_styles
        # (noisy slice + prev slice + height 1 + time 16 + cond)
        self.net = MLP([2 * self.SDIM + 1 + 16 + cond_dim, 128, 128,
                        self.SDIM], seed=seed)

    def _condition(self, t_idx, styles_or_attrs, heights):
        B = len(t_idx)
        te = _time_embed(t_idx / T) * 2.0
        if self.attr_dim > 0:
            # attributes: already a (B, attr_dim) float array
            cond = np.asarray(styles_or_attrs, dtype=np.float32)
        else:
            onehot = np.zeros((B, self.n_styles), dtype=np.float32)
            onehot[np.arange(B), styles_or_attrs] = 3.0
            cond = onehot
        h = (np.asarray(heights, dtype=np.float32).reshape(-1, 1)
             / 15.0 * 2 - 1)
        return np.concatenate([te, cond, h], axis=1)

    def _q_sample(self, x0, t_idx, rng):
        alpha = (1.0 - (t_idx / T) * 0.85).reshape(-1, 1)
        noise = rng.standard_normal(x0.shape).astype(np.float32)
        return (alpha * x0 + np.sqrt(1 - alpha ** 2) * noise).astype(
            np.float32)

    def train_step(self, prev, x0, styles, heights, opt, rng):
        B = len(x0)
        t_idx = rng.integers(1, T + 1, size=B)
        xt = self._q_sample(x0, t_idx, rng)
        cond = self._condition(t_idx, styles, heights)
        inp = np.concatenate([xt, prev, cond], axis=1)
        pred = self.net.forward(inp)
        err = pred - x0
        # weight walls: sparse structure matters
        w = np.where(x0 > 0, 3.0, 1.0)
        loss = float(np.mean(w * err ** 2))
        self.net.backward((2.0 * w * err / err.size).astype(np.float32))
        opt.step()
        return loss

    def fit(self, prevs, slices, styles, heights, steps=3000, batch=128,
            seed=0, log_every=500):
        rng = np.random.default_rng(seed)
        opt = Adam(self.net.params_and_grads(), lr=1e-3)
        N = len(slices)
        t0 = time.time()
        for step in range(1, steps + 1):
            idx = rng.integers(N, size=batch)
            loss = self.train_step(prevs[idx], slices[idx], styles[idx],
                                   heights[idx], opt, rng)
            if log_every and step % log_every == 0:
                print(f"slice step {step:5d}/{steps} loss={loss:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        return self

    def _denoise(self, xt, prev, style_or_attr, height, rng, steps=T):
        x = xt
        for ti in range(steps, 0, -1):
            t_idx = np.array([ti])
            if self.attr_dim > 0:
                cond = self._condition(t_idx, style_or_attr.reshape(1, -1),
                                       np.array([height]))
            else:
                cond = self._condition(t_idx, np.array([style_or_attr]),
                                       np.array([height]))
            x0_hat = self.net.forward(
                np.concatenate([x, prev.reshape(1, -1), cond], axis=1))
            if ti > 1:
                alpha = 1.0 - (ti / T) * 0.85
                x = (alpha * x0_hat + np.sqrt(max(1 - alpha ** 2, 0))
                     * rng.standard_normal((1, self.SDIM))
                     .astype(np.float32)).astype(np.float32)
            else:
                x = x0_hat
        return x[0]

    def sample(self, style=0, seed=None):
        rng = np.random.default_rng(seed)
        v = np.zeros((G, G, G), dtype=np.int64)
        v[:, :, 0] = WALL
        v[0, :, 1:] = v[-1, :, 1:] = WALL
        v[:, 0, 1:] = v[:, -1, 1:] = WALL
        prev = np.where(v[:, :, 0].reshape(-1) > 0, 1.0, -1.0
                        ).astype(np.float32).reshape(1, -1)
        for z in range(1, G):
            xt = rng.standard_normal((1, self.SDIM)).astype(np.float32)
            s = self._denoise(xt, prev, int(style) % self.n_styles,
                              z, rng)
            sl = (s.reshape(G, G) > 0).astype(np.int64)
            # keep border walls structural
            sl[0, :] = sl[-1, :] = sl[:, 0] = sl[:, -1] = 1
            v[:, :, z] = sl
            prev = np.where(s > 0, 1.0, -1.0).astype(np.float32
                                                     ).reshape(1, -1)
        return v


    def sample_attrs(self, attr_vec, seed=None):
        """Generate voxels conditioned on attribute vector (6-dim)."""
        assert self.attr_dim > 0, "model not trained with attributes"
        rng = np.random.default_rng(seed)
        v = np.zeros((G, G, G), dtype=np.int64)
        v[:, :, 0] = WALL
        v[0, :, 1:] = v[-1, :, 1:] = WALL
        v[:, 0, 1:] = v[:, -1, 1:] = WALL
        prev = np.where(v[:, :, 0].reshape(-1) > 0, 1.0, -1.0
                        ).astype(np.float32).reshape(1, -1)
        avec = np.asarray(attr_vec, dtype=np.float32)
        for z in range(1, G):
            xt = rng.standard_normal((1, self.SDIM)).astype(np.float32)
            s = self._denoise(xt, prev, avec, z, rng)
            sl = (s.reshape(G, G) > 0).astype(np.int64)
            sl[0, :] = sl[-1, :] = sl[:, 0] = sl[:, -1] = 1
            v[:, :, z] = sl
            prev = np.where(s > 0, 1.0, -1.0).astype(np.float32
                                                     ).reshape(1, -1)
        return v

    def save(self, path):
        arrays = {f"net.{k}": v
                  for k, v in self.net.parameters().items()}
        arrays["n_styles"] = np.array([self.n_styles])
        arrays["attr_dim"] = np.array([self.attr_dim])
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path, seed=0):
        d = np.load(path)
        attr_dim = int(d["attr_dim"][0]) if "attr_dim" in d else 0
        self = cls(n_styles=int(d["n_styles"][0]), seed=seed,
                   attr_dim=attr_dim)
        self.net.load_parameters(
            {k.split(".", 1)[1]: d[k] for k in d.files
             if k.startswith("net.")})
        return self


class HeightfieldPrior:
    """Diffusion over (footprint G*G + heightmap G*G) dims.

    A 2.5D representation: the footprint says WHERE walls are (our proven
    2D regime), the heightmap says HOW TALL. Reconstructs to G^3 voxels.
    """

    HDIM = 2 * G * G

    def __init__(self, n_styles=N_STYLES, seed=0):
        self.n_styles = n_styles
        self.net = MLP([self.HDIM + 16 + n_styles, 128, 128,
                        self.HDIM], seed=seed)

    @staticmethod
    def encode(voxels):
        """voxels (G,G,G) -> (512,) float32 in [-1,1]: [footprint, height]."""
        fp = (voxels[:, :, 1:].max(axis=2) > 0).astype(np.float32)
        h = np.zeros((G, G), dtype=np.float32)
        for x in range(G):
            for y in range(G):
                col = np.where(voxels[x, y, 1:] > 0)[0]
                h[x, y] = col.max() + 1 if len(col) else 0
        h = h / (G - 1) * 2 - 1
        return np.concatenate([(fp * 2 - 1).reshape(-1),
                               h.reshape(-1)]).astype(np.float32)

    @staticmethod
    def decode(hvec):
        """(2*G*G,) -> (G,G,G) int voxels."""
        gg = G * G
        fp = (hvec[:gg].reshape(G, G) > 0)
        h = np.round((hvec[gg:].reshape(G, G) + 1) / 2 * (G - 1)
                     ).astype(int).clip(0, G - 1)
        v = np.zeros((G, G, G), dtype=np.int64)
        v[:, :, 0] = WALL
        v[0, :, 1:] = v[-1, :, 1:] = WALL
        v[:, 0, 1:] = v[:, -1, 1:] = WALL
        for x in range(G):
            for y in range(G):
                if fp[x, y] and h[x, y] > 0:
                    v[x, y, 1:h[x, y] + 1] = WALL
        return v

    def _condition(self, t_idx, styles):
        B = len(t_idx)
        te = _time_embed(t_idx / T) * 2.0
        onehot = np.zeros((B, self.n_styles), dtype=np.float32)
        onehot[np.arange(B), styles] = 3.0
        return np.concatenate([te, onehot], axis=1)

    def _q_sample(self, x0, t_idx, rng):
        alpha = (1.0 - (t_idx / T) * 0.85).reshape(-1, 1)
        noise = rng.standard_normal(x0.shape).astype(np.float32)
        return (alpha * x0 + np.sqrt(1 - alpha ** 2) * noise).astype(
            np.float32)

    def train_step(self, x0, styles, opt, rng):
        B = len(x0)
        t_idx = rng.integers(1, T + 1, size=B)
        xt = self._q_sample(x0, t_idx, rng)
        cond = self._condition(t_idx, styles)
        pred = self.net.forward(np.concatenate([xt, cond], axis=1))
        err = pred - x0
        loss = float(np.mean(err ** 2))
        self.net.backward((2.0 * err / err.size).astype(np.float32))
        opt.step()
        return loss

    def fit(self, hvecs, styles, steps=2500, batch=64, seed=0,
            log_every=500):
        rng = np.random.default_rng(seed)
        opt = Adam(self.net.params_and_grads(), lr=1e-3)
        N = len(hvecs)
        t0 = time.time()
        for step in range(1, steps + 1):
            idx = rng.integers(N, size=batch)
            loss = self.train_step(hvecs[idx], styles[idx], opt, rng)
            if log_every and step % log_every == 0:
                print(f"heightfield step {step:5d}/{steps} "
                      f"loss={loss:.4f} ({time.time()-t0:.0f}s)", flush=True)
        return self

    def sample(self, style=0, n=1, seed=None, steps=T):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n, self.HDIM)).astype(np.float32)
        styles = np.full(n, int(style) % self.n_styles)
        for ti in range(steps, 0, -1):
            t_idx = np.full(n, ti)
            cond = self._condition(t_idx, styles)
            x0_hat = self.net.forward(np.concatenate([x, cond], axis=1))
            if ti > 1:
                alpha = 1.0 - (ti / T) * 0.85
                x = (alpha * x0_hat + np.sqrt(max(1 - alpha ** 2, 0))
                     * rng.standard_normal((n, self.HDIM))
                     .astype(np.float32)).astype(np.float32)
            else:
                x = x0_hat
        return [self.decode(x[i]) for i in range(n)]

    def save(self, path):
        arrays = {f"net.{k}": v
                  for k, v in self.net.parameters().items()}
        arrays["n_styles"] = np.array([self.n_styles])
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path, seed=0):
        d = np.load(path)
        self = cls(n_styles=int(d["n_styles"][0]), seed=seed)
        self.net.load_parameters(
            {k.split(".", 1)[1]: d[k] for k in d.files
             if k.startswith("net.")})
        return self
