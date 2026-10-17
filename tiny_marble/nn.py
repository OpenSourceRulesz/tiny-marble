"""Tiny pure-numpy neural network building blocks.

Zero dependencies beyond numpy. MLPs with ReLU hidden layers and a linear
output layer, He initialization, and a minimal Adam optimizer.
"""

import numpy as np


def relu(x):
    return np.maximum(x, 0.0)


def relu_backward(g, x):
    return g * (x > 0)


class Linear:
    def __init__(self, in_dim, out_dim, rng):
        self.W = (rng.standard_normal((in_dim, out_dim))
                  * np.sqrt(2.0 / in_dim)).astype(np.float32)
        self.b = np.zeros(out_dim, dtype=np.float32)
        self.gW = np.zeros_like(self.W)
        self.gb = np.zeros_like(self.b)
        self.x = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, g):
        # NOTE: grads are written in-place so an Adam holding (param, grad)
        # references always sees the latest values.
        gx = g @ self.W.T
        self.gW[:] = self.x.T @ g
        self.gb[:] = g.sum(axis=0)
        return gx


class MLP:
    """sizes=[in, h1, ..., out]; ReLU on hidden layers, linear output."""

    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.layers = [Linear(a, b, rng) for a, b in zip(sizes[:-1], sizes[1:])]
        self._cache = []

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        self._cache = []
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            pre = x
            if i < len(self.layers) - 1:
                x = relu(x)
            self._cache.append(pre)
        return x

    def backward(self, g):
        """Backprop; returns grad w.r.t. the MLP input. Gradients for each
        Linear layer are stored on the layer as .gW / .gb."""
        for i in reversed(range(len(self.layers))):
            if i < len(self.layers) - 1:
                g = relu_backward(g, self._cache[i])
            g = self.layers[i].backward(g)
        return g

    def params_and_grads(self):
        out = []
        for l in self.layers:
            out.append((l.W, l.gW))
            out.append((l.b, l.gb))
        return out

    def parameters(self):
        """Flat dict {f"{i}.W": ..., f"{i}.b": ...} for save/load."""
        p = {}
        for i, l in enumerate(self.layers):
            p[f"{i}.W"] = l.W.copy()
            p[f"{i}.b"] = l.b.copy()
        return p

    def load_parameters(self, p):
        for i, l in enumerate(self.layers):
            l.W[:] = np.asarray(p[f"{i}.W"], dtype=np.float32)
            l.b[:] = np.asarray(p[f"{i}.b"], dtype=np.float32)


class Adam:
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8):
        # params: list of (array, grad_array) tuples
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p) for p, _ in params]
        self.v = [np.zeros_like(p) for p, _ in params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, (p, g) in enumerate(self.params):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)
