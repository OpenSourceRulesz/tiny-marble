"""tiny-Marble: prompt-to-world for tiny-4D.

Text -> attributes -> voxels. The attribute vector is the bridge:
words map to a small vocabulary of world-attributes (Marble-style),
and the generator conditions on attributes.

Two parsers:
- parse_prompt: fast keyword matching (no dependencies)
- parse_prompt_llm: Ollama LLM for natural language understanding

Two generators:
- dream_from_prompt: procedural (Phase 1, always works)
- dream_from_prompt_neural: diffusion dreamer (Phase 2, needs checkpoint)
"""

import re

import numpy as np

from .voxel import (G, VoxelWorld, procedural_voxel, dream_voxel_world,
                    EMPTY, WALL, LAVA, WATER)

__all__ = ["parse_prompt", "parse_prompt_llm", "dream_from_prompt",
           "dream_from_prompt_neural", "ATTRIBUTES", "attrs_to_vec"]

# attribute vocabulary: the words we understand -> world features
ATTRIBUTES = {
    "style": ["rooms", "cavern", "arena"],
    "lava": False,
    "water": False,
    "dark": False,
    "pillars": False,
}

_KEYWORDS = [
    # (pattern, attribute, value)
    (r"\b(dungeon|rooms?|castle|fortress|maze)\b", "style", "rooms"),
    (r"\b(cavern|cave|grotto|tunnel)s?\b", "style", "cavern"),
    (r"\b(arena|open|field|plaza|courtyard)\b", "style", "arena"),
    (r"\b(lava|magma|fire|burning|volcan)\w*\b", "lava", True),
    (r"\b(water|lake|pool|river|ocean|pond)\b", "water", True),
    (r"\b(dark|night|spooky|creepy|gloom|shadow)\w*\b", "dark", True),
    (r"\b(pillars?|columns?)\b", "pillars", True),
]


def parse_prompt(text):
    """'a dark cavern with lava' -> {style: 'cavern', lava: True, ...}."""
    attrs = {"style": "arena", "lava": False, "water": False,
             "dark": False, "pillars": False}
    low = text.lower()
    for pat, key, val in _KEYWORDS:
        if re.search(pat, low):
            attrs[key] = val
    return attrs


def _add_pools(v, rng, material, n_pools):
    """Scatter glowing pools on the floor (z=1), each with clear sky above
    so the pool is actually visible from the camera (cavern rock blobs
    would otherwise bury them under an overhang)."""
    placed = 0
    for _ in range(n_pools * 40):  # attempts; bail out gracefully
        if placed >= n_pools:
            break
        cx, cy = int(rng.integers(3, G - 3)), int(rng.integers(3, G - 3))
        if not (v[cx, cy, 2:22] == EMPTY).all():
            continue  # rock overhang above -- pool would be invisible
        r = int(rng.integers(2, 5))
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    x, y = cx + dx, cy + dy
                    if 1 <= x < G - 1 and 1 <= y < G - 1:
                        if v[x, y, 1] == EMPTY:
                            v[x, y, 1] = material
        placed += 1


def dream_from_prompt(text, seed=None):
    """'a dark cavern with lava' -> walkable VoxelWorld. Tiny Marble."""
    rng = np.random.default_rng(seed)
    attrs = parse_prompt(text)
    style_idx = {"rooms": 0, "cavern": 1, "arena": 2}[attrs["style"]]
    v, _ = procedural_voxel(style=style_idx,
                            seed=int(rng.integers(1e9)))
    if attrs["lava"]:
        _add_pools(v, rng, LAVA, int(rng.integers(2, 5)))
    if attrs["water"]:
        _add_pools(v, rng, WATER, int(rng.integers(2, 4)))
    if attrs["pillars"]:
        for _ in range(int(rng.integers(8, 16))):
            x, y = int(rng.integers(3, G - 3)), int(rng.integers(3, G - 3))
            h = int(rng.integers(8, 18))
            if (v[x, y, 1:4] == EMPTY).all():
                v[x, y, 1:1 + h] = 1  # WALL
    # clear a spawn pocket
    cx, cy = int(rng.integers(6, G - 6)), int(rng.integers(6, G - 6))
    v[cx - 4:cx + 5, cy - 4:cy + 5, 1:15] = EMPTY
    ambient = 0.7 if attrs["dark"] else 1.0
    world = VoxelWorld(v, spawn=(cx, cy),
                       seed=int(rng.integers(1e9)), ambient=ambient)
    world.prompt = text
    world.attrs = attrs
    return world


# ----------------------------------------------------------------------
# LLM parser (Ollama, local, no cloud)
# ----------------------------------------------------------------------

_LLM_SYSTEM = """You parse world descriptions into attributes. Reply with ONLY a JSON object, no other text.

Attributes:
- style: "rooms" (dungeon/castle/maze), "cavern" (cave/grotto), or "arena" (open field/plaza)
- lava: true if lava/magma/fire/volcanic mentioned
- water: true if water/lake/pool/river mentioned
- dark: true if dark/night/spooky/creepy mentioned
- pillars: true if pillars/columns mentioned

Example: "a dark cavern with lava" -> {"style": "cavern", "lava": true, "water": false, "dark": true, "pillars": false}
"""


def parse_prompt_llm(text, model="qwen2:1.5b", host=None):
    """Parse with a local Ollama LLM. Falls back to keywords on failure.

    Requires Ollama running (user has it on their machine).
    """
    import json
    import urllib.request

    host = host or "http://127.0.0.1:11434"
    try:
        req = urllib.request.Request(
            f"{host}/api/generate",
            data=json.dumps({
                "model": model,
                "system": _LLM_SYSTEM,
                "prompt": text,
                "stream": False,
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            out = json.loads(resp.read())["response"]
        # extract JSON (LLM might add chatter)
        start = out.find("{")
        end = out.rfind("}") + 1
        attrs = json.loads(out[start:end])
        # validate
        result = {"style": "arena", "lava": False, "water": False,
                  "dark": False, "pillars": False}
        if attrs.get("style") in ("rooms", "cavern", "arena"):
            result["style"] = attrs["style"]
        for k in ("lava", "water", "dark", "pillars"):
            result[k] = bool(attrs.get(k))
        return result
    except Exception:
        # Ollama not running or failed: fall back to keywords
        return parse_prompt(text)


# ----------------------------------------------------------------------
# Neural dreamer (Phase 2): attributes -> diffusion -> voxels
# ----------------------------------------------------------------------

# attribute vector: [style_onehot(3), lava, water, pillars] = 6 dims
ATTR_DIM = 6


def attrs_to_vec(attrs):
    """{'style': 'cavern', 'lava': True, ...} -> (6,) float32."""
    style_idx = {"rooms": 0, "cavern": 1, "arena": 2}[attrs["style"]]
    vec = np.zeros(ATTR_DIM, dtype=np.float32)
    vec[style_idx] = 3.0
    vec[3] = 3.0 if attrs.get("lava") else 0.0
    vec[4] = 3.0 if attrs.get("water") else 0.0
    vec[5] = 3.0 if attrs.get("pillars") else 0.0
    return vec


def dream_from_prompt_neural(text, seed=None, use_llm=False,
                             checkpoint=None):
    """Phase 2: text -> attributes -> DIFFUSION-dreamed voxels.

    Uses the attribute-conditioned SlicePrior (trained on procedural
    worlds). Genuinely dreams novel geometry, not assembled templates.
    """
    import os
    from .voxel_diffusion import SlicePrior

    attrs = parse_prompt_llm(text) if use_llm else parse_prompt(text)
    avec = attrs_to_vec(attrs)

    if checkpoint is None:
        checkpoint = os.path.join(os.path.dirname(__file__),
                                  "checkpoints", "attr_slice_v1.npz")
    model = SlicePrior.load(checkpoint)
    v = model.sample_attrs(avec, seed=seed)

    # add lava/water as materials (diffusion learns structure, not materials)
    rng = np.random.default_rng(seed)
    if attrs["lava"]:
        # find low flat areas for lava pools
        for _ in range(3):
            cx, cy = int(rng.integers(4, G-4)), int(rng.integers(4, G-4))
            if (v[cx-1:cx+2, cy-1:cy+2, 1] == EMPTY).all():
                v[cx, cy, 1] = LAVA
    if attrs["water"]:
        for _ in range(3):
            cx, cy = int(rng.integers(4, G-4)), int(rng.integers(4, G-4))
            if (v[cx-1:cx+2, cy-1:cy+2, 1] == EMPTY).all():
                v[cx, cy, 1] = WATER

    # clear spawn
    cx, cy = int(rng.integers(6, G-6)), int(rng.integers(6, G-6))
    v[cx-4:cx+5, cy-4:cy+5, 1:15] = EMPTY
    ambient = 0.7 if attrs["dark"] else 1.0
    world = VoxelWorld(v, spawn=(cx, cy),
                       seed=int(rng.integers(1e9)), ambient=ambient)
    world.prompt = text
    world.attrs = attrs
    world.neural = True
    return world
