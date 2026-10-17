"""Tests for tiny-Marble prompt-to-world (worldgen.prompt)."""

import numpy as np

from tiny_marble import (parse_prompt, dream_from_prompt,
                                LAVA, WATER)


def test_parse_prompt_keywords():
    a = parse_prompt("a dark cavern with lava")
    assert a["style"] == "cavern" and a["lava"] and a["dark"]
    assert not a["water"]
    b = parse_prompt("an open arena with water")
    assert b["style"] == "arena" and b["water"]
    c = parse_prompt("a dungeon with pillars")
    assert c["style"] == "rooms" and c["pillars"]
    d = parse_prompt("something completely unrecognized")
    assert d["style"] == "arena"  # sensible default


def test_dream_from_prompt_lava():
    w = dream_from_prompt("a dark cavern with lava", seed=0)
    assert (w.voxels == LAVA).sum() > 0, "no lava generated"
    assert w.ambient < 1.0, "dark not applied"
    assert w.attrs["lava"] and w.attrs["dark"]


def test_dream_from_prompt_water():
    w = dream_from_prompt("an arena with a lake", seed=1)
    assert (w.voxels == WATER).sum() > 0, "no water generated"


def test_prompt_world_walkable():
    w = dream_from_prompt("a dungeon", seed=2)
    p0 = w.pos.copy()
    for _ in range(20):
        w.step(forward=1.0, dyaw=0.1)
    assert np.linalg.norm(w.pos - p0) > 0.5, "spawn trapped"
    f = w.render()
    # taichi: 256x256, numpy fallback: 64x64
    assert f.shape[2] == 3 and f.shape[0] == f.shape[1]
    assert f.shape[0] in (64, 256)
    assert (f >= 0).all() and (f <= 1).all()
