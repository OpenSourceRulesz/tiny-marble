# tiny-marble

**Prompt-to-world 3D.** Type words, walk through the world they describe.

```python
from tiny_marble import dream_from_prompt
world = dream_from_prompt("a dark cavern with lava")
world.step(forward=1.0)
frame = world.render()  # 256x256 first-person RGB
```

Like World Labs' Marble, tiny: 32³ voxels, Taichi-accelerated rendering
(256×256 at 60-100 FPS with soft shadows, water reflections, ambient
occlusion), pure Python. Numpy fallback (64×64) if Taichi isn't installed.

![showcase](examples/showcase.png)

## Install

PyPI release is pending — install from GitHub until then:

```bash
pip install git+https://github.com/OpenSourceRulesz/tiny-marble.git
pip install "tiny-marble[fast] @ git+https://github.com/OpenSourceRulesz/tiny-marble.git"  # + Taichi (256x256, 60-100 FPS)
```

Once 0.1.0 lands on PyPI, the short forms work: `pip install tiny-marble`
and `pip install tiny-marble[fast]`.

## How it works

Text → attributes → voxels. The prompt is parsed into a small attribute
vocabulary (style, lava, water, dark, pillars) — by keyword matching, or by
a local LLM via Ollama (`parse_prompt_llm()`; falls back to keywords when
Ollama isn't running). Phase 1 is procedural generation; Phase 2 is a trained
attribute-conditioned diffusion prior (`dream_from_prompt_neural()`), which
dreams novel geometry instead of assembling templates.

The renderer is a DDA raycaster with:
- Soft shadows (shadow rays toward lava pools)
- Water reflections (fresnel-blended)
- Voxel ambient occlusion
- Per-voxel texture noise
- Lava glow light grid

## Examples

```bash
python examples/prompt_demo.py "a dark cavern with lava"  # -> prompt_world.gif
python examples/showcase.py    # -> showcase.png (triptych)
python examples/walk4d_demo.py # -> walk4d.gif (explore + time reverse)
python examples/walk_interactive.py "a dark cavern with lava"  # real-time walk: WASD move, mouse/arrow look, Q/E fly, ESC quit (Taichi GUI)
```

## License

Apache-2.0. Inspired by World Labs' Marble and InSpatio's 4D world architecture.
Original implementation, no code copied.
