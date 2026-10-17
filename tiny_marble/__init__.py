"""tiny-marble: prompt-to-world 3D voxel engine.

Type words -> walk through the 3D world they describe.
"a dark cavern with lava" -> a dark cavern with lava, in first person.

Like World Labs' Marble, tiny: 32^3 voxels, Taichi-accelerated rendering
(256x256 at 60-100 FPS), pure Python.
"""

from .voxel import (VoxelWorld, dream_voxel_world, procedural_voxel,
                    raycast, G, RES, EMPTY, WALL, ACCENT, WISP, LAVA, WATER,
                    STYLES)
from .prompt import parse_prompt, dream_from_prompt, ATTRIBUTES
from .voxel_diffusion import VoxelPrior, HeightfieldPrior, SlicePrior

__all__ = [
    "VoxelWorld", "dream_voxel_world", "procedural_voxel", "raycast",
    "G", "RES", "EMPTY", "WALL", "ACCENT", "WISP", "LAVA", "WATER",
    "STYLES", "parse_prompt", "dream_from_prompt", "ATTRIBUTES",
    "VoxelPrior", "HeightfieldPrior", "SlicePrior",
]

# taichi renderer is optional (pip install tiny-marble[fast])
try:
    from .taichi_render import render as taichi_render, is_available
    __all__ += ["taichi_render", "is_available"]
except ImportError:
    pass

__version__ = "0.1.0"
