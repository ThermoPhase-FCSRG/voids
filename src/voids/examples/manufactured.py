from __future__ import annotations

from pathlib import Path

import numpy as np


def make_manufactured_void_image(shape: tuple[int, int, int] = (48, 48, 48)) -> np.ndarray:
    """Create a deterministic synthetic porous image for PoreSpy extraction examples.

    Returns a boolean 3D image where ``True`` denotes void space. The geometry consists of a
    chain of overlapping spherical voids spanning the x-direction plus a few side branches.
    This is *manufactured* (synthetic) but intended to be realistic enough for a PoreSpy
    ``snow2`` extraction demo.
    """
    nx, ny, nz = shape
    X, Y, Z = np.indices(shape)
    im = np.zeros(shape, dtype=bool)

    # Main spanning chain (centers and radii in voxel units)
    chain = [
        (6, ny // 2, nz // 2, 7),
        (14, ny // 2 + 1, nz // 2, 7),
        (22, ny // 2 - 1, nz // 2 + 1, 7),
        (30, ny // 2, nz // 2 - 1, 7),
        (38, ny // 2 + 1, nz // 2, 7),
    ]
    branches = [
        (20, ny // 2 + 10, nz // 2, 5),
        (28, ny // 2 - 10, nz // 2 + 2, 5),
        (34, ny // 2 + 6, nz // 2 + 8, 4),
    ]
    for cx, cy, cz, r in chain + branches:
        mask = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 <= r**2
        im |= mask

    # Add a thin connecting tunnel to reduce segmentation ambiguity between two chain pores
    y0 = ny // 2
    z0 = nz // 2
    im[12:17, y0 - 1 : y0 + 2, z0 - 1 : z0 + 2] = True

    return im


def save_default_manufactured_void_image(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, make_manufactured_void_image())
    return path
