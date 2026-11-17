"""Saving result images and versioned output folders (never overwrite earlier results)."""
import numpy as np
from PIL import Image


def new_run_dir(base):
    """First free folder among base, base_v2, base_v3, ... Earlier runs are never touched."""
    if not base.exists():
        return base
    v = 2
    while (base.parent / f"{base.name}_v{v}").exists():
        v += 1
    return base.parent / f"{base.name}_v{v}"


def save_png(img, path):
    """Gray (H x W) or colour (H x W x 3) float image in [0, 1] -> 8-bit PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)


def save_kernel_png(k, path, zoom=8):
    """Kernel scaled to its max, enlarged with nearest neighbour so it is visible."""
    big = np.kron(k / max(k.max(), 1e-12), np.ones((zoom, zoom)))
    save_png(big, path)
