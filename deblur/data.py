# Build and load the sharp image set (grayscale, square, float in [0, 1])
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import laplace
from skimage import color, transform
from skimage import data as skdata

from . import config

_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def to_gray_square(arr, size=config.IMAGE_SIZE):
    # Any image array -> grayscale, center-cropped square, resized, in [0, 1]
    arr = np.asarray(arr)
    if arr.dtype not in (np.float64, np.float32):
        arr = arr.astype(np.float64) / 255.0
    if arr.ndim == 3:
        arr = color.rgb2gray(arr[..., :3])
    h, w = arr.shape
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    arr = arr[top:top + s, left:left + s]
    arr = transform.resize(arr, (size, size), anti_aliasing=True)
    return np.clip(arr, 0, 1)


def to_color_square(arr, size=config.IMAGE_SIZE):
    arr = np.asarray(arr)
    if arr.dtype not in (np.float64, np.float32):
        arr = arr.astype(np.float64) / 255.0
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    arr = arr[..., :3]
    h, w = arr.shape[:2]
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    arr = transform.resize(arr[top:top + s, left:left + s], (size, size), anti_aliasing=True)
    return np.clip(arr, 0, 1)


def sharpness(img):
    # Higher = sharper
    return float(laplace(img).var())


def collect_sources():
    config.RAW_OWN_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for p in sorted(config.RAW_OWN_DIR.glob("*")):
        if p.suffix.lower() in _EXTS:
            im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            items.append((f"own_{p.stem}", to_gray_square(np.array(im))))
    for name in config.SKIMAGE_SAMPLES:
        items.append((f"sk_{name}", to_gray_square(getattr(skdata, name)())))
    return items


def save_sharp_set(items, folder=config.SHARP_DIR):
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.png"):
        old.unlink()
    for name, img in items:
        Image.fromarray((img * 255).round().astype(np.uint8)).save(folder / f"{name}.png")


def load_sharp(folder=config.SHARP_DIR):
    out = []
    for p in sorted(folder.glob("*.png")):
        out.append((p.stem, np.asarray(Image.open(p).convert("L"), dtype=np.float64) / 255.0))
    if not out:
        raise FileNotFoundError(f"No images in {folder}. Run experiments.step1_prepare_data first.")
    return out
