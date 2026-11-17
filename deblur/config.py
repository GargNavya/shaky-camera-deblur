from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_OWN_DIR = DATA_DIR / "raw_own"
SHARP_DIR = DATA_DIR / "sharp"
RESULTS_DIR = ROOT / "results"

IMAGE_SIZE = 512
SKIMAGE_SAMPLES = ("camera", "astronaut", "coffee")

KERNEL_SIZE = 31


def results_dir(step):
    p = RESULTS_DIR / step
    p.mkdir(parents=True, exist_ok=True)
    return p
