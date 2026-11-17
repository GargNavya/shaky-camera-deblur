import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # parallelism comes from worker processes

import json
import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from skimage import data as skdata

from deblur import config
from deblur.data import load_sharp, to_gray_square
from deblur.deconv import richardson_lucy
from deblur.kernels import curved_kernel, line_kernel, trajectory_kernel
from deblur.metrics import mse_aligned, psnr_from_mse

# ---------------------------------------------------------------- protocol constants
FULL = dict(
    pca_train_n=5000, pca_train_sizes=[500, 2000, 5000], pca_kmax=60,
    capacity_K=[1, 2, 5, 10, 20, 40, 60],
    dev_images=["chelsea", "brick", "immunohistochemistry", "coins"],
    dev_sigmas=[0.005, 0.01, 0.02],
    dev_kernels=dict(straight=[(35, 12), (100, 18), (160, 22)], curved=4, traj=3),
    tune_proxy_nsr=[3e-3, 1e-2, 3e-2], tune_keep=[0.05, 0.2, 1.0],
    tune_free_beta=[1e-3, 1e-2, 1e-1], tune_pca_K=[5, 10, 20], tune_pca_ridge=[1e-3, 1e-2, 1e-1],
    test_sigmas=[0.005, 0.01, 0.02],
    test_kernels=dict(straight=[(t, L) for t in (20, 75, 140) for L in (10, 15, 25)], curved=12, traj=12),
    ladder_proxies=["oracle", "oracle+noise", "oracle+blur"],
    proxy_noise=0.02, proxy_blur=1.0,
    rounds=8, rounds_scored=[1, 2, 4, 8],
    ablation_K=[2, 5, 10, 20, 40], init_offsets=[5.0, 10.0],
)
SMOKE = dict(FULL,
    pca_train_n=500, pca_train_sizes=[200, 500], pca_kmax=30, capacity_K=[2, 10, 30],
    dev_images=["chelsea"], dev_sigmas=[0.01],
    dev_kernels=dict(straight=[(35, 12)], curved=1, traj=1),
    tune_proxy_nsr=[1e-2], tune_keep=[0.2, 1.0], tune_free_beta=[1e-2], tune_pca_K=[5, 10], tune_pca_ridge=[1e-2],
    test_sigmas=[0.01], test_kernels=dict(straight=[(20, 15)], curved=1, traj=1),
    rounds=2, rounds_scored=[1, 2], ablation_K=[2, 5], init_offsets=[5.0],
)
MODELS = ["free", "pca", "pca_unaligned", "straight"]
FAMILIES = ["straight", "curved", "traj", "real"]
CURVED_LENGTH = (10.0, 22.0)     # test/dev curved kernels: length and bend ranges
CURVED_BEND = (30.0, 130.0)
TRAJ_LENGTH = (10.0, 25.0)
SEED_BASE = dict(dev=200_000, test=300_000)   # PCA training uses deblur.pca.TRAIN_SEED
REAL_KERNEL_DIR = config.DATA_DIR / "kernels_real"
PHASES = ["6a", "6b", "6c", "6d"]


def cfg_for(smoke):
    return SMOKE if smoke else FULL


def out_root(smoke):
    return config.RESULTS_DIR / ("step6_smoke" if smoke else "step6")


def phase_dir(root, phase):
    name = {"6a": "6a_pca", "6b": "6b_tune", "6c": "6c_test", "6d": "6d_analysis"}[phase]
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- logging
def get_logger(root, phase):
    (root / "logs").mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(f"{root.name}.{phase}")
    log.handlers.clear()
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.FileHandler(root / "logs" / f"{phase}.log"), logging.StreamHandler()):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.propagate = False
    return log


# ---------------------------------------------------------------- data
def dev_images(cfg):
    out = []
    for n in cfg["dev_images"]:
        out.append((f"dev_{n}", to_gray_square(getattr(skdata, n)(), config.IMAGE_SIZE)))
    return out


def real_kernels():
    """Optional measured kernels: data/kernels_real/*.png or *.npy, at most 31 x 31."""
    if not REAL_KERNEL_DIR.exists():
        return []
    from PIL import Image
    ks = config.KERNEL_SIZE
    out = []
    for p in sorted(REAL_KERNEL_DIR.iterdir()):
        if p.suffix.lower() == ".npy":
            k = np.load(p).astype(float)
        elif p.suffix.lower() in (".png", ".bmp", ".tif", ".tiff"):
            k = np.asarray(Image.open(p).convert("L"), float)
        else:
            continue
        if k.shape[0] > ks or k.shape[1] > ks:
            continue
        big = np.zeros((ks, ks))
        oy, ox = (ks - k.shape[0]) // 2, (ks - k.shape[1]) // 2
        big[oy:oy + k.shape[0], ox:ox + k.shape[1]] = k
        out.append((p.stem, big / big.sum()))
    return out


def family_kernels(spec, split):
    """[(family, kernel_id, kernel)] for a kernel spec dict and split ('dev' or 'test')."""
    base = SEED_BASE[split]
    out = [("straight", f"line_t{t}_L{L}", line_kernel(t, L)) for t, L in spec["straight"]]
    for i in range(spec["curved"]):
        rng = np.random.default_rng(base + 1000 + i)
        out.append(("curved", f"curved_{i}", curved_kernel(rng.uniform(*CURVED_LENGTH), rng.uniform(*CURVED_BEND), rng)))
    for i in range(spec["traj"]):
        rng = np.random.default_rng(base + 2000 + i)
        out.append(("traj", f"traj_{i}", trajectory_kernel(rng.uniform(*TRAJ_LENGTH), rng)))
    if split == "test":
        out += [("real", f"real_{n}", k) for n, k in real_kernels()]
    return out


def make_cases(images, kernels, sigmas, split):
    cases = []
    for ii, (iname, _) in enumerate(images):
        for ki, (fam, kid, _) in enumerate(kernels):
            for si, s in enumerate(sigmas):
                cases.append(dict(case_id=f"{iname}__{kid}__s{s:g}", image=iname, family=fam, kernel_id=kid,
                                  sigma=s, seed=SEED_BASE[split] + 100_000 + 10_000 * ii + 10 * ki + si))
    return cases


# ---------------------------------------------------------------- scoring
def restore_score(x, y, k, sigma, params):
    """RL with the step 3 iteration count for this noise level; shift-aligned PSNR and MSE."""
    _, it = params[sigma]
    m = mse_aligned(x, richardson_lucy(y, k, it))
    return psnr_from_mse(m), m


# ---------------------------------------------------------------- parallel, resumable runner
def run_cases(fn, cases, case_dir, workers, log, init=None, initargs=()):
    """Run fn(case) -> list[dict] for every case without a saved result; save one JSON per case.
    Returns all rows (saved earlier + new)."""
    case_dir.mkdir(parents=True, exist_ok=True)
    todo = [c for c in cases if not (case_dir / f"{c['case_id']}.json").exists()]
    log.info(f"{len(cases)} cases, {len(cases) - len(todo)} already done, {len(todo)} to run, {workers} workers")
    t0, done = time.time(), 0
    if todo:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=init, initargs=initargs) as ex:
            futs = {ex.submit(fn, c): c for c in todo}
            for f in as_completed(futs):
                c = futs[f]
                rows = f.result()          # a worker error stops the phase (logged by the queue)
                tmp = case_dir / f"{c['case_id']}.json.tmp"
                tmp.write_text(json.dumps(rows))
                tmp.replace(case_dir / f"{c['case_id']}.json")
                done += 1
                if done % max(1, len(todo) // 50) == 0 or done == len(todo):
                    el = time.time() - t0
                    log.info(f"  {done}/{len(todo)} cases, {el / 60:.1f} min, ETA {el / done * (len(todo) - done) / 60:.1f} min")
    rows = []
    for c in cases:
        rows += json.loads((case_dir / f"{c['case_id']}.json").read_text())
    return rows


def default_workers():
    return max(1, (os.cpu_count() or 2) - 1)


def mark_done(root, phase, info=""):
    (root / ".done").mkdir(exist_ok=True)
    (root / ".done" / phase).write_text(time.strftime("%Y-%m-%d %H:%M:%S") + " " + info + "\n")


def is_done(root, phase):
    return (root / ".done" / phase).exists()


def load_frozen(root):
    p = phase_dir(root, "6b") / "frozen_settings.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run phase 6b first")
    return json.loads(p.read_text())


def load_test_images(smoke):
    """Test images: data/sharp (the 8 project images). Smoke run: two dev images instead."""
    if smoke:
        return [(f"smoke_{n}", to_gray_square(getattr(skdata, n)(), config.IMAGE_SIZE)) for n in ("chelsea", "brick")]
    return load_sharp()
