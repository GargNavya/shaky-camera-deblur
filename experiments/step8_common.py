import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import itertools

import numpy as np

from experiments.step6_common import (FULL as S6, SMOKE as S6_SMOKE, default_workers, dev_images, family_kernels,
                                      get_logger, is_done, load_test_images, make_cases, mark_done, run_cases)
from deblur import config
from deblur.deconv import prior_deconv, richardson_lucy
from deblur.metrics import psnr_regions, region_masks
from deblur.noise import nearest_level
from deblur.postproc import guided_smooth, mask_blend, notch_filter

SIGMAS = [0.001, 0.005, 0.01, 0.02]      # 0.001 ~ what real photos read as (step 7b)
GRIDS = dict(
    notch=[dict(eps=e) for e in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)],
    blend=[dict(gentle_iters=g, tau=t) for g, t in itertools.product((5, 10, 20), (0.5, 1, 2, 4))],
    guided=[dict(r=r, eps=e) for r, e in itertools.product((1, 2, 4), (1e-4, 3e-4, 1e-3, 3e-3, 1e-2))],
    tv=[dict(lam=l) for l in (30, 100, 300, 1000, 3000, 10000)],
    hl=[dict(lam=l) for l in (30, 100, 300, 1000, 3000, 10000)],
)
BRANCH = {"8a": ["notch", "blend", "guided"], "8b": ["tv", "hl"]}
LABEL = {"rl": "RL", "notch": "RL + notch", "blend": "RL + blend", "guided": "RL + guided",
         "tv": "TV deconv", "hl": "HL deconv", "blurred": "blurred",
         "rl_true": "RL, true kernel", "tv_true": "TV, true kernel", "hl_true": "HL, true kernel"}
PHASES = ["8tune", "8test", "8analyze", "8real"]


def cfg_for(smoke):
    if not smoke:
        return dict(dev_images=S6["dev_images"], dev_kernels=S6["dev_kernels"], test_kernels=S6["test_kernels"],
                    sigmas=SIGMAS, grids=GRIDS)
    small = {m: g[:2] for m, g in GRIDS.items()}
    return dict(dev_images=S6_SMOKE["dev_images"], dev_kernels=S6_SMOKE["dev_kernels"],
                test_kernels=S6_SMOKE["test_kernels"], sigmas=[0.001, 0.01], grids=small)


def out_root(smoke):
    return config.RESULTS_DIR / ("step8_smoke" if smoke else "step8")


def rl_iters(sigma, params):
    """Step 3 rule: RL iterations of the nearest tuned noise level."""
    return params[nearest_level(sigma, params.keys())][1]


def apply(method, y, k, s, rl_out):
    """One method with settings s. rl_out: {iterations: RL image} (main count + gentle checkpoints)."""
    main = max(rl_out)
    if method == "rl":
        return rl_out[main]
    if method == "notch":
        return notch_filter(rl_out[main], k, s["eps"])
    if method == "blend":
        return mask_blend(rl_out[main], rl_out[s["gentle_iters"]], y, s["tau"])
    if method == "guided":
        return guided_smooth(rl_out[main], s["r"], s["eps"])
    if method == "tv":
        return prior_deconv(y, k, s["lam"], alpha=1.0)
    if method == "hl":
        return prior_deconv(y, k, s["lam"], alpha=2 / 3)
    raise ValueError(method)


def rl_with_checkpoints(y, k, it, gentle=(5, 10, 20)):
    gs = sorted({g for g in gentle if g < it} | {it})
    out = richardson_lucy(y, k, it, checkpoints=gs)
    for g in gentle:                     # gentle >= main count: use the main result
        out.setdefault(g, out[it])
    return out


def scores(x, xh, masks):
    return psnr_regions(x, xh, masks)
