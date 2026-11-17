import json

import numpy as np

from experiments.step8_common import (apply, family_kernels, load_test_images, make_cases, rl_iters,
                                      rl_with_checkpoints, run_cases, scores)
from deblur.degrade import make_blurred
from deblur.direction import estimate_line_kernel
from deblur.kernels import line_kernel
from deblur.metrics import region_masks
from deblur.tables import write_csv
from deblur.tuning import step3_params

METHODS = ["rl", "notch", "blend", "guided", "tv", "hl"]
_W = {}


def _init(root, cfg, smoke):
    _W["cfg"], _W["images"] = cfg, dict(load_test_images(smoke))
    _W["kernels"] = {kid: k for _, kid, k in family_kernels(cfg["test_kernels"], "test")}
    _W["params"] = step3_params()
    _W["frozen"] = json.loads((root / "tune" / "frozen_settings.json").read_text())


def _case(c):
    x, kt = _W["images"][c["image"]], _W["kernels"][c["kernel_id"]]
    y = make_blurred(x, kt, c["sigma"], np.random.default_rng(c["seed"]))
    th, L = estimate_line_kernel(y)
    masks = region_masks(x)
    fz = _W["frozen"]["per_sigma"][str(c["sigma"])]
    it = rl_iters(c["sigma"], _W["params"])
    base = dict(c, theta_est=th, L_est=L, rl_iters=it)
    rows = [dict(base, kernel="none", method="blurred", **scores(x, y, masks))]
    for kname, k in (("estimated", line_kernel(th, L)), ("true", kt)):
        rl_out = rl_with_checkpoints(y, k, it)
        for m in METHODS:
            s = fz.get(m, {})
            rows.append(dict(base, kernel=kname, method=m, settings=json.dumps(s, sort_keys=True),
                             **scores(x, apply(m, y, k, s, rl_out), masks)))
    return rows


def run(cfg, root, workers, log, smoke=False):
    out = root / "test"
    out.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((root / "tune" / "frozen_settings.json").read_text())
    images = load_test_images(smoke)
    kernels = family_kernels(cfg["test_kernels"], "test")
    cases = make_cases(images, kernels, cfg["sigmas"], "test")
    mpath = out / "manifest.json"
    if mpath.exists() and json.loads(mpath.read_text())["frozen"] != frozen:
        raise RuntimeError("frozen settings changed since this test run started; start a new step 8 folder")
    mpath.write_text(json.dumps(dict(images=[n for n, _ in images], kernels=[k[1] for k in kernels],
                                     sigmas=cfg["sigmas"], frozen=frozen, n_cases=len(cases)), indent=2))
    log.info(f"{len(images)} images, {len(kernels)} kernels, sigmas {cfg['sigmas']}: {len(cases)} cases")
    rows = run_cases(_case, cases, out / "cases", workers, log, init=_init, initargs=(root, cfg, smoke))
    keys = sorted({k for r in rows for k in r})
    write_csv(out / "results_long.csv", [{k: r.get(k, "") for k in keys} for r in rows])
    return f"{len(cases)} cases"
