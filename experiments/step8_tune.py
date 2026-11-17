import json

import numpy as np

from experiments.step8_common import (BRANCH, apply, dev_images, family_kernels, make_cases, rl_iters,
                                      rl_with_checkpoints, run_cases, scores)
from deblur.degrade import make_blurred
from deblur.direction import estimate_line_kernel
from deblur.kernels import line_kernel
from deblur.metrics import region_masks
from deblur.tables import write_csv
from deblur.tuning import step3_params

_W = {}


def _init(cfg):
    _W["cfg"], _W["images"] = cfg, dict(dev_images(cfg))
    _W["kernels"] = {kid: k for _, kid, k in family_kernels(cfg["dev_kernels"], "dev")}
    _W["params"] = step3_params()


def _case(c):
    cfg, params = _W["cfg"], _W["params"]
    x, kt = _W["images"][c["image"]], _W["kernels"][c["kernel_id"]]
    y = make_blurred(x, kt, c["sigma"], np.random.default_rng(c["seed"]))
    th, L = estimate_line_kernel(y)
    k = line_kernel(th, L)
    masks = region_masks(x)
    rl_out = rl_with_checkpoints(y, k, rl_iters(c["sigma"], params))
    rows = [dict(c, method="rl", settings="{}", **scores(x, rl_out[max(rl_out)], masks))]
    for m, grid in cfg["grids"].items():
        for s in grid:
            rows.append(dict(c, method=m, settings=json.dumps(s, sort_keys=True), **scores(x, apply(m, y, k, s, rl_out), masks)))
    return rows


def run(cfg, root, workers, log, smoke=False):
    out = root / "tune"
    out.mkdir(parents=True, exist_ok=True)
    cases = make_cases(dev_images(cfg), family_kernels(cfg["dev_kernels"], "dev"), cfg["sigmas"], "dev")
    rows = run_cases(_case, cases, out / "cases", workers, log, init=_init, initargs=(cfg,))
    write_csv(out / "tuning_raw.csv", rows)
    summary, frozen = [], {"per_sigma": {}}
    for s in cfg["sigmas"]:
        frozen["per_sigma"][str(s)] = {}
        for m in cfg["grids"]:
            groups = {}
            for r in rows:
                if r["method"] == m and r["sigma"] == s:
                    groups.setdefault(r["settings"], []).append(r["psnr"])
            for st, v in groups.items():
                summary.append(dict(sigma=s, method=m, settings=st, mean_psnr=np.mean(v), n=len(v)))
            best = max(groups.items(), key=lambda kv: np.mean(kv[1]))
            frozen["per_sigma"][str(s)][m] = json.loads(best[0])
        base = np.mean([r["psnr"] for r in rows if r["method"] == "rl" and r["sigma"] == s])
        summary.append(dict(sigma=s, method="rl", settings="{}", mean_psnr=base, n=sum(1 for r in rows if r["method"] == "rl" and r["sigma"] == s)))
    write_csv(out / "tuning_summary.csv", summary)
    # primary method per branch: best mean dev PSNR (each at its frozen per-sigma setting)
    def frozen_mean(m):
        v = [r["psnr"] for r in rows if r["method"] == m
             and r["settings"] == json.dumps(frozen["per_sigma"][str(r["sigma"])][m], sort_keys=True)]
        return float(np.mean(v))
    frozen["dev_mean_psnr"] = {m: frozen_mean(m) for m in cfg["grids"]}
    frozen["dev_mean_psnr"]["rl"] = float(np.mean([r["psnr"] for r in rows if r["method"] == "rl"]))
    frozen["primary"] = {b: max(ms, key=frozen_mean) for b, ms in BRANCH.items()}
    frozen["_meta"] = dict(objective="mean shift-aligned PSNR with the estimated kernel, dev images, per noise level",
                           dev_images=cfg["dev_images"])
    (out / "frozen_settings.json").write_text(json.dumps(frozen, indent=2))
    for m, v in frozen["dev_mean_psnr"].items():
        log.info(f"dev mean PSNR {m:7s} {v:.3f}")
    log.info(f"primary methods: {frozen['primary']}")
    for s, d in frozen["per_sigma"].items():
        log.info(f"sigma {s}: {d}")
    return f"primary {frozen['primary']}"
