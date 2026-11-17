import itertools
import json

import numpy as np

from experiments.step6_common import (MODELS, dev_images, family_kernels, make_cases, phase_dir, run_cases)
from deblur.degrade import make_blurred
from deblur.direction import estimate_line_kernel
from deblur.kernels import kernel_similarity, line_kernel
from deblur.kstep import GradSystem, kernel_step, make_proxy
from deblur.pca import KernelBasis
from deblur.tables import write_csv

_W = {}


def _init(root, cfg):
    from experiments.step6_common import phase_dir as pd
    _W["cfg"] = cfg
    _W["images"] = dict(dev_images(cfg))
    _W["kernels"] = {kid: k for _, kid, k in family_kernels(cfg["dev_kernels"], "dev")}
    _W["Ba"] = KernelBasis.load(pd(root, "6a") / "basis_aligned.npz")
    _W["Bu"] = KernelBasis.load(pd(root, "6a") / "basis_unaligned.npz")


def model_grid(cfg, model):
    if model == "free":
        return [dict(free_beta=b) for b in cfg["tune_free_beta"]]
    if model in ("pca", "pca_unaligned"):
        return [dict(pca_K=K, pca_ridge=r) for K, r in itertools.product(cfg["tune_pca_K"], cfg["tune_pca_ridge"])]
    return [dict()]


def _case(c):
    cfg = _W["cfg"]
    x, kt = _W["images"][c["image"]], _W["kernels"][c["kernel_id"]]
    y = make_blurred(x, kt, c["sigma"], np.random.default_rng(c["seed"]))
    th, L = estimate_line_kernel(y)
    k0 = line_kernel(th, L)
    rows = []
    for nsr in cfg["tune_proxy_nsr"]:
        proxy = make_proxy(y, k0, nsr)
        for keep in cfg["tune_keep"]:
            sysm = GradSystem(y, proxy, keep)
            for model in MODELS:
                for p in model_grid(cfg, model):
                    k, _, _ = kernel_step(model, sysm, th, L, p, _W["Ba"], _W["Bu"])
                    rows.append(dict(c, model=model, proxy_nsr=nsr, keep=keep, params=json.dumps(p, sort_keys=True),
                                     similarity=kernel_similarity(k, kt), init_similarity=kernel_similarity(k0, kt)))
    return rows


def run(cfg, root, workers, log):
    out = phase_dir(root, "6b")
    images = dev_images(cfg)
    kernels = family_kernels(cfg["dev_kernels"], "dev")
    cases = make_cases(images, kernels, cfg["dev_sigmas"], "dev")
    rows = run_cases(_case, cases, out / "cases", workers, log, init=_init, initargs=(root, cfg))
    write_csv(out / "tuning_raw.csv", rows)
    summary, frozen = [], {}
    for model in MODELS:
        configs = {}
        for r in rows:
            if r["model"] == model:
                configs.setdefault((r["proxy_nsr"], r["keep"], r["params"]), []).append(r["similarity"])
        for (nsr, keep, p), v in configs.items():
            summary.append(dict(model=model, proxy_nsr=nsr, keep=keep, params=p, mean_similarity=np.mean(v), n=len(v)))
        (nsr, keep, p), v = max(configs.items(), key=lambda kv: np.mean(kv[1]))
        frozen[model] = dict(proxy_nsr=nsr, keep=keep, **json.loads(p), dev_mean_similarity=float(np.mean(v)))
        log.info(f"frozen {model:14s}: proxy_nsr={nsr:g} keep={keep:g} {p}  dev similarity {np.mean(v):.3f}")
    init = np.mean([r["init_similarity"] for r in rows if r["model"] == "straight"])
    log.info(f"dev similarity of the straight starting estimate (no refinement): {init:.3f}")
    write_csv(out / "tuning_summary.csv", sorted(summary, key=lambda r: (r["model"], -r["mean_similarity"])))
    frozen["_meta"] = dict(objective="mean kernel similarity after one blind round, dev images only",
                           dev_images=cfg["dev_images"], dev_init_similarity=float(init))
    (out / "frozen_settings.json").write_text(json.dumps(frozen, indent=2))
    return "settings frozen"
