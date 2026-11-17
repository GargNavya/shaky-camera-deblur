import json

import numpy as np
from scipy.ndimage import gaussian_filter

from experiments.step6_common import (MODELS, family_kernels, load_frozen, load_test_images, make_cases,
                                      phase_dir, restore_score, run_cases)
from deblur.degrade import make_blurred
from deblur.direction import estimate_line_kernel
from deblur.kernels import kernel_angle_deg, kernel_similarity, line_kernel
from deblur.kstep import GradSystem, kernel_step, make_proxy
from deblur.metrics import mse_aligned, psnr_from_mse
from deblur.pca import KernelBasis
from deblur.tuning import step3_params

_W = {}


def _init(root, cfg, smoke):
    _W["cfg"] = cfg
    _W["images"] = dict(load_test_images(smoke))
    _W["kernels"] = {kid: k for _, kid, k in family_kernels(cfg["test_kernels"], "test")}
    _W["Ba"] = KernelBasis.load(phase_dir(root, "6a") / "basis_aligned.npz")
    _W["Bu"] = KernelBasis.load(phase_dir(root, "6a") / "basis_unaligned.npz")
    _W["frozen"] = load_frozen(root)
    _W["params"] = step3_params()


def _case(c):
    cfg, fz, params = _W["cfg"], _W["frozen"], _W["params"]
    Ba, Bu = _W["Ba"], _W["Bu"]
    x, kt = _W["images"][c["image"]], _W["kernels"][c["kernel_id"]]
    rng = np.random.default_rng(c["seed"])
    y = make_blurred(x, kt, c["sigma"], rng)
    th0, L0 = estimate_line_kernel(y)
    k0 = line_kernel(th0, L0)
    base = dict(c, theta_est=th0, L_est=L0, true_angle=kernel_angle_deg(kt))
    rows = []

    def add(exp, method, k, **extra):
        p, m = restore_score(x, y, k, c["sigma"], params)
        rows.append(dict(base, experiment=exp, method=method, similarity=kernel_similarity(k, kt),
                         psnr=p, mse=m, **extra))

    mb = mse_aligned(x, y)
    rows.append(dict(base, experiment="ref", method="blurred", similarity=np.nan, psnr=psnr_from_mse(mb), mse=mb))
    add("ref", "true kernel", kt)
    add("ref", "start (straight estimate)", k0)
    add("ref", "pca projection of truth", Ba.project(kt, fz["pca"]["pca_K"]))

    def one_step(model, proxy, th, L, settings=None):
        s = settings or fz[model]
        return kernel_step(model, GradSystem(y, proxy, s["keep"]), th, L, s, Ba, Bu)[0]

    # ladder: idealized proxies, one kernel step from the straight estimate
    noisy = x + np.random.default_rng(c["seed"] + 1).normal(0, cfg["proxy_noise"], x.shape)
    proxies = {"oracle": x, "oracle+noise": noisy, "oracle+blur": gaussian_filter(x, cfg["proxy_blur"])}
    for pname in cfg["ladder_proxies"]:
        for model in MODELS:
            add("ladder", model, one_step(model, proxies[pname], th0, L0), proxy=pname)

    # blind rounds from the straight estimate
    for model in MODELS:
        k, th, L = k0, th0, L0
        for r in range(1, cfg["rounds"] + 1):
            s = fz[model]
            k, th, L = kernel_step(model, GradSystem(y, make_proxy(y, k, s["proxy_nsr"]), s["keep"]), th, L, s, Ba, Bu)
            if model != "straight":
                th = kernel_angle_deg(k) if model == "pca" else th
            if r in cfg["rounds_scored"]:
                add("rounds", model, k, round=r)
            else:
                rows.append(dict(base, experiment="rounds", method=model, round=r,
                                 similarity=kernel_similarity(k, kt), psnr=np.nan, mse=np.nan))

    # K ablation (rotated PCA, one blind round)
    s = fz["pca"]
    proxy0 = make_proxy(y, k0, s["proxy_nsr"])
    for K in cfg["ablation_K"]:
        if K != s["pca_K"]:
            add("ablK", "pca", one_step("pca", proxy0, th0, L0, dict(s, pca_K=K)), K=K)

    # starting point: worse angle, or the true kernel
    for off in cfg["init_offsets"]:
        ks = line_kernel((th0 + off) % 180, L0)
        for model in ("free", "pca", "straight"):
            add("init", model, one_step(model, make_proxy(y, ks, fz[model]["proxy_nsr"]), (th0 + off) % 180, L0),
                start=f"angle+{off:g}")
    ta = kernel_angle_deg(kt)
    for model in ("free", "pca"):
        add("init", model, one_step(model, make_proxy(y, kt, fz[model]["proxy_nsr"]), ta, L0), start="true kernel")
    return rows


def run(cfg, root, workers, log, smoke=False):
    from deblur.tables import write_csv
    out = phase_dir(root, "6c")
    frozen = load_frozen(root)
    images = load_test_images(smoke)
    kernels = family_kernels(cfg["test_kernels"], "test")
    cases = make_cases(images, kernels, cfg["test_sigmas"], "test")
    fams = {f: sum(1 for k in kernels if k[0] == f) for f in ("straight", "curved", "traj", "real")}
    log.info(f"{len(images)} images, kernels per family {fams}, sigmas {cfg['test_sigmas']}")
    if fams["real"] == 0:
        log.info("no measured kernels in data/kernels_real/ -> 'real' family skipped")
    manifest = dict(images=[n for n, _ in images], kernels=[k[1] for k in kernels], sigmas=cfg["test_sigmas"],
                    frozen=frozen, n_cases=len(cases))
    mpath = out / "manifest.json"
    if mpath.exists() and json.loads(mpath.read_text()).get("frozen") != json.loads(json.dumps(frozen)):
        raise RuntimeError("frozen settings changed since this test run started; delete 6c_test/ to restart")
    mpath.write_text(json.dumps(manifest, indent=2))
    rows = run_cases(_case, cases, out / "cases", workers, log, init=_init, initargs=(root, cfg, smoke))
    keys = sorted({k for r in rows for k in r})
    write_csv(out / "results_long.csv", [{k: r.get(k, "") for k in keys} for r in rows])
    return f"{len(cases)} cases, {len(rows)} rows"
