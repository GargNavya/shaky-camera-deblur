"""Save the restored images themselves (not only figures) into results/<step>/images/.

Run from the project root:  python -m experiments.export_images [--steps 2 3 5 6 7]
Each step gets a new folder results/<step>/images/ (or images_v2, ... if one exists; nothing is
overwritten). Every image is rebuilt with the same kernel and noise seed as in that step's
experiment, so it matches the reported numbers. Each folder has index.csv (file, image,
condition, method, PSNR where ground truth exists) and kernels/ with the kernel used.
PSNR in index.csv is shift-aligned (as in steps 6 to 8); steps 3 and 5 reported plain PSNR,
so their numbers can differ slightly. PNGs are 8-bit.

Per step (all 8 project images unless noted):
  step2  first 3 images: sharp, line blur (L 15, 30 deg), curved blur; sigma 0.01
  step3  line L 15 at 30 deg, every noise level: blurred, Wiener and RL with the step 3 best settings
  step5  straight L 15 at 20 deg, sigma 0.005 / 0.01 / 0.02: RL with true, estimated, ST-angle,
         random-angle kernel; curved bend 70 and 120 (sigma 0.01): true vs estimated
  step6  sigma 0.01, kernels line_t75_L15, curved_0, traj_0: RL with true kernel, straight start,
         one blind round of free / PCA / PCA unrotated / straight model, PCA projection of truth
  step7  synthetic: NAFNet on the step 6 cases above; real: every photo in data/real_shaky at 512 px,
         blurred, ours (straight, step 3 rule and 30 iterations), ours + PCA round, NAFNet
"""
import argparse
import json

import numpy as np

from deblur import config
from deblur.data import load_sharp, to_color_square
from deblur.deconv import richardson_lucy, wiener
from deblur.degrade import make_blurred
from deblur.direction import estimate_line_kernel
from deblur.io import new_run_dir, save_kernel_png, save_png
from deblur.kernels import curved_kernel, line_kernel
from deblur.metrics import mse_aligned, psnr_from_mse
from deblur.tables import write_csv
from deblur.tuning import step3_params


class Exporter:
    def __init__(self, step):
        self.dir = new_run_dir(config.results_dir(step) / "images")
        self.dir.mkdir(parents=True)
        self.rows, self.kernels = [], {}
        print(f"{step}: writing to {self.dir}")

    def add(self, img, image, condition, method, x=None, kernel=None, kernel_name=None):
        fname = f"{image}__{condition}__{method}.png".replace(" ", "_").replace("/", "-")
        save_png(img, self.dir / fname)
        row = dict(file=fname, image=image, condition=condition, method=method,
                   psnr=psnr_from_mse(mse_aligned(x, img)) if x is not None else "")
        if kernel is not None:
            kn = (kernel_name or f"{image}__{condition}__{method}").replace(" ", "_").replace("/", "-") + ".png"
            save_kernel_png(kernel, self.dir / "kernels" / kn)
            row["kernel_file"] = f"kernels/{kn}"
        self.rows.append(row)

    def close(self):
        keys = sorted({k for r in self.rows for k in r})
        write_csv(self.dir / "index.csv", [{k: r.get(k, "") for k in keys} for r in self.rows])
        print(f"  {len(self.rows)} images")


def step2(images):
    ex = Exporter("step2")
    for r, (name, x) in enumerate(images[:3]):
        kl, kc = line_kernel(30, 15), curved_kernel(18, 70, np.random.default_rng(10 + r))
        ex.add(x, name, "sharp", "sharp")
        ex.add(make_blurred(x, kl, 0.01, np.random.default_rng(100 + r)), name, "line_L15_t30_s0.01", "blurred", x, kl)
        ex.add(make_blurred(x, kc, 0.01, np.random.default_rng(200 + r)), name, "curved_b70_s0.01", "blurred", x, kc)
    ex.close()


def step3(images, params):
    ex = Exporter("step3")
    k = line_kernel(30, 15)
    for ii, (name, x) in enumerate(images):
        ex.add(x, name, "sharp", "sharp")
        for si, s in enumerate([0.0, 0.005, 0.01, 0.02, 0.03]):
            y = make_blurred(x, k, s, np.random.default_rng(1000 + 10 * ii + si))
            nsr, it = params[s]
            c = f"line_L15_t30_s{s:g}"
            ex.add(y, name, c, "blurred", x, k)
            ex.add(wiener(y, k, nsr), name, c, f"wiener_nsr{nsr:.2g}", x)
            ex.add(richardson_lucy(y, k, it), name, c, f"rl_{it}it", x)
    ex.close()


def step5(images, params):
    from experiments import step5_deblur_estimated as S5
    ex = Exporter("step5")
    theta, L = 20, 15
    for ii, (name, x) in enumerate(images):
        for s in (0.005, 0.01, 0.02):
            si = S5.SIGMAS.index(s)
            rng = np.random.default_rng(50_000 + 1000 * si + 100 * ii + theta + L)
            y = make_blurred(x, line_kernel(theta, L), s, rng)
            c = f"line_L{L}_t{theta}_s{s:g}"
            ex.add(y, name, c, "blurred", x)
            it = params[s][1]
            for kn, (tk, lk) in S5.candidate_kernels(y, theta, L, rng).items():
                k = line_kernel(tk, lk)
                ex.add(richardson_lucy(y, k, it), name, c, f"rl_{kn}_t{tk:.0f}_L{lk:.1f}", x, k)
        for bi, bend in enumerate(S5.BENDS):
            if bend not in (70, 120):
                continue
            rng = np.random.default_rng(90_000 + 1000 * bi + 100 * ii + 0)
            kt = curved_kernel(S5.CURVED_LENGTH, bend, rng)
            y = make_blurred(x, kt, S5.CURVED_SIGMA, rng)
            te, le = estimate_line_kernel(y)
            c = f"curved_b{bend}_s{S5.CURVED_SIGMA:g}"
            it = params[S5.CURVED_SIGMA][1]
            ex.add(y, name, c, "blurred", x, kt)
            ex.add(richardson_lucy(y, kt, it), name, c, "rl_true_kernel", x, kt)
            ex.add(richardson_lucy(y, line_kernel(te, le), it), name, c, "rl_estimated", x, line_kernel(te, le))
    ex.close()


STEP6_KERNELS = ["line_t75_L15", "curved_0", "traj_0"]
STEP6_SIGMA = 0.01


def step6_cases(images):
    from experiments.step6_common import FULL, family_kernels, make_cases
    kernels = family_kernels(FULL["test_kernels"], "test")
    cases = make_cases(images, kernels, FULL["test_sigmas"], "test")
    kd = {kid: k for _, kid, k in kernels}
    return [(c, kd[c["kernel_id"]]) for c in cases if c["kernel_id"] in STEP6_KERNELS and c["sigma"] == STEP6_SIGMA]


def step6(images, params):
    from deblur.kstep import GradSystem, kernel_step, make_proxy
    from deblur.pca import KernelBasis
    root = config.RESULTS_DIR / "step6"
    fz = json.loads((root / "6b_tune" / "frozen_settings.json").read_text())
    Ba = KernelBasis.load(root / "6a_pca" / "basis_aligned.npz")
    Bu = KernelBasis.load(root / "6a_pca" / "basis_unaligned.npz")
    ex = Exporter("step6")
    imgs = dict(images)
    it = params[STEP6_SIGMA][1]
    for c, kt in step6_cases(images):
        x = imgs[c["image"]]
        y = make_blurred(x, kt, c["sigma"], np.random.default_rng(c["seed"]))
        th, L = estimate_line_kernel(y)
        k0 = line_kernel(th, L)
        cond = f"{c['kernel_id']}_s{c['sigma']:g}"
        ex.add(y, c["image"], cond, "blurred", x, kt)
        out = {"rl_true_kernel": kt, "rl_start_straight": k0,
               "rl_pca_projection_of_truth": Ba.project(kt, fz["pca"]["pca_K"])}
        for m in ("free", "pca", "pca_unaligned", "straight"):
            s = fz[m]
            out[f"rl_round1_{m}"] = kernel_step(m, GradSystem(y, make_proxy(y, k0, s["proxy_nsr"]), s["keep"]),
                                                th, L, s, Ba, Bu)[0]
        for meth, k in out.items():
            ex.add(richardson_lucy(y, k, it), c["image"], cond, meth, x, k)
    ex.close()


def step7(images, params):
    from deblur.nafnet import load_nafnet, nafnet_deblur
    from experiments.step7b_real_photos import EXTS, REAL_DIR, estimate, gray, load_photo, rl_color
    from deblur.kstep import GradSystem, kernel_step, make_proxy
    from deblur.noise import nearest_level
    from deblur.pca import KernelBasis
    net, device = load_nafnet()
    ex = Exporter("step7")
    imgs = dict(images)
    for c, kt in step6_cases(images):
        x = imgs[c["image"]]
        y = make_blurred(x, kt, c["sigma"], np.random.default_rng(c["seed"]))
        cond = f"synthetic_{c['kernel_id']}_s{c['sigma']:g}"
        ex.add(y, c["image"], cond, "blurred", x, kt)
        ex.add(nafnet_deblur(net, y, device), c["image"], cond, "nafnet", x)
    root = config.RESULTS_DIR / "step6"
    fz = json.loads((root / "6b_tune" / "frozen_settings.json").read_text())
    Ba = KernelBasis.load(root / "6a_pca" / "basis_aligned.npz")
    photos = sorted(p for p in REAL_DIR.glob("*") if p.suffix.lower() in EXTS) if REAL_DIR.exists() else []
    for p in photos:
        rgb = to_color_square(load_photo(p), 512)
        g = gray(rgb)
        e = estimate(g)
        it = params[nearest_level(e["sigma_est"], params.keys())][1]
        k0 = line_kernel(e["theta"], e["L"])
        s = fz["pca"]
        kp = kernel_step("pca", GradSystem(g, make_proxy(g, k0, s["proxy_nsr"]), s["keep"]), e["theta"], e["L"], s, Ba)[0]
        cond = "real_512px"
        ex.add(rgb, p.stem, cond, "blurred")
        ex.add(rl_color(rgb, k0, it), p.stem, cond, f"ours_straight_rl{it}it", kernel=k0)
        ex.add(rl_color(rgb, k0, 30), p.stem, cond, "ours_straight_rl30it", kernel=k0)
        ex.add(rl_color(rgb, kp, it), p.stem, cond, f"ours_pca_rl{it}it", kernel=kp)
        ex.add(rl_color(rgb, kp, 30), p.stem, cond, "ours_pca_rl30it", kernel=kp)
        ex.add(nafnet_deblur(net, rgb, device), p.stem, cond, "nafnet")
    ex.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", type=int, default=[2, 3, 5, 6, 7])
    a = ap.parse_args()
    images = load_sharp()
    params = step3_params()
    for s in a.steps:
        {2: lambda: step2(images), 3: lambda: step3(images, params), 5: lambda: step5(images, params),
         6: lambda: step6(images, params), 7: lambda: step7(images, params)}[s]()


if __name__ == "__main__":
    main()
