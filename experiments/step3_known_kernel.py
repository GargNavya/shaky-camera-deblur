# Wiener and Richardson-Lucy with a KNOWN kernel
import time

import numpy as np

from deblur import config
from deblur.data import load_sharp
from deblur.deconv import richardson_lucy, wiener
from deblur.degrade import make_blurred
from deblur.kernels import curved_kernel, line_kernel
from deblur.metrics import psnr, ssim
from deblur.plotting import ablation_figure, image_grid
from deblur.tables import write_csv

SIGMAS = [0.0, 0.005, 0.01, 0.02, 0.03]
NSRS = np.logspace(-6, -1, 11)
ITERS = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300]
THETA, LENGTH = 30, 15
KERNEL = line_kernel(THETA, LENGTH)


def run_grid(images, kernel=KERNEL):
    # Score blurred input, Wiener (every nsr) and RL (every checkpoint) on every image and noise level
    # Arrays are indexed [sigma, image, parameter]
    nS, nI = len(SIGMAS), len(images)
    res = dict(blur_p=np.zeros((nS, nI)), blur_s=np.zeros((nS, nI)),
               w_p=np.zeros((nS, nI, len(NSRS))), w_s=np.zeros((nS, nI, len(NSRS))),
               r_p=np.zeros((nS, nI, len(ITERS))), r_s=np.zeros((nS, nI, len(ITERS))))
    t0 = time.time()
    for si, sigma in enumerate(SIGMAS):
        for ii, (_, x) in enumerate(images):
            y = make_blurred(x, kernel, sigma, np.random.default_rng(1000 + 10 * ii + si))
            res["blur_p"][si, ii], res["blur_s"][si, ii] = psnr(x, y), ssim(x, y)
            for ni, nsr in enumerate(NSRS):
                xh = wiener(y, kernel, nsr)
                res["w_p"][si, ii, ni], res["w_s"][si, ii, ni] = psnr(x, xh), ssim(x, xh)
            rl = richardson_lucy(y, kernel, max(ITERS), checkpoints=ITERS)
            for ti, it in enumerate(ITERS):
                res["r_p"][si, ii, ti], res["r_s"][si, ii, ti] = psnr(x, rl[it]), ssim(x, rl[it])
        print(f"  sigma={sigma:.3f} done ({time.time() - t0:.0f}s)")
    return res


def summarize(res):
    # One global best parameter per noise level (by mean PSNR over images)
    rows = []
    for si, sigma in enumerate(SIGMAS):
        bw = int(np.argmax(res["w_p"][si].mean(0)))
        br = int(np.argmax(res["r_p"][si].mean(0)))
        rows.append(dict(
            sigma=sigma,
            blur_psnr=res["blur_p"][si].mean(), blur_ssim=res["blur_s"][si].mean(),
            wiener_nsr=NSRS[bw], wiener_psnr=res["w_p"][si, :, bw].mean(),
            wiener_ssim=res["w_s"][si, :, bw].mean(),
            rl_iters=ITERS[br], rl_psnr=res["r_p"][si, :, br].mean(),
            rl_ssim=res["r_s"][si, :, br].mean(),
            wiener_gain=res["w_p"][si, :, bw].mean() - res["blur_p"][si].mean(),
            rl_gain=res["r_p"][si, :, br].mean() - res["blur_p"][si].mean()))
    return rows


def per_image(images, res, summary):
    rows = []
    for si, srow in enumerate(summary):
        bw, br = list(NSRS).index(srow["wiener_nsr"]), ITERS.index(srow["rl_iters"])
        for ii, (name, _) in enumerate(images):
            rows.append(dict(sigma=srow["sigma"], image=name,
                             blur_psnr=res["blur_p"][si, ii], blur_ssim=res["blur_s"][si, ii],
                             wiener_psnr=res["w_p"][si, ii, bw], wiener_ssim=res["w_s"][si, ii, bw],
                             rl_psnr=res["r_p"][si, ii, br], rl_ssim=res["r_s"][si, ii, br]))
    return rows


def qualitative(images, summary, sigma=0.01):
    srow = next(r for r in summary if r["sigma"] == sigma)
    nsr, it = srow["wiener_nsr"], srow["rl_iters"]
    name, x = images[0]
    kernels = [("line", KERNEL), ("curved", curved_kernel(18, 70, np.random.default_rng(7)))]
    rows = []
    for kn, kern in kernels:
        y = make_blurred(x, kern, sigma, np.random.default_rng(55))
        xw, xr = wiener(y, kern, nsr), richardson_lucy(y, kern, it)
        rows.append([(x, f"{name}: sharp (truth)", None),
                     (y, f"blurred ({kn}) {psnr(x, y):.1f} dB", None),
                     (xw, f"Wiener nsr={nsr:.0e} {psnr(x, xw):.1f} dB", None),
                     (xr, f"RL {it} it {psnr(x, xr):.1f} dB", None)])
    return rows


def print_summary(summary):
    print("\nMean over images; best parameter per noise level (by mean PSNR).")
    print(f"{'sigma':>6s} | {'blurred':>13s} | {'Wiener nsr':>10s} {'PSNR':>6s} {'SSIM':>6s} "
          f"| {'RL iters':>8s} {'PSNR':>6s} {'SSIM':>6s} | gain W / RL (dB)")
    for r in summary:
        print(f"{r['sigma']:6.3f} | {r['blur_psnr']:6.2f} {r['blur_ssim']:6.3f} | "
              f"{r['wiener_nsr']:10.1e} {r['wiener_psnr']:6.2f} {r['wiener_ssim']:6.3f} | "
              f"{r['rl_iters']:8d} {r['rl_psnr']:6.2f} {r['rl_ssim']:6.3f} | "
              f"{r['wiener_gain']:+.2f} / {r['rl_gain']:+.2f}")


def main():
    images = load_sharp()
    out = config.results_dir("step3")
    print(f"{len(images)} images, kernel: line theta={THETA} L={LENGTH}")
    res = run_grid(images)
    summary = summarize(res)
    print_summary(summary)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "per_image.csv", per_image(images, res, summary))
    ablation_figure(SIGMAS, NSRS, res["w_p"].mean(1), ITERS, res["r_p"].mean(1),
                    res["blur_p"].mean(1), out / "ablation.png")
    image_grid(qualitative(images, summary), out / "qualitative.png", cell=3.4)
    print(f"\nfigures and tables -> {out}")


if __name__ == "__main__":
    main()
