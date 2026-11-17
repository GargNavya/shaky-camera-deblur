import numpy as np

from deblur import config
from deblur.data import load_sharp
from deblur.degrade import make_blurred
from deblur.kernels import curved_kernel, line_kernel
from deblur.metrics import psnr, ssim
from deblur.plotting import image_grid
from deblur.tables import write_csv

THETA = 30
LENGTHS = [5, 10, 15, 25]
SIGMAS = [0.0, 0.01, 0.03]


def psnr_table(x):
    rows = []
    for L in LENGTHS:
        for s in SIGMAS:
            y = make_blurred(x, line_kernel(THETA, L), s, np.random.default_rng(1))
            rows.append(dict(length=L, sigma=s, psnr=round(psnr(x, y), 2), ssim=round(ssim(x, y), 3)))
    return rows


def demo_rows(images, n=3):
    rows = []
    for r, (name, x) in enumerate(images[:n]):
        kl = line_kernel(THETA, 15)
        kc = curved_kernel(18, 70, np.random.default_rng(10 + r))
        yl = make_blurred(x, kl, 0.01, np.random.default_rng(100 + r))
        yc = make_blurred(x, kc, 0.01, np.random.default_rng(200 + r))
        rows.append([(x, f"{name}: sharp", None),
                     (yl, f"line L=15 th={THETA}, PSNR {psnr(x, yl):.1f}", kl),
                     (yc, f"curved L=18, PSNR {psnr(x, yc):.1f}", kc)])
    return rows


def main():
    images = load_sharp()
    out = config.results_dir("step2")
    name0, x0 = images[0]
    rows = psnr_table(x0)
    write_csv(out / "psnr_table.csv", rows)
    print(f"PSNR/SSIM of blurred vs sharp, image '{name0}', theta={THETA}")
    print(f"{'L':>4s} {'sigma':>6s} {'PSNR':>7s} {'SSIM':>6s}")
    for r in rows:
        print(f"{r['length']:4d} {r['sigma']:6.2f} {r['psnr']:7.2f} {r['ssim']:6.3f}")
    image_grid(demo_rows(images), out / "blur_demo.png", cell=3.4)
    print(f"\nfigures -> {out}")


if __name__ == "__main__":
    main()
