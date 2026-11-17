import numpy as np

from experiments.step6_common import FULL, family_kernels
from deblur import config
from deblur.data import load_sharp
from deblur.deconv import richardson_lucy
from deblur.degrade import make_blurred
from deblur.metrics import mse_aligned, psnr_from_mse
from deblur.nafnet import load_nafnet, nafnet_deblur
from deblur.plotting import image_grid, method_curves
from deblur.tables import write_csv

SIGMAS = [0.0, 0.005, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.03]
RL_ITERS = 30


def main():
    from experiments.step7b_real_photos import new_run_dir
    out = new_run_dir(config.results_dir("step7") / "7c_noise_sweep")
    out.mkdir(parents=True)
    print(f"writing to {out}")
    net, device = load_nafnet()
    name, x = load_sharp()[0]
    ks = {kid: k for _, kid, k in family_kernels(FULL["test_kernels"], "test")}
    kernels = [("line_t75_L15", ks["line_t75_L15"]), ("curved_0", ks["curved_0"])]
    rows, grid = [], []
    for kid, k in kernels:
        row_imgs = []
        for s in SIGMAS:
            y = make_blurred(x, k, s, np.random.default_rng(123))
            xn = nafnet_deblur(net, y, device)
            pn = psnr_from_mse(mse_aligned(x, xn))
            pr = psnr_from_mse(mse_aligned(x, richardson_lucy(y, k, RL_ITERS)))
            rows.append(dict(kernel=kid, sigma=s, blurred=psnr_from_mse(mse_aligned(x, y)), nafnet=pn,
                             rl_true_kernel=pr, nafnet_out_min=float(xn.min()), nafnet_out_max=float(xn.max()),
                             nafnet_out_mean=float(xn.mean()), input_mean=float(y.mean())))
            print(f"{kid:14s} sigma={s:<7g} blurred {rows[-1]['blurred']:5.1f}  NAFNet {pn:5.1f}  RL(true,{RL_ITERS}) {pr:5.1f}  "
                  f"NAFNet output min/mean/max {xn.min():.2f}/{xn.mean():.2f}/{xn.max():.2f} (input mean {y.mean():.2f})")
            if s in (0.0, 0.01, 0.015, 0.02, 0.03):
                row_imgs.append((xn, f"{kid}, sigma={s:g}: NAFNet {pn:.1f} dB", None))
        grid.append(row_imgs)
    write_csv(out / "sweep.csv", rows)
    curves = {i: {m: [r[m] for r in rows if r["kernel"] == kid] for m in ("blurred", "nafnet", "rl_true_kernel")}
              for i, (kid, _) in enumerate(kernels)}
    method_curves(SIGMAS, curves, "noise sigma", ["PSNR (dB)"] * 2, [f"{name}, {kid}" for kid, _ in kernels],
                  out / "sweep.png", style={"blurred": ("#bdbdbd", ":"), "nafnet": ("#d94801", "-"),
                                            "rl_true_kernel": ("#222222", "--")})
    image_grid(grid, out / "outputs.png", cell=3.0, title_size=7)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
