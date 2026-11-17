import json

import numpy as np
from PIL import Image, ImageOps

from deblur import config
from deblur.data import to_color_square
from deblur.deconv import richardson_lucy
from deblur.direction import cepstrum, cepstrum_estimate, structure_tensor_angle
from deblur.kernels import kernel_angle_deg, line_kernel
from deblur.kstep import GradSystem, kernel_step, make_proxy
from deblur.noise import estimate_sigma, nearest_level
from deblur.pca import KernelBasis
from deblur.plotting import cepstrum_figure, image_grid, plt
from deblur.tables import write_csv
from deblur.tuning import step3_params

SIZES = [384, 512, 768]
SWEEP_ITERS = [10, 30, 75, 300]
MAIN_SIZE = 512
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".heic"}
REAL_DIR = config.DATA_DIR / "real_shaky"


def new_run_dir(base):
    if not base.exists():
        return base
    v = 2
    while (base.parent / f"{base.name}_v{v}").exists():
        v += 1
    return base.parent / f"{base.name}_v{v}"


def load_photo(p):
    return np.asarray(ImageOps.exif_transpose(Image.open(p)).convert("RGB"))


def gray(rgb):
    return rgb @ np.array([0.2125, 0.7154, 0.0721])


def estimate(g):
    th_st, aniso = structure_tensor_angle(g)
    th, L, fc = cepstrum_estimate(g, theta_hint=th_st)
    return dict(theta_st=th_st, st_anisotropy=aniso, theta=th, L=min(L, 28.0), f_cutoff=fc,
                sigma_est=estimate_sigma(g))


def rl_color(rgb, k, it):
    return np.stack([richardson_lucy(rgb[..., c], k, it) for c in range(3)], axis=2)


def srgb_to_linear(v):
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(v):
    v = np.clip(v, 0, 1)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * v ** (1 / 2.4) - 0.055)


def iteration_sweep(rgb, k, name, path):
    lin = srgb_to_linear(rgb)
    grid = []
    for label, f in [("gamma-encoded", lambda it: rl_color(rgb, k, it)),
                     ("linear light", lambda it: linear_to_srgb(rl_color(lin, k, it)))]:
        grid.append([(f(it), f"{name}, {label}, RL {it} it", k if it == SWEEP_ITERS[-1] else None) for it in SWEEP_ITERS])
    image_grid(grid, path, cell=3.6, dpi=120, title_size=8)


def main():
    photos = sorted(p for p in REAL_DIR.glob("*") if p.suffix.lower() in EXTS) if REAL_DIR.exists() else []
    if not photos:
        raise SystemExit(f"no photos in {REAL_DIR}")
    out = new_run_dir(config.results_dir("step7") / "7b_real")
    out.mkdir(parents=True)
    print(f"writing to {out}")
    params = step3_params()
    root6 = config.RESULTS_DIR / "step6"
    fz = json.loads((root6 / "6b_tune" / "frozen_settings.json").read_text())
    Ba = KernelBasis.load(root6 / "6a_pca" / "basis_aligned.npz")
    try:
        from deblur.nafnet import load_nafnet, nafnet_deblur
        net, device = load_nafnet()
    except Exception as e:
        print(f"NAFNet unavailable ({e}); continuing without it")
        net = None
    rows = []
    for p in photos:
        rgb_full = load_photo(p)
        print(f"{p.name}: {rgb_full.shape[1]}x{rgb_full.shape[0]}")
        for size in SIZES:
            g = gray(to_color_square(rgb_full, size))
            e = estimate(g)
            rows.append(dict(photo=p.stem, size=size, **e, L_per_512=e["L"] * 512 / size))
            print(f"  {size}px: theta {e['theta']:.0f} deg (ST {e['theta_st']:.0f}), L {e['L']:.2f} px, "
                  f"sigma_est {e['sigma_est']:.4f}, cutoff {e['f_cutoff']:.3f}")

        rgb = to_color_square(rgb_full, MAIN_SIZE)
        g = gray(rgb)
        e = next(r for r in rows if r["photo"] == p.stem and r["size"] == MAIN_SIZE)
        lvl = nearest_level(e["sigma_est"], params.keys())
        it = params[lvl][1]
        k0 = line_kernel(e["theta"], e["L"])
        s = fz["pca"]
        kp = kernel_step("pca", GradSystem(g, make_proxy(g, k0, s["proxy_nsr"]), s["keep"]),
                         e["theta"], e["L"], s, Ba)[0]
        panels = [(rgb, f"{p.stem}: blurred (sigma est {e['sigma_est']:.3f})", None),
                  (rl_color(rgb, k0, it), f"ours: straight, {e['theta']:.0f} deg, L={e['L']:.1f}, RL {it} it", k0),
                  (rl_color(rgb, kp, it), f"ours: + PCA round (angle {kernel_angle_deg(kp):.0f} deg)", kp)]
        if net is not None:
            panels.append((nafnet_deblur(net, rgb, device), "NAFNet (GoPro)", None))
        image_grid([panels], out / f"{p.stem}_comparison.png", cell=4.2, dpi=120, title_size=8)
        iteration_sweep(rgb, k0, p.stem, out / f"{p.stem}_iterations.png")
        C, _ = cepstrum(g)
        cepstrum_figure([(C, e["theta"], e["theta"], e["L"], e["L"], f"{p.stem}: cepstrum, estimate {e['theta']:.0f} deg, L={e['L']:.1f}")],
                        out / f"{p.stem}_cepstrum.png")

    write_csv(out / "estimates.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for name in sorted({r["photo"] for r in rows}):
        r = [x for x in rows if x["photo"] == name]
        axes[0].plot([x["size"] for x in r], [x["L_per_512"] for x in r], marker="o", label=name)
        axes[1].plot([x["size"] for x in r], [x["theta"] for x in r], marker="o", label=name)
    axes[0].set_ylabel("estimated length, rescaled to 512 px")
    axes[1].set_ylabel("estimated angle (deg)")
    for ax, t in zip(axes, ["Length should be flat if real", "Angle should be flat if real"]):
        ax.set_xlabel("working size (px)")
        ax.set_title(t, fontsize=10, loc="left")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "scale_consistency.png", dpi=110)
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
