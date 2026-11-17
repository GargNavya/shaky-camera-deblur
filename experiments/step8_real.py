import json

import numpy as np

from experiments.step7b_real_photos import EXTS, REAL_DIR, estimate, gray, load_photo
from experiments.step8_common import LABEL, apply, rl_iters, rl_with_checkpoints
from deblur.data import to_color_square
from deblur.io import new_run_dir, save_kernel_png, save_png
from deblur.kernels import line_kernel
from deblur.noise import nearest_level
from deblur.plotting import image_grid
from deblur.tables import write_csv
from deblur.tuning import step3_params

METHODS = ["rl", "notch", "blend", "guided", "tv", "hl"]
SIZE = 512


def run(cfg, root, workers, log, smoke=False):
    photos = sorted(p for p in REAL_DIR.glob("*") if p.suffix.lower() in EXTS) if REAL_DIR.exists() else []
    if not photos:
        log.info(f"no photos in {REAL_DIR}; nothing to do")
        return "no photos"
    out = new_run_dir(root / "real")
    out.mkdir(parents=True)
    frozen = json.loads((root / "tune" / "frozen_settings.json").read_text())
    params = step3_params()
    try:
        from deblur.nafnet import load_nafnet, nafnet_deblur
        net, device = load_nafnet()
    except Exception as e:
        log.info(f"NAFNet unavailable ({e}); continuing without it")
        net = None
    rows = []
    for p in photos:
        rgb = to_color_square(load_photo(p), SIZE)
        g = gray(rgb)
        e = estimate(g)
        lvl = nearest_level(e["sigma_est"], cfg["sigmas"])
        it = rl_iters(lvl, params)
        k = line_kernel(e["theta"], e["L"])
        fz = frozen["per_sigma"][str(lvl)]
        rl_ch = [rl_with_checkpoints(rgb[..., c], k, it) for c in range(3)]
        results = {}
        for m in METHODS:
            results[m] = np.stack([apply(m, rgb[..., c], k, fz.get(m, {}), rl_ch[c]) for c in range(3)], axis=2)
        if net is not None:
            results["nafnet"] = nafnet_deblur(net, rgb, device)
        save_png(rgb, out / "images" / f"{p.stem}__blurred.png")
        save_kernel_png(k, out / "images" / f"{p.stem}__kernel.png")
        for m, im in results.items():
            save_png(im, out / "images" / f"{p.stem}__{m}.png")
        lab = dict(LABEL, nafnet="NAFNet (GoPro)")
        panels = [(rgb, f"{p.stem}: blurred", k)] + [(im, lab[m], None) for m, im in results.items()]
        image_grid([panels[:4], panels[4:]], out / f"{p.stem}_all.png", cell=4.0, dpi=110, title_size=9)
        c0 = SIZE // 2 - 96
        zoom = [(im[c0:c0 + 192, c0:c0 + 192], t, None) for im, t, _ in panels]
        image_grid([zoom[:4], zoom[4:]], out / f"{p.stem}_zoom.png", cell=3.6, dpi=110, title_size=9)
        rows.append(dict(photo=p.stem, theta=e["theta"], L=e["L"], sigma_est=e["sigma_est"], noise_level_used=lvl,
                         rl_iters=it, settings=json.dumps(fz, sort_keys=True)))
        log.info(f"{p.stem}: theta {e['theta']:.0f}, L {e['L']:.1f}, sigma est {e['sigma_est']:.4f} -> level {lvl}, RL {it} it")
    write_csv(out / "settings.csv", rows)
    return f"{len(photos)} photos in {out.name}"
