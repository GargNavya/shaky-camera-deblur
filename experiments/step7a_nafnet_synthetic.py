import argparse
import csv
import json
import time

import numpy as np

from experiments.step6_common import cfg_for, family_kernels, load_test_images, out_root, phase_dir
from deblur import config
from deblur.degrade import make_blurred
from deblur.deconv import richardson_lucy
from deblur.direction import estimate_line_kernel
from deblur.kernels import line_kernel
from deblur.metrics import mse_aligned, psnr_from_mse
from deblur.nafnet import load_nafnet, nafnet_deblur
from deblur.plotting import image_grid, method_curves
from deblur.stats import paired_test
from deblur.tables import write_csv
from deblur.tuning import step3_params

OURS = {"true kernel": "true kernel (upper bound)", "start (straight estimate)": "ours: straight estimate",
        "pca": "ours: + one PCA round", "blurred": "blurred"}
STYLE = {"true kernel (upper bound)": ("#222222", "--"), "ours: straight estimate": ("#08519c", "-"),
         "ours: + one PCA round": ("#6baed6", "-"), "NAFNet (GoPro)": ("#d94801", "-")}


def our_scores(root):
    """{case_id: {label: row}} from the step 6c long table."""
    out, meta = {}, {}
    with open(phase_dir(root, "6c") / "results_long.csv") as f:
        for r in csv.DictReader(f):
            lab = None
            if r["experiment"] == "ref" and r["method"] in OURS:
                lab = OURS[r["method"]]
            elif r["experiment"] == "rounds" and r["method"] == "pca" and r.get("round") in ("1", "1.0"):
                lab = OURS["pca"]
            if lab:
                out.setdefault(r["case_id"], {})[lab] = float(r["psnr"])
                meta[r["case_id"]] = dict(image=r["image"], family=r["family"], kernel_id=r["kernel_id"],
                                          sigma=float(r["sigma"]), seed=int(r["seed"]))
    return out, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    root6, cfg = out_root(a.smoke), cfg_for(a.smoke)
    out = config.results_dir("step7_smoke" if a.smoke else "step7") / "7a_synthetic"
    (out / "cases").mkdir(parents=True, exist_ok=True)
    ours, meta = our_scores(root6)
    images = dict(load_test_images(a.smoke))
    kernels = {kid: k for _, kid, k in family_kernels(cfg["test_kernels"], "test")}
    net, device = load_nafnet()
    print(f"NAFNet on {device}; {len(meta)} cases")
    t0 = time.time()
    for i, (cid, m) in enumerate(sorted(meta.items())):
        p = out / "cases" / f"{cid}.json"
        if p.exists():
            continue
        x, k = images[m["image"]], kernels[m["kernel_id"]]
        y = make_blurred(x, k, m["sigma"], np.random.default_rng(m["seed"]))
        xn = nafnet_deblur(net, y, device)
        p.write_text(json.dumps(dict(case_id=cid, nafnet_psnr=psnr_from_mse(mse_aligned(x, xn)))))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(meta)} ({(time.time() - t0) / 60:.1f} min)")

    rows = []
    for cid, m in sorted(meta.items()):
        r = dict(case_id=cid, **m, **ours[cid])
        r["NAFNet (GoPro)"] = json.loads((out / "cases" / f"{cid}.json").read_text())["nafnet_psnr"]
        rows.append(r)
    write_csv(out / "nafnet_scores.csv", rows)

    methods = ["true kernel (upper bound)", "ours: straight estimate", "ours: + one PCA round", "NAFNet (GoPro)"]
    fams = sorted({r["family"] for r in rows})
    sigmas = sorted({r["sigma"] for r in rows})
    table = []
    for f in fams:
        for s in sigmas:
            sel = [r for r in rows if r["family"] == f and r["sigma"] == s]
            for meth in methods:
                g = np.array([r[meth] - r["blurred"] for r in sel])
                table.append(dict(family=f, sigma=s, method=meth, n=len(sel), gain_mean=g.mean(),
                                  gain_median=np.median(g), frac_improved=np.mean(g > 0)))
    write_csv(out / "comparison_table.csv", table)

    tests = []
    for f in fams + ["all"]:
        sel = [r for r in rows if f == "all" or r["family"] == f]
        for b in ("ours: straight estimate", "ours: + one PCA round"):
            t = paired_test([r["NAFNet (GoPro)"] for r in sel], [r[b] for r in sel], [r["image"] for r in sel])
            tests.append(dict(family=f, a="NAFNet (GoPro)", b=b, **t))
    write_csv(out / "comparison_tests.csv", tests)

    curves = {i: {meth: [next(t["gain_mean"] for t in table if t["family"] == f and t["sigma"] == s and t["method"] == meth)
                         for s in sigmas] for meth in methods} for i, f in enumerate(fams)}
    method_curves(sigmas, curves, "noise sigma", ["PSNR gain over blurred (dB)"] * len(fams),
                  [f"{f} kernels" for f in fams], out / "gain_by_family.png",
                  hline={i: 0.0 for i in range(len(fams))}, style=STYLE)

    # qualitative: first image, one kernel per family, sigma 0.01 (or the first level)
    params = step3_params()
    name, x = next(iter(images.items()))
    s = 0.01 if 0.01 in sigmas else sigmas[0]
    grid = []
    for f in fams:
        cid, m = next((c, m) for c, m in sorted(meta.items()) if m["image"] == name and m["family"] == f and m["sigma"] == s)
        k = kernels[m["kernel_id"]]
        y = make_blurred(x, k, s, np.random.default_rng(m["seed"]))
        th, L = estimate_line_kernel(y)
        ke = line_kernel(th, L)
        it = params[s][1]
        grid.append([(y, f"{name}, {m['kernel_id']}, sigma={s:g}: blurred", k),
                     (richardson_lucy(y, k, it), f"true kernel: {ours[cid]['true kernel (upper bound)']:.1f} dB", k),
                     (richardson_lucy(y, ke, it), f"ours (straight): {ours[cid]['ours: straight estimate']:.1f} dB", ke),
                     (nafnet_deblur(net, y, device), f"NAFNet: {next(r for r in rows if r['case_id'] == cid)['NAFNet (GoPro)']:.1f} dB", None)])
    image_grid(grid, out / "qualitative.png", cell=3.2, title_size=7)

    print("\nMean PSNR gain over blurred (dB): true kernel | ours straight | ours + PCA | NAFNet")
    for f in fams:
        for s in sigmas:
            v = [next(t["gain_mean"] for t in table if t["family"] == f and t["sigma"] == s and t["method"] == meth) for meth in methods]
            print(f"  {f:8s} sigma={s:<6g} " + " | ".join(f"{x:+6.2f}" for x in v))
    print("\nNAFNet minus ours (mean dB, 95% cluster-bootstrap CI, NAFNet win rate):")
    for t in tests:
        print(f"  {t['family']:8s} vs {t['b']:24s} {t['mean_diff']:+.2f} [{t['ci_lo']:+.2f}, {t['ci_hi']:+.2f}]  win {t['win_rate']:.0%}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
