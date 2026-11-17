import time

import numpy as np

from deblur import config
from deblur.data import load_sharp
from deblur.degrade import make_blurred
from deblur.direction import angle_error_deg, cepstrum, cepstrum_estimate, structure_tensor_angle
from deblur.kernels import curved_kernel, kernel_angle_deg, kernel_anisotropy, line_kernel
from deblur.plotting import cepstrum_figure, method_curves
from deblur.tables import write_csv

SIGMAS = [0.0, 0.005, 0.01, 0.02, 0.03]
THETAS = list(range(0, 180, 15))
LENGTHS = [5, 10, 15, 25]
BENDS = [0, 30, 70, 120]
CURVED_PER_IMAGE = 6
CURVED_LENGTH = 15
CURVED_SIGMAS = [0.0, 0.01]
METHODS = ["structure tensor", "cepstrum (free)", "cepstrum + ST hint"]
SUCCESS_DEG = 5.0


def estimate_all(y):
    th_st, _ = structure_tensor_angle(y)
    cep = cepstrum(y)
    tf, lf, _ = cepstrum_estimate(y, cep=cep)
    tg, lg, _ = cepstrum_estimate(y, theta_hint=th_st, cep=cep)
    return {METHODS[0]: (th_st, np.nan), METHODS[1]: (tf, lf), METHODS[2]: (tg, lg)}


def run_straight(images):
    rows, t0 = [], time.time()
    for si, sigma in enumerate(SIGMAS):
        for ii, (name, x) in enumerate(images):
            for L in LENGTHS:
                for th in THETAS:
                    rng = np.random.default_rng(10_000 * si + 100 * ii + th + L)
                    y = make_blurred(x, line_kernel(th, L), sigma, rng)
                    for m, (te, le) in estimate_all(y).items():
                        rows.append(dict(sigma=sigma, image=name, L=L, theta=th, method=m,
                                         theta_est=te, L_est=le,
                                         ang_err=angle_error_deg(te, th),
                                         len_err=abs(le - L) if not np.isnan(le) else np.nan))
        print(f"  straight sigma={sigma:.3f} done ({time.time() - t0:.0f}s)")
    return rows


def summarize_straight(rows):
    out = []
    for sigma in SIGMAS:
        for L in LENGTHS + ["all"]:
            for m in METHODS:
                sel = [r for r in rows if r["sigma"] == sigma and r["method"] == m
                       and (L == "all" or r["L"] == L)]
                a = np.array([r["ang_err"] for r in sel])
                le = np.array([r["len_err"] for r in sel])
                out.append(dict(sigma=sigma, L=L, method=m, n=len(sel),
                                ang_median=np.median(a), within5=np.mean(a <= SUCCESS_DEG),
                                len_median=np.nanmedian(le) if not np.all(np.isnan(le)) else np.nan))
    return out


def run_curved(images):
    rows, t0 = [], time.time()
    for sigma in CURVED_SIGMAS:
        for bi, bend in enumerate(BENDS):
            for ii, (name, x) in enumerate(images):
                for j in range(CURVED_PER_IMAGE):
                    rng = np.random.default_rng(777 + 1000 * bi + 100 * ii + 10 * j)
                    k = curved_kernel(CURVED_LENGTH, bend, rng)
                    true = kernel_angle_deg(k)
                    y = make_blurred(x, k, sigma, rng)
                    for m, (te, _) in estimate_all(y).items():
                        rows.append(dict(sigma=sigma, bend=bend, image=name, method=m,
                                         kernel_aniso=kernel_anisotropy(k), theta_true=true,
                                         theta_est=te, ang_err=angle_error_deg(te, true)))
        print(f"  curved sigma={sigma:.3f} done ({time.time() - t0:.0f}s)")
    return rows


def summarize_curved(rows):
    out = []
    for sigma in CURVED_SIGMAS:
        for bend in BENDS:
            for m in METHODS:
                sel = [r for r in rows if r["sigma"] == sigma and r["bend"] == bend and r["method"] == m]
                a = np.array([r["ang_err"] for r in sel])
                out.append(dict(sigma=sigma, bend=bend, method=m, n=len(sel),
                                kernel_aniso=np.mean([r["kernel_aniso"] for r in sel]),
                                ang_median=np.median(a), within5=np.mean(a <= SUCCESS_DEG)))
    return out


def example_panels(images):
    name, x = images[0]
    panels = []
    for sigma in (0.0, 0.02):
        th, L = 40, 15
        y = make_blurred(x, line_kernel(th, L), sigma, np.random.default_rng(3))
        C, _ = cepstrum(y)
        te, le, _ = cepstrum_estimate(y, cep=(C, _))
        panels.append((C, th, te, L, le, f"{name}, sigma={sigma:g}: cepstrum near origin"))
    return panels


def print_table(summary):
    print(f"\n{'sigma':>6s} | " + " | ".join(f"{m:^26s}" for m in METHODS))
    print(f"{'':>6s} | " + " | ".join(f"{'med err':>8s} {'<=5deg':>7s} {'Lerr':>7s}" for _ in METHODS))
    for sigma in SIGMAS:
        cells = []
        for m in METHODS:
            r = next(r for r in summary if r["sigma"] == sigma and r["L"] == "all" and r["method"] == m)
            cells.append(f"{r['ang_median']:7.1f}° {r['within5']:7.0%} {r['len_median']:7.2f}")
        print(f"{sigma:6.3f} | " + " | ".join(cells))
    print("(random guess: median angle error 45°; Lerr = median |L_est - L| in px)")


def main():
    images = load_sharp()
    out = config.results_dir("step4")
    print(f"{len(images)} images")
    rows = run_straight(images)
    summary = summarize_straight(rows)
    print_table(summary)
    write_csv(out / "straight_raw.csv", rows)
    write_csv(out / "straight_summary.csv", summary)

    def get(L, key):
        return {m: [next(r[key] for r in summary if r["sigma"] == s and r["L"] == L and r["method"] == m)
                    for s in SIGMAS] for m in METHODS}
    curves = {0: get("all", "ang_median"), 1: get("all", "within5"),
              2: {m: v for m, v in get("all", "len_median").items() if m != METHODS[0]}}
    method_curves(SIGMAS, curves, "noise sigma",
                  ["median angle error (deg)", f"fraction within {SUCCESS_DEG:g} deg", "median length error (px)"],
                  ["Angle error", "Angle success rate", "Length error"],
                  out / "straight_accuracy.png", hline={0: 45.0})

    crows = run_curved(images)
    csummary = summarize_curved(crows)
    write_csv(out / "curved_raw.csv", crows)
    write_csv(out / "curved_summary.csv", csummary)
    print("\nCurved kernels (median angle error deg / within 5 deg), by bend:")
    for sigma in CURVED_SIGMAS:
        for bend in BENDS:
            cells = []
            for m in METHODS:
                r = next(r for r in csummary if r["sigma"] == sigma and r["bend"] == bend and r["method"] == m)
                cells.append(f"{r['ang_median']:5.1f} / {r['within5']:4.0%}")
            an = next(r for r in csummary if r["sigma"] == sigma and r["bend"] == bend)["kernel_aniso"]
            print(f"sigma={sigma:.3f} bend={bend:3d} aniso={an:.3f} | " + " | ".join(cells))
    for sigma in CURVED_SIGMAS:
        cur = {i: {m: [next(r[k] for r in csummary if r["sigma"] == sigma and r["bend"] == b and r["method"] == m)
                       for b in BENDS] for m in METHODS} for i, k in enumerate(["ang_median", "within5"])}
        method_curves(BENDS, cur, "bend (deg)", ["median angle error (deg)", f"fraction within {SUCCESS_DEG:g} deg"],
                      [f"Curved kernels, sigma={sigma:g}: error", "success rate"],
                      out / f"curved_accuracy_sigma{sigma:g}.png", hline={0: 45.0})
    cepstrum_figure(example_panels(images), out / "cepstrum_examples.png")
    print(f"\nfigures and tables -> {out}")


if __name__ == "__main__":
    main()
