import time

import numpy as np

from deblur import config
from deblur.data import load_sharp
from deblur.deconv import richardson_lucy, wiener
from deblur.degrade import make_blurred
from deblur.direction import angle_error_deg, cepstrum_estimate, estimate_line_kernel, structure_tensor_angle
from deblur.kernels import curved_kernel, kernel_angle_deg, line_kernel
from deblur.metrics import psnr, ssim
from deblur.plotting import image_grid, method_curves
from deblur.tables import write_csv
from deblur.tuning import step3_params

SIGMAS = [0.0, 0.005, 0.01, 0.02, 0.03]
THETAS = [20, 75, 140]                 # off the step 4 grid on purpose
LENGTHS = [5, 10, 15, 25]
KERNELS = ["true kernel", "estimated", "ST angle", "random angle"]
STYLE = {"true kernel": ("#222222", "--"), "estimated": ("#08519c", "-"),
         "ST angle": ("#6baed6", "-"), "random angle": ("#bdbdbd", "-"),
         "blurred": ("#bdbdbd", ":")}

SENS_SIGMAS = [0.005, 0.02]
SENS_L = 15
ANGLE_OFFSETS = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30]
LENGTH_OFFSETS = [-6, -4, -2, -1, 0, 1, 2, 4, 6]

BENDS = [0, 30, 70, 120]
CURVED_PER_IMAGE = 4
CURVED_LENGTH = 15
CURVED_SIGMA = 0.01


def deblur_scores(x, y, k, sigma, params):
    nsr, it = params[sigma]
    xw, xr = wiener(y, k, nsr), richardson_lucy(y, k, it)
    return dict(wiener_psnr=psnr(x, xw), wiener_ssim=ssim(x, xw),
                rl_psnr=psnr(x, xr), rl_ssim=ssim(x, xr)), xr


def candidate_kernels(y, theta, L, rng):
    #{name: (theta, length)} for every kernel in KERNELS
    te, le = estimate_line_kernel(y)
    st, _ = structure_tensor_angle(y)
    _, lc, _ = cepstrum_estimate(y, theta_hint=st)
    return {"true kernel": (theta, L), "estimated": (te, le),
            "ST angle": (st, float(np.clip(lc, 1, 28))), "random angle": (rng.uniform(0, 180), L)}


def run_straight(images, params):
    rows, t0 = [], time.time()
    for si, sigma in enumerate(SIGMAS):
        for ii, (name, x) in enumerate(images):
            for L in LENGTHS:
                for theta in THETAS:
                    rng = np.random.default_rng(50_000 + 1000 * si + 100 * ii + theta + L)
                    y = make_blurred(x, line_kernel(theta, L), sigma, rng)
                    base = dict(sigma=sigma, image=name, L=L, theta=theta)
                    rows.append(dict(base, kernel="blurred", theta_used=np.nan, L_used=np.nan,
                                     ang_err=np.nan, wiener_psnr=psnr(x, y), wiener_ssim=ssim(x, y),
                                     rl_psnr=psnr(x, y), rl_ssim=ssim(x, y)))
                    for kn, (tk, lk) in candidate_kernels(y, theta, L, rng).items():
                        sc, _ = deblur_scores(x, y, line_kernel(tk, lk), sigma, params)
                        rows.append(dict(base, kernel=kn, theta_used=tk, L_used=lk,
                                         ang_err=angle_error_deg(tk, theta), **sc))
        print(f"  straight sigma={sigma:.3f} done ({time.time() - t0:.0f}s)")
    return rows


def summarize(rows, keys, groups):
    # Mean of score columns per group, also PSNR gain over 'blurred' and loss vs 'true kernel'
    def mean(sel, c):
        return float(np.mean([r[c] for r in sel]))
    out = []
    for g in groups:
        sel_g = [r for r in rows if all(r[k] == v for k, v in g.items())]
        ref = {kn: [r for r in sel_g if r["kernel"] == kn] for kn in ["blurred", "true kernel"]}
        for kn in ["blurred"] + KERNELS:
            sel = [r for r in sel_g if r["kernel"] == kn]
            if not sel:
                continue
            row = dict(g, kernel=kn, n=len(sel))
            for c in keys:
                row[c] = mean(sel, c)
            for m in ("wiener", "rl"):
                row[f"{m}_gain"] = row[f"{m}_psnr"] - mean(ref["blurred"], f"{m}_psnr")
                if ref["true kernel"]:
                    row[f"{m}_loss_vs_true"] = mean(ref["true kernel"], f"{m}_psnr") - row[f"{m}_psnr"]
            out.append(row)
    return out


def run_sensitivity(images, params):
    rows, t0 = [], time.time()
    for sigma in SENS_SIGMAS:
        for ii, (name, x) in enumerate(images):
            for theta in THETAS:
                rng = np.random.default_rng(70_000 + 100 * ii + theta + int(1000 * sigma))
                y = make_blurred(x, line_kernel(theta, SENS_L), sigma, rng)
                b = psnr(x, y)
                trials = [("angle", d, theta + d, SENS_L) for d in ANGLE_OFFSETS] + \
                         [("length", d, theta, SENS_L + d) for d in LENGTH_OFFSETS if d != 0]
                for kind, off, tk, lk in trials:
                    sc, _ = deblur_scores(x, y, line_kernel(tk, lk), sigma, params)
                    rows.append(dict(sigma=sigma, image=name, theta=theta, kind=kind, offset=off,
                                     blurred_psnr=b, **sc))
        print(f"  sensitivity sigma={sigma:.3f} done ({time.time() - t0:.0f}s)")
    return rows


def run_curved(images, params):
    rows, t0 = [], time.time()
    for bi, bend in enumerate(BENDS):
        for ii, (name, x) in enumerate(images):
            for j in range(CURVED_PER_IMAGE):
                rng = np.random.default_rng(90_000 + 1000 * bi + 100 * ii + j)
                k = curved_kernel(CURVED_LENGTH, bend, rng)
                y = make_blurred(x, k, CURVED_SIGMA, rng)
                te, le = estimate_line_kernel(y)
                base = dict(bend=bend, image=name, kernel_idx=j,
                            ang_err=angle_error_deg(te, kernel_angle_deg(k)))
                rows.append(dict(base, kernel="blurred", wiener_psnr=psnr(x, y), wiener_ssim=ssim(x, y),
                                 rl_psnr=psnr(x, y), rl_ssim=ssim(x, y)))
                for kn, kk in [("true kernel", k), ("estimated", line_kernel(te, le))]:
                    sc, _ = deblur_scores(x, y, kk, CURVED_SIGMA, params)
                    rows.append(dict(base, kernel=kn, **sc))
        print(f"  curved bend={bend} done ({time.time() - t0:.0f}s)")
    return rows


def qualitative(images, params, sigma=0.01):
    name, x = images[0]
    rows = []
    cases = [("line L=15", line_kernel(40, 15)),
             ("curved bend=70", curved_kernel(18, 70, np.random.default_rng(7)))]
    for label, k in cases:
        y = make_blurred(x, k, sigma, np.random.default_rng(11))
        te, le = estimate_line_kernel(y)
        ke = line_kernel(te, le)
        _, xt = deblur_scores(x, y, k, sigma, params)
        _, xe = deblur_scores(x, y, ke, sigma, params)
        _, xrnd = deblur_scores(x, y, line_kernel((te + 90) % 180, le), sigma, params)
        rows.append([(y, f"{name}, {label}, sigma={sigma:g}: blurred {psnr(x, y):.1f} dB", k),
                     (xt, f"RL, true kernel {psnr(x, xt):.1f} dB", k),
                     (xe, f"RL, estimated ({te:.0f} deg, L={le:.1f}) {psnr(x, xe):.1f} dB", ke),
                     (xrnd, f"RL, perpendicular kernel {psnr(x, xrnd):.1f} dB", line_kernel((te + 90) % 180, le))])
    return rows


def pick(summary, key, **match):
    return next(r[key] for r in summary if all(r[k] == v for k, v in match.items()))


def main():
    params = step3_params()
    images = load_sharp()
    out = config.results_dir("step5")
    print(f"{len(images)} images; deconvolution settings from step 3: {params}")
    score_cols = ["wiener_psnr", "wiener_ssim", "rl_psnr", "rl_ssim"]

    # straight kernels
    rows = run_straight(images, params)
    groups = [dict(sigma=s, L=L) for s in SIGMAS for L in LENGTHS]
    summ = summarize(rows, score_cols + ["ang_err"], groups)
    summ_all = summarize(rows, score_cols, [dict(sigma=s) for s in SIGMAS])
    write_csv(out / "straight_raw.csv", rows)
    write_csv(out / "straight_summary.csv", summ + [dict(r, L="all") for r in summ_all])
    print("\nStraight kernels, mean over images, angles and lengths. RL PSNR gain over blurred (dB):")
    print(f"{'sigma':>6s} | " + " | ".join(f"{k:>12s}" for k in KERNELS) + " | est. loss vs true")
    for s in SIGMAS:
        print(f"{s:6.3f} | " + " | ".join(f"{pick(summ_all, 'rl_gain', sigma=s, kernel=k):+12.2f}" for k in KERNELS)
              + f" | {pick(summ_all, 'rl_loss_vs_true', sigma=s, kernel='estimated'):.2f}")
    print("Same for Wiener:")
    for s in SIGMAS:
        print(f"{s:6.3f} | " + " | ".join(f"{pick(summ_all, 'wiener_gain', sigma=s, kernel=k):+12.2f}" for k in KERNELS)
              + f" | {pick(summ_all, 'wiener_loss_vs_true', sigma=s, kernel='estimated'):.2f}")
    curves = {i: {k: [pick(summ, "rl_gain", sigma=s, L=L, kernel=k) for s in SIGMAS] for k in KERNELS}
              for i, L in enumerate(LENGTHS)}
    method_curves(SIGMAS, curves, "noise sigma", ["RL PSNR gain over blurred (dB)"] * len(LENGTHS),
                  [f"L = {L}" for L in LENGTHS], out / "straight_gain.png", hline={i: 0.0 for i in range(4)},
                  style=STYLE)

    # ensitivity to angle and length errors
    srows = run_sensitivity(images, params)
    write_csv(out / "sensitivity_raw.csv", srows)
    sens = {}
    for kind, offs in [("angle", ANGLE_OFFSETS), ("length", LENGTH_OFFSETS)]:
        for s in SENS_SIGMAS:
            vals = []
            for d in offs:
                kk = "angle" if d == 0 else kind
                sel = [r for r in srows if r["sigma"] == s and r["kind"] == kk and r["offset"] == d]
                vals.append(np.mean([r["rl_psnr"] - r["blurred_psnr"] for r in sel]))
            sens[(kind, s)] = vals
    print(f"\nSensitivity (L={SENS_L}), RL gain over blurred (dB):")
    for s in SENS_SIGMAS:
        print(f"  sigma={s}: angle offsets {ANGLE_OFFSETS} -> " + " ".join(f"{v:+.2f}" for v in sens[('angle', s)]))
        print(f"  sigma={s}: length offsets {LENGTH_OFFSETS} -> " + " ".join(f"{v:+.2f}" for v in sens[('length', s)]))
    sstyle = {f"sigma={s:g}": (c, "-") for s, c in zip(SENS_SIGMAS, ["#3182bd", "#08519c"])}
    for kind, offs, xl in [("angle", ANGLE_OFFSETS, "angle error of kernel (deg)"),
                           ("length", LENGTH_OFFSETS, "length error of kernel (px)")]:
        method_curves(offs, {0: {f"sigma={s:g}": sens[(kind, s)] for s in SENS_SIGMAS}}, xl,
                      ["RL PSNR gain over blurred (dB)"], [f"Sensitivity to {kind} error (L = {SENS_L})"],
                      out / f"sensitivity_{kind}.png", hline={0: 0.0}, style=sstyle)

    # curved kernels, deblurred with a straight estimate
    crows = run_curved(images, params)
    csumm = summarize(crows, score_cols + ["ang_err"], [dict(bend=b) for b in BENDS])
    write_csv(out / "curved_raw.csv", crows)
    write_csv(out / "curved_summary.csv", csumm)
    print(f"\nCurved kernels (L={CURVED_LENGTH}, sigma={CURVED_SIGMA}), RL gain over blurred (dB):")
    for b in BENDS:
        print(f"  bend={b:3d}: true kernel {pick(csumm, 'rl_gain', bend=b, kernel='true kernel'):+.2f}, "
              f"straight estimate {pick(csumm, 'rl_gain', bend=b, kernel='estimated'):+.2f}")
    method_curves(BENDS, {0: {k: [pick(csumm, "rl_gain", bend=b, kernel=k) for b in BENDS]
                              for k in ["true kernel", "estimated"]}},
                  "bend (deg)", ["RL PSNR gain over blurred (dB)"],
                  [f"Curved shake, straight estimate (sigma = {CURVED_SIGMA:g})"],
                  out / "curved_gain.png", hline={0: 0.0}, style=STYLE)

    image_grid(qualitative(images, params), out / "qualitative.png", cell=3.6)
    print(f"\nfigures and tables -> {out}")


if __name__ == "__main__":
    main()
