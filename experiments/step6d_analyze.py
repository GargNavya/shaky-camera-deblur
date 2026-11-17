import json

import numpy as np

from experiments.step6_common import (MODELS, family_kernels, load_frozen, load_test_images, phase_dir,
                                      restore_score)
from deblur.degrade import make_blurred
from deblur.deconv import richardson_lucy
from deblur.direction import estimate_line_kernel
from deblur.kernels import kernel_angle_deg, line_kernel
from deblur.kstep import GradSystem, kernel_step, make_proxy
from deblur.pca import KernelBasis
from deblur.plotting import image_grid, method_curves, plt
from deblur.stats import holm, paired_test
from deblur.tables import write_csv
from deblur.tuning import step3_params

NAMES = {"free": "free-form", "pca": "PCA (rotated)", "pca_unaligned": "PCA (not rotated)", "straight": "straight"}
STYLE = {"true kernel": ("#222222", "--"), "start (straight estimate)": ("#969696", ":"),
         "pca projection of truth": ("#6a51a3", "--"), "free-form": ("#d94801", "-"),
         "PCA (rotated)": ("#08519c", "-"), "PCA (not rotated)": ("#6baed6", "-"), "straight": ("#41ab5d", "-")}
MARGIN_H2 = 0.25   # dB, non-inferiority margin for H2
ALPHA = 0.05


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def load_rows(root):
    rows = []
    for p in sorted((phase_dir(root, "6c") / "cases").glob("*.json")):
        rows += json.loads(p.read_text())
    return rows


def primary(rows):
    """{(case_id, method_label): row} for refs + blind round 1."""
    out = {}
    for r in rows:
        if r["experiment"] == "ref":
            out[(r["case_id"], r["method"])] = r
        elif r["experiment"] == "rounds" and r.get("round") == 1:
            out[(r["case_id"], NAMES[r["method"]])] = r
    return out


def main_table(rows, prim):
    cases = {r["case_id"]: r for r in rows if r["experiment"] == "ref" and r["method"] == "blurred"}
    methods = ["true kernel", "start (straight estimate)", "pca projection of truth"] + [NAMES[m] for m in MODELS]
    table = []
    for fam in sorted({c["family"] for c in cases.values()}):
        for s in sorted({c["sigma"] for c in cases.values()}):
            ids = [i for i, c in cases.items() if c["family"] == fam and c["sigma"] == s]
            for m in methods:
                sel = [(prim[(i, m)], prim[(i, "blurred")], prim[(i, "true kernel")]) for i in ids if (i, m) in prim]
                if not sel:
                    continue
                ratio = np.array([a["mse"] / t["mse"] for a, _, t in sel])
                table.append(dict(family=fam, sigma=s, method=m, n=len(sel),
                                  psnr=np.mean([a["psnr"] for a, _, _ in sel]),
                                  gain_over_blurred=np.mean([a["psnr"] - b["psnr"] for a, b, _ in sel]),
                                  loss_vs_true=np.mean([t["psnr"] - a["psnr"] for a, _, t in sel]),
                                  similarity=np.nanmean([_f(a["similarity"]) for a, _, _ in sel]),
                                  error_ratio_median=float(np.median(ratio)),
                                  success_ratio_le2=float(np.mean(ratio <= 2)),
                                  success_ratio_le3=float(np.mean(ratio <= 3))))
    return table


def hypotheses(rows, prim):
    cases = {r["case_id"]: r for r in rows if r["experiment"] == "ref" and r["method"] == "blurred"}
    smax = max(c["sigma"] for c in cases.values())
    spec = [
        ("H1a", "curved shake: PCA beats straight", "PCA (rotated)", "straight", lambda c: c["family"] == "curved", 0.0),
        ("H1b", "curved shake: PCA beats free-form", "PCA (rotated)", "free-form", lambda c: c["family"] == "curved", 0.0),
        ("H2", f"straight shake: PCA no worse than straight (margin {MARGIN_H2} dB)", "PCA (rotated)", "straight",
         lambda c: c["family"] == "straight", MARGIN_H2),
        ("H3", f"highest noise (sigma={smax:g}), all families: PCA beats free-form", "PCA (rotated)", "free-form",
         lambda c: c["sigma"] == smax, 0.0),
        ("H4a", "unseen trajectory kernels: PCA beats straight", "PCA (rotated)", "straight", lambda c: c["family"] == "traj", 0.0),
        ("H4b", "unseen trajectory kernels: PCA beats free-form", "PCA (rotated)", "free-form", lambda c: c["family"] == "traj", 0.0),
    ]
    out = []
    for hid, text, a, b, sel, margin in spec:
        ids = [i for i, c in cases.items() if sel(c) and (i, a) in prim and (i, b) in prim]
        if not ids:
            continue
        t = paired_test([prim[(i, a)]["psnr"] for i in ids], [prim[(i, b)]["psnr"] for i in ids],
                        [cases[i]["image"] for i in ids], alternative="greater", margin=margin)
        out.append(dict(id=hid, hypothesis=text, a=a, b=b, margin=margin, **t))
    adj = holm([h["p_wilcoxon_images"] for h in out])
    for h, p in zip(out, adj):
        h["p_holm"] = float(p)
        ci_ok = h["ci_lo"] > -h["margin"]
        h["verdict"] = "supported" if (p < ALPHA and ci_ok) else "not supported"
    return out


def secondary(rows, prim):
    cases = {r["case_id"]: r for r in rows if r["experiment"] == "ref" and r["method"] == "blurred"}
    comps = [("PCA (rotated)", "start (straight estimate)"), ("free-form", "start (straight estimate)"),
             ("straight", "start (straight estimate)"), ("PCA (rotated)", "PCA (not rotated)"),
             ("PCA (rotated)", "pca projection of truth")]
    out = []
    for fam in sorted({c["family"] for c in cases.values()}):
        for a, b in comps:
            ids = [i for i, c in cases.items() if c["family"] == fam and (i, a) in prim and (i, b) in prim]
            if ids:
                t = paired_test([prim[(i, a)]["psnr"] for i in ids], [prim[(i, b)]["psnr"] for i in ids],
                                [cases[i]["image"] for i in ids])
                out.append(dict(family=fam, a=a, b=b, **t))
    return out


def figures(rows, table, out, cfg):
    fams = sorted({r["family"] for r in table})
    sig = sorted({r["sigma"] for r in table})
    methods = ["true kernel", "start (straight estimate)", "pca projection of truth"] + [NAMES[m] for m in MODELS]
    for key, ylab, fname in [("gain_over_blurred", "RL PSNR gain over blurred (dB)", "main_gain.png"),
                             ("similarity", "kernel similarity to truth", "main_similarity.png")]:
        curves = {i: {m: [next((r[key] for r in table if r["family"] == f and r["sigma"] == s and r["method"] == m), np.nan)
                          for s in sig] for m in methods if not (key == "similarity" and m == "true kernel")}
                  for i, f in enumerate(fams)}
        method_curves(sig, curves, "noise sigma", [ylab] * len(fams), [f"{f} kernels" for f in fams],
                      out / fname, hline={i: 0.0 for i in range(len(fams))} if "gain" in key else None, style=STYLE)

    # ladder: proxy quality -> similarity
    order = cfg["ladder_proxies"] + ["realistic (blind round 1)"]
    curves = {}
    for i, f in enumerate(fams):
        curves[i] = {}
        for m in MODELS:
            v = []
            for p in order:
                if p.startswith("realistic"):
                    sel = [r for r in rows if r["family"] == f and r["experiment"] == "rounds" and r.get("round") == 1 and r["method"] == m]
                else:
                    sel = [r for r in rows if r["family"] == f and r["experiment"] == "ladder" and r.get("proxy") == p and r["method"] == m]
                v.append(np.mean([_f(r["similarity"]) for r in sel]) if sel else np.nan)
            curves[i][NAMES[m]] = v
    method_curves(list(range(len(order))), curves, "sharp proxy used for the kernel step",
                  ["kernel similarity to truth"] * len(fams), [f"{f}: proxy quality" for f in fams],
                  out / "ladder_similarity.png", style=STYLE,
                  xticklabels=[{"oracle": "true image", "oracle+noise": "+ noise", "oracle+blur": "+ blur"}.get(p, "realistic")
                               for p in order])

    # rounds
    R = list(range(1, cfg["rounds"] + 1))
    curves = {i: {NAMES[m]: [np.mean([_f(r["similarity"]) for r in rows if r["family"] == f and r["experiment"] == "rounds"
                                      and r["method"] == m and r.get("round") == k]) for k in R] for m in MODELS}
              for i, f in enumerate(fams)}
    for i, f in enumerate(fams):
        start = np.mean([_f(r["similarity"]) for r in rows if r["family"] == f and r["method"] == "start (straight estimate)"])
        curves[i]["start (straight estimate)"] = [start] * len(R)
    method_curves(R, curves, "blind round", ["kernel similarity to truth"] * len(fams),
                  [f"{f}: repeated rounds" for f in fams], out / "rounds_similarity.png", style=STYLE)
    RS = cfg["rounds_scored"]
    curves = {i: {NAMES[m]: [np.mean([r["psnr"] - next(b["psnr"] for b in rows if b["case_id"] == r["case_id"] and b["method"] == "blurred")
                                      for r in rows if r["family"] == f and r["experiment"] == "rounds" and r["method"] == m
                                      and r.get("round") == k]) for k in RS] for m in MODELS} for i, f in enumerate(fams)}
    method_curves(RS, curves, "blind round", ["RL PSNR gain over blurred (dB)"] * len(fams),
                  [f"{f}: repeated rounds" for f in fams], out / "rounds_gain.png",
                  hline={i: 0.0 for i in range(len(fams))}, style=STYLE)

    # K ablation (frozen K taken from blind round 1)
    Ks = sorted(set(cfg["ablation_K"]))
    curves = {}
    for i, f in enumerate(fams):
        v = []
        for K in Ks:
            sel = [r for r in rows if r["family"] == f and r["experiment"] == "ablK" and r.get("K") == K]
            if not sel:
                sel = [r for r in rows if r["family"] == f and r["experiment"] == "rounds" and r["method"] == "pca" and r.get("round") == 1]
            v.append(np.mean([_f(r["similarity"]) for r in sel]))
        curves[i] = {"PCA (rotated)": v}
    method_curves(Ks, curves, "number of components K", ["kernel similarity to truth"] * len(fams),
                  [f"{f}: K ablation" for f in fams], out / "ablation_K.png", style=STYLE)

    # starting point
    starts = ["estimate"] + [f"angle+{o:g}" for o in cfg["init_offsets"]] + ["true kernel"]
    curves = {}
    for i, f in enumerate(fams):
        curves[i] = {}
        for m in ("free", "pca", "straight"):
            v = []
            for s in starts:
                if s == "estimate":
                    sel = [r for r in rows if r["family"] == f and r["experiment"] == "rounds" and r["method"] == m and r.get("round") == 1]
                else:
                    sel = [r for r in rows if r["family"] == f and r["experiment"] == "init" and r["method"] == m and r.get("start") == s]
                v.append(np.mean([_f(r["similarity"]) for r in sel]) if sel else np.nan)
            curves[i][NAMES[m]] = v
    method_curves(list(range(len(starts))), curves, "starting kernel",
                  ["kernel similarity after one round"] * len(fams), [f"{f}: starting point" for f in fams],
                  out / "init_start.png", style=STYLE, xticklabels=starts)

    # error-ratio cumulative curves (Levin et al. style)
    prim = primary(rows)
    ids = [r["case_id"] for r in rows if r["experiment"] == "ref" and r["method"] == "blurred"]
    fig, axes = plt.subplots(1, len(fams), figsize=(4.2 * len(fams), 3.6), squeeze=False)
    grid = np.linspace(1, 6, 51)
    for ax, f in zip(axes[0], fams):
        fids = [i for i in ids if prim[(i, "blurred")]["family"] == f]
        for m in ["start (straight estimate)", "pca projection of truth"] + [NAMES[x] for x in MODELS]:
            rr = np.array([prim[(i, m)]["mse"] / prim[(i, "true kernel")]["mse"] for i in fids if (i, m) in prim])
            col, ls = STYLE[m]
            ax.plot(grid, [(rr <= g).mean() for g in grid], color=col, ls=ls, lw=2, label=m)
        ax.set_xlabel("error ratio (MSE estimate / MSE true kernel)")
        ax.set_ylabel("fraction of cases at or below")
        ax.set_title(f"{f} kernels", fontsize=10, loc="left")
        ax.grid(alpha=0.25)
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "error_ratio_cdf.png", dpi=110)
    plt.close(fig)


def qualitative(root, cfg, smoke, out):
    """Re-run one curved and one trajectory case on the first test image to show kernels and results."""
    fz = load_frozen(root)
    Ba = KernelBasis.load(phase_dir(root, "6a") / "basis_aligned.npz")
    Bu = KernelBasis.load(phase_dir(root, "6a") / "basis_unaligned.npz")
    params = step3_params()
    name, x = load_test_images(smoke)[0]
    sigma = 0.01 if 0.01 in cfg["test_sigmas"] else cfg["test_sigmas"][0]
    kernels = {f: (kid, k) for f, kid, k in family_kernels(cfg["test_kernels"], "test")}
    grid = []
    for fam in ("curved", "traj"):
        kid, kt = kernels[fam]
        y = make_blurred(x, kt, sigma, np.random.default_rng(4242))
        th, L = estimate_line_kernel(y)
        k0 = line_kernel(th, L)
        row = [(y, f"{name}, {kid}, sigma={sigma:g}: blurred", kt)]
        _, it = params[sigma]
        for label, k in [("true kernel", kt), ("start (straight)", k0)] + [
                (NAMES[m], kernel_step(m, GradSystem(y, make_proxy(y, k0, fz[m]["proxy_nsr"]), fz[m]["keep"]),
                                       th, L, fz[m], Ba, Bu)[0]) for m in ("free", "pca", "straight")]:
            p, _ = restore_score(x, y, k, sigma, params)
            row.append((richardson_lucy(y, k, it), f"{label}: {p:.1f} dB", k))
        grid.append(row)
    image_grid(grid, out / "qualitative.png", cell=3.0, title_size=7)


def run(cfg, root, workers, log, smoke=False):
    out = phase_dir(root, "6d")
    rows = load_rows(root)
    for r in rows:
        for k in ("similarity", "psnr", "mse"):
            r[k] = _f(r.get(k))
    prim = primary(rows)
    table = main_table(rows, prim)
    write_csv(out / "main_table.csv", table)
    hyp = hypotheses(rows, prim)
    write_csv(out / "hypotheses.csv", hyp)
    sec = secondary(rows, prim)
    write_csv(out / "secondary_tests.csv", sec)
    figures(rows, table, out, cfg)
    qualitative(root, cfg, smoke, out)

    fz = load_frozen(root)
    lines = ["STEP 6 SUMMARY (generated by step6d_analyze.py; protocol in PROTOCOL.txt)", "",
             f"cases: {len({r['case_id'] for r in rows})}, images: {len({r['image'] for r in rows})}",
             "frozen settings (from dev images): " + json.dumps({m: {k: v for k, v in fz[m].items() if k != 'dev_mean_similarity'} for m in MODELS}),
             "", "HYPOTHESES (primary metric: aligned RL PSNR after one blind round; one-sided; Holm over all six)"]
    for h in hyp:
        lines.append(f"  {h['id']:4s} {h['verdict']:13s} {h['hypothesis']}")
        lines.append(f"       mean diff {h['mean_diff']:+.3f} dB, 95% CI [{h['ci_lo']:+.3f}, {h['ci_hi']:+.3f}], "
                     f"win rate {h['win_rate']:.0%}, p(images) {h['p_wilcoxon_images']:.4f}, Holm {h['p_holm']:.4f}, "
                     f"p(cases) {h['p_wilcoxon_cases']:.2g}, n={h['n_cases']} cases / {h['n_images']} images")
    lines += ["", "MAIN TABLE (mean RL PSNR gain over blurred, dB | kernel similarity | error ratio <= 2)"]
    for fam in sorted({r["family"] for r in table}):
        for s in sorted({r["sigma"] for r in table}):
            lines.append(f"  {fam} sigma={s:g}")
            for r in table:
                if r["family"] == fam and r["sigma"] == s:
                    lines.append(f"    {r['method']:28s} {r['gain_over_blurred']:+6.2f} | {r['similarity']:.3f} | {r['success_ratio_le2']:.0%}")
    lines += ["", "SECONDARY (two-sided, not corrected; descriptive)"]
    for t in sec:
        lines.append(f"  {t['family']:8s} {t['a']} - {t['b']}: {t['mean_diff']:+.3f} dB [{t['ci_lo']:+.3f}, {t['ci_hi']:+.3f}], win {t['win_rate']:.0%}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    for l in lines:
        log.info(l)
    return "analysis written"
