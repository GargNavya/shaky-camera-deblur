import json

import numpy as np

from experiments.step8_common import BRANCH, LABEL, apply, family_kernels, load_test_images, rl_iters, rl_with_checkpoints
from deblur.degrade import make_blurred
from deblur.direction import estimate_line_kernel
from deblur.io import new_run_dir, save_kernel_png, save_png
from deblur.kernels import line_kernel
from deblur.plotting import method_curves, plt
from deblur.stats import holm, paired_test
from deblur.tables import write_csv
from deblur.tuning import step3_params

METRICS = ["psnr", "psnr_flat", "psnr_edge", "ssim"]
ALPHA = 0.05
STYLE = {"RL": ("#969696", "-"), "RL + notch": ("#fdae6b", "-"), "RL + blend": ("#e6550d", "-"),
         "RL + guided": ("#a63603", "-"), "TV deconv": ("#6baed6", "-"), "HL deconv": ("#08519c", "-"),
         "RL, true kernel": ("#222222", "--"), "HL, true kernel": ("#08519c", "--"), "TV, true kernel": ("#6baed6", "--")}


def load(root):
    rows = []
    for p in sorted((root / "test" / "cases").glob("*.json")):
        rows += json.loads(p.read_text())
    by = {}
    for r in rows:
        by.setdefault(r["case_id"], {})[(r["kernel"], r["method"])] = r
    return by


def branch_table(by, methods, kernel):
    out = []
    cases = list(by.values())
    fams = sorted({c[("none", "blurred")]["family"] for c in cases})
    sigmas = sorted({c[("none", "blurred")]["sigma"] for c in cases})
    for f in fams + ["all"]:
        for s in sigmas + ["all"]:
            sel = [c for c in cases if (f == "all" or c[("none", "blurred")]["family"] == f)
                   and (s == "all" or c[("none", "blurred")]["sigma"] == s)]
            for m in ["rl"] + methods:
                row = dict(family=f, sigma=s, kernel=kernel, method=LABEL[m], n=len(sel))
                for k in METRICS:
                    v = np.array([c[(kernel, m)][k] for c in sel])
                    b = np.array([c[(kernel, "rl")][k] for c in sel])
                    z = np.array([c[("none", "blurred")][k] for c in sel])
                    row[f"{k}"] = v.mean()
                    row[f"{k}_vs_rl"] = (v - b).mean()
                    row[f"{k}_vs_blurred"] = (v - z).mean()
                row["win_vs_rl"] = float(np.mean([c[(kernel, m)]["psnr"] > c[(kernel, "rl")]["psnr"] for c in sel]))
                out.append(row)
    return out


def branch_figures(table, methods, out, title):
    sig = sorted({r["sigma"] for r in table if r["sigma"] != "all"})
    labs = ["RL"] + [LABEL[m] for m in methods]
    for kernel in ("estimated", "true"):
        t = [r for r in table if r["kernel"] == kernel and r["family"] == "all"]
        curves = {i: {l: [next(r[k] for r in t if r["sigma"] == s and r["method"] == l) for s in sig] for l in labs}
                  for i, k in enumerate(["psnr_vs_blurred", "psnr_flat_vs_rl", "psnr_edge_vs_rl"])}
        method_curves(sig, curves, "noise sigma",
                      ["PSNR gain over blurred (dB)", "flat-region PSNR vs RL (dB)", "edge-region PSNR vs RL (dB)"],
                      [f"{title}, {kernel} kernel: overall", "ringing (flat regions)", "detail (edges)"],
                      out / f"curves_{kernel}_kernel.png", hline={0: 0.0, 1: 0.0, 2: 0.0}, style=STYLE)
    # trade-off: ringing removed vs detail lost, every (method, sigma, family)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for l in labs[1:]:
        pts = [r for r in table if r["kernel"] == "estimated" and r["method"] == l and r["sigma"] != "all" and r["family"] != "all"]
        col = STYLE.get(l, ("#555555", "-"))[0]
        ax.scatter([r["psnr_edge_vs_rl"] for r in pts], [r["psnr_flat_vs_rl"] for r in pts], s=18, color=col, label=l, alpha=0.8)
    ax.axhline(0, color="#999", lw=0.8)
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_xlabel("edge-region PSNR vs RL (dB): detail kept (right) or lost (left)")
    ax.set_ylabel("flat-region PSNR vs RL (dB): ringing removed")
    ax.set_title(f"{title}: one point per family x noise level", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "tradeoff_flat_vs_edge.png", dpi=110)
    plt.close(fig)


def hypotheses(by, primary):
    a, b = primary["8a"], primary["8b"]
    cases = list(by.values())
    img = [c[("none", "blurred")]["image"] for c in cases]
    spec = [("H8a-1", f"{LABEL[a]} > RL (PSNR)", a, "rl", "psnr"),
            ("H8a-2", f"{LABEL[a]} > RL (flat-region PSNR)", a, "rl", "psnr_flat"),
            ("H8b-1", f"{LABEL[b]} > RL (PSNR)", b, "rl", "psnr"),
            ("H8b-2", f"{LABEL[b]} > RL (flat-region PSNR)", b, "rl", "psnr_flat"),
            ("H8c", f"{LABEL[b]} > {LABEL[a]} (PSNR)", b, a, "psnr")]
    out = []
    for hid, text, m1, m2, k in spec:
        t = paired_test([c[("estimated", m1)][k] for c in cases], [c[("estimated", m2)][k] for c in cases], img,
                        alternative="greater")
        extra = {f"{q}_diff": float(np.mean([c[("estimated", m1)][q] - c[("estimated", m2)][q] for c in cases]))
                 for q in METRICS}
        out.append(dict(id=hid, hypothesis=text, metric=k, **t, **extra))
    for h, p in zip(out, holm([h["p_wilcoxon_images"] for h in out])):
        h["p_holm"] = float(p)
        h["verdict"] = "supported" if (p < ALPHA and h["ci_lo"] > 0) else "not supported"
    return out


def overall_figure(by, primary, out):
    cases = list(by.values())
    sig = sorted({c[("none", "blurred")]["sigma"] for c in cases})
    lines = [("estimated", "rl"), ("estimated", primary["8a"]), ("estimated", primary["8b"]),
             ("true", "rl"), ("true", primary["8b"])]
    curves = {}
    for i, f in enumerate(sorted({c[("none", "blurred")]["family"] for c in cases})):
        curves[i] = {}
        for kn, m in lines:
            lab = LABEL[m] if kn == "estimated" else LABEL[f"{m}_true"]
            curves[i][lab] = [np.mean([c[(kn, m)]["psnr"] - c[("none", "blurred")]["psnr"] for c in cases
                                       if c[("none", "blurred")]["family"] == f and c[("none", "blurred")]["sigma"] == s]) for s in sig]
    fams = sorted({c[("none", "blurred")]["family"] for c in cases})
    method_curves(sig, curves, "noise sigma", ["PSNR gain over blurred (dB)"] * len(fams),
                  [f"{f} kernels" for f in fams], out / "overall_gain.png",
                  hline={i: 0.0 for i in range(len(fams))}, style=STYLE)


def examples(root, cfg, smoke, primary, frozen, out):
    """Full images + zoomed 192 px centre crops for a few test cases of the first image (lowest noise
    level and 0.01); PNGs in comparison/images/."""
    from experiments.step8_common import make_cases
    params = step3_params()
    images = load_test_images(smoke)
    name, x = images[0]
    kernels = family_kernels(cfg["test_kernels"], "test")
    kd = {kid: k for _, kid, k in kernels}
    seeds = {(c["kernel_id"], c["sigma"]): c["seed"] for c in make_cases(images[:1], kernels, cfg["sigmas"], "test")}
    picks = [kid for kid in ("curved_0", "traj_0", "line_t75_L15") if kid in kd] or list(kd)[:2]
    methods = ["rl", primary["8a"], primary["8b"]]
    imgdir = out / "images"
    fig_rows = []
    for s in [cfg["sigmas"][0], 0.01 if 0.01 in cfg["sigmas"] else cfg["sigmas"][-1]]:
        for kid in picks:
            kt = kd[kid]
            y = make_blurred(x, kt, s, np.random.default_rng(seeds[(kid, s)]))   # same case as in 8test
            th, L = estimate_line_kernel(y)
            k = line_kernel(th, L)
            rl_out = rl_with_checkpoints(y, k, rl_iters(s, params))
            cond = f"{kid}_s{s:g}"
            save_png(x, imgdir / f"{name}__sharp.png")
            save_png(y, imgdir / f"{name}__{cond}__blurred.png")
            save_kernel_png(kt, imgdir / "kernels" / f"{cond}__true.png")
            save_kernel_png(k, imgdir / "kernels" / f"{cond}__estimated.png")
            row = [(y, f"{cond}: blurred", kt)]
            for m in methods:
                xh = apply(m, y, k, frozen["per_sigma"][str(s)].get(m, {}), rl_out)
                save_png(xh, imgdir / f"{name}__{cond}__{m}.png")
                row.append((xh, f"{LABEL[m]}", k if m == "rl" else None))
            row.append((x, "sharp (truth)", None))
            fig_rows.append(row)
    n, c0 = x.shape[0], x.shape[0] // 2 - 96
    from deblur.plotting import image_grid
    image_grid(fig_rows, out / "examples_full.png", cell=3.0, title_size=7)
    crop = [[(im[c0:c0 + 192, c0:c0 + 192], t, None) for im, t, _ in row] for row in fig_rows]
    image_grid(crop, out / "examples_zoom.png", cell=3.0, title_size=7)


def run(cfg, root, workers, log, smoke=False):
    by = load(root)
    frozen = json.loads((root / "tune" / "frozen_settings.json").read_text())
    primary = frozen["primary"]
    comp = new_run_dir(root / "comparison")
    suffix = comp.name[len("comparison"):]
    dirs = {"8a": root / f"8a_postprocess{suffix}", "8b": root / f"8b_prior_deconv{suffix}", "cmp": comp}
    for d in dirs.values():
        d.mkdir(parents=True)
    for br, title in (("8a", "8a post-processing"), ("8b", "8b prior deconvolution")):
        t = branch_table(by, BRANCH[br], "estimated") + branch_table(by, BRANCH[br], "true")
        write_csv(dirs[br] / "table.csv", t)
        branch_figures(t, BRANCH[br], dirs[br], title)
    hyp = hypotheses(by, primary)
    write_csv(comp / "hypotheses.csv", hyp)
    overall_figure(by, primary, comp)
    examples(root, cfg, smoke, primary, frozen, comp)

    t_all = branch_table(by, BRANCH["8a"] + BRANCH["8b"], "estimated") + branch_table(by, BRANCH["8a"] + BRANCH["8b"], "true")
    write_csv(comp / "all_methods_table.csv", t_all)
    lines = ["STEP 8 SUMMARY (generated by step8_analyze.py; protocol in PROTOCOL.txt)", "",
             f"cases: {len(by)}; primary methods (chosen on dev): 8a = {LABEL[primary['8a']]}, 8b = {LABEL[primary['8b']]}", "",
             "HYPOTHESES (estimated kernel; one-sided; Holm over five)"]
    for h in hyp:
        lines.append(f"  {h['id']:6s} {h['verdict']:13s} {h['hypothesis']}")
        lines.append(f"         {h['metric']}: {h['mean_diff']:+.3f} dB [{h['ci_lo']:+.3f}, {h['ci_hi']:+.3f}], win {h['win_rate']:.0%}, "
                     f"p(images) {h['p_wilcoxon_images']:.4f}, Holm {h['p_holm']:.4f} | edge PSNR {h['psnr_edge_diff']:+.2f} dB, "
                     f"SSIM {h['ssim_diff']:+.4f}")
    lines += ["", "ALL METHODS, all families (mean over cases): PSNR gain over blurred | vs RL: PSNR, flat, edge | win vs RL"]
    for kernel in ("estimated", "true"):
        for s in sorted({r["sigma"] for r in t_all if r["sigma"] != "all"}) + ["all"]:
            lines.append(f"  {kernel} kernel, sigma={s}")
            for r in t_all:
                if r["kernel"] == kernel and r["family"] == "all" and r["sigma"] == s:
                    lines.append(f"    {r['method']:14s} {r['psnr_vs_blurred']:+6.2f} | {r['psnr_vs_rl']:+6.2f} {r['psnr_flat_vs_rl']:+6.2f} "
                                 f"{r['psnr_edge_vs_rl']:+6.2f} | {r['win_vs_rl']:.0%}")
    (comp / "summary.txt").write_text("\n".join(lines) + "\n")
    for l in lines:
        log.info(l)
    return f"analysis in {comp.name}"
